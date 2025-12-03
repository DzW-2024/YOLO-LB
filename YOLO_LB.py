from Convs import Conv, DWConv, GhostConv, SCConv
from C3k2 import C3k2
from SPPF import SPPF
from C2PSA import C2PSA
from Attentions import *


class Backbone(nn.Module):
    def __init__(self, scales=[0.50, 0.25, 1024]):
        super().__init__()
        depth_scale, width_scale, max_channels = scales
        self.get_channels = lambda c: min(int(c * width_scale), max_channels)

        self.backbone = nn.Sequential(
            Conv(3, self.get_channels(64), 3, 2),
            Conv(self.get_channels(64), self.get_channels(128), 3, 2),
            C3k2(self.get_channels(128), self.get_channels(256), 2, False, 0.25),
            Conv(self.get_channels(256), self.get_channels(256), 3, 2),
            C3k2(self.get_channels(256), self.get_channels(512), 2, False, 0.25),
            Conv(self.get_channels(512), self.get_channels(512), 3, 2),
            C3k2(self.get_channels(512), self.get_channels(512), 2, True),
            Conv(self.get_channels(512), self.get_channels(1024), 3, 2),
            C3k2(self.get_channels(1024), self.get_channels(1024), 2, True),
            SPPF(self.get_channels(1024), self.get_channels(1024), 5),
            MobileMQA(self.get_channels(1024)),
            C2PSA(self.get_channels(1024), self.get_channels(1024), 2)
        )

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [1, 3, 5, 10]:
                outputs.append(x)
        return outputs


class PoseHead(nn.Module):
    def __init__(self, in_channels_list, num_classes=1, num_keypoints=3, anchors=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

        self.num_anchors = anchors.shape[1] if anchors is not None else 3
        self.out_channels = self.num_anchors * (num_classes + 4 + num_keypoints * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.down1 = SCConv(in_channels_list[0], in_channels_list[0])
        self.down2 = SCConv(in_channels_list[1], in_channels_list[1])

        self.cv1 = C3k2(in_channels_list[2] + in_channels_list[3], in_channels_list[2], 2, False)
        self.cv2 = C3k2(in_channels_list[1] + in_channels_list[2], in_channels_list[1], 2, False)
        self.cv3 = C3k2(in_channels_list[1] + in_channels_list[2], in_channels_list[2], 2, False)
        self.cv4 = C3k2(in_channels_list[2] + in_channels_list[3], in_channels_list[3], 2, False)

        self.predict1 = Conv(in_channels_list[2], self.out_channels, 1, 1, 0)
        self.predict2 = Conv(in_channels_list[3], self.out_channels, 1, 1, 0)
        self.predict3 = Conv(in_channels_list[3], self.out_channels, 1, 1, 0)

    def forward(self, x):
        P2, P3, P4, P5 = x

        up1 = self.up1(P5)
        cat1 = torch.cat([up1, P4], dim=1)
        out1 = self.cv1(cat1)

        up2 = self.up2(out1)
        cat2 = torch.cat([up2, P3], dim=1)
        out2 = self.cv2(cat2)

        down1 = self.down1(out2)
        cat3 = torch.cat([down1, out1], dim=1)
        out3 = self.cv3(cat3)

        down2 = self.down2(out3)
        cat4 = torch.cat([down2, P5], dim=1)
        out4 = self.cv4(cat4)

        pred3 = self.predict2(out3)
        pred4 = self.predict3(out4)
        pred5 = self.predict4(P5)

        return [pred3, pred4, pred5]


class YOLO_LB(nn.Module):
    def __init__(self, num_classes=1, num_keypoints=17):
        super().__init__()
        self.backbone_out_channels = [32, 64, 128, 256]
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
            [[326, 373], [459, 401], [512, 512]]
        ]
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32)

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

        self.backbone = Backbone()
        self.head = PoseHead(self.backbone_out_channels, num_classes, num_keypoints, self.anchors)

    def forward(self, x):
        features = self.backbone(x)
        preds = self.head(features)
        if self.training:
            return preds
        else:
            return self.decode_preds(preds, x.shape)

    def decode_preds(self, preds, input_shape):
        """
        Decode the prediction results: Only predict (x, y), and v is obtained from the labels
        (when reasoning, all predicted key points are defaulted to be plotted)
        """
        device = preds[0].device
        batch_size = preds[0].shape[0]
        output = []

        for i, (pred, anchors) in enumerate(zip(preds, self.anchors)):
            # pred shape: (B, 39, H, W) → (B, H, W, 3, 13)
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(B, H, W, self.num_anchors, -1)

            # Decode coordinates
            grid_x = torch.arange(W, device=device).repeat(H, 1).unsqueeze(-1)
            grid_y = torch.arange(H, device=device).repeat(W, 1).t().unsqueeze(-1)
            grid = torch.cat([grid_x, grid_y], dim=-1).float()

            # Bounding box decoding
            box_xy = (torch.sigmoid(pred[..., 0:2]) + grid) / W  # to 0-1
            box_wh = torch.exp(pred[..., 2:4]) * anchors.to(device) / input_shape[2:]
            box_corner = torch.cat([box_xy - box_wh / 2, box_xy + box_wh / 2], dim=-1)  # (x1,y1,x2,y2)

            # Classification score
            conf = torch.sigmoid(pred[..., 4:5])

            # Keypoint decoding
            kpts = torch.sigmoid(pred[..., 5:]) * 2.0 - 0.5  # to -0.5~1.5
            kpts = kpts.view(B, H, W, self.num_anchors, self.num_keypoints, 2)
            kpts[..., 0] = (kpts[..., 0] + grid_x) / W  # x-coordinate normalization
            kpts[..., 1] = (kpts[..., 1] + grid_y) / H  # y-coordinate normalization

            # The result of the concatenation
            pred_i = torch.cat([box_corner, conf, kpts.view(B, H, W, self.num_anchors, -1)], dim=-1)
            output.append(pred_i.reshape(B, -1, pred_i.shape[-1]))

        return torch.cat(output, dim=1)
