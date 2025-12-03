import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, anchors, num_classes=1, num_keypoints=17, img_size=640):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.img_size = img_size
        self.balance = [4.0, 1.0, 0.4, 0.1]  # Weights
        self.box_loss = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        """
        preds: Output, list of 4 tensors (P2-P5)
        targets: Label, shape (B, N, 5 + 2*K) → (x1,y1,x2,y2,conf,kpt1_x,kpt1_y,...)
        """
        device = preds[0].device
        B = preds[0].shape[0]
        total_loss = 0.0

        for i, (pred, anchors) in enumerate(zip(preds, self.anchors)):
            # Prediction results reshape
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(B, H, W, 3, -1)

            # Generate grid
            grid_x = torch.arange(W, device=device).repeat(H, 1).unsqueeze(-1)
            grid_y = torch.arange(H, device=device).repeat(W, 1).t().unsqueeze(-1)
            grid = torch.cat([grid_x, grid_y], dim=-1).float()

            # Decode bbox
            box_xy = torch.sigmoid(pred[..., 0:2]) + grid
            box_wh = torch.exp(pred[..., 2:4]) * anchors.to(device)
            box_pred = torch.cat([box_xy / W * self.img_size, box_wh], dim=-1)

            # Match positive and negative samples
            iou, target_box, target_conf, target_kpts, target_kpts_v = self.match_targets(box_pred, targets, H, W)

            # Caculate loss
            loss_box = self.balance[i] * self.compute_box_loss(box_pred, target_box, iou)
            loss_conf = self.balance[i] * F.binary_cross_entropy_with_logits(pred[..., 4], target_conf)
            loss_kpts = self.balance[i] * self.compute_kpt_loss(pred[..., 5:], target_kpts, target_kpts_v, target_conf)

            total_loss += loss_box + loss_conf + loss_kpts

        return total_loss / B

    def match_targets(self, box_pred, targets, H, W):
        """Match positive and negative samples (Extract x, y, v)"""
        B = box_pred.shape[0]
        iou = torch.zeros(B, H, W, 3, device=box_pred.device)
        target_box = torch.zeros(B, H, W, 3, 4, device=box_pred.device)
        target_conf = torch.zeros(B, H, W, 3, device=box_pred.device)
        target_kpts = torch.zeros(B, H, W, 3, self.num_keypoints * 2, device=box_pred.device)  # x,y
        target_kpts_v = torch.zeros(B, H, W, 3, self.num_keypoints, device=box_pred.device)  # visibility

        for b in range(B):
            if targets[b].shape[0] == 0:
                continue
            # (x1,y1,x2,y2) converted to (cx,cy,w,h)
            t_box = self.box_corner_to_center(targets[b][:, :4])
            # Pre bbox (cx,cy,w,h)
            p_box = self.box_corner_to_center(box_pred[b].reshape(-1, 4))

            # Calculate IoU
            iou_b = self.compute_iou(p_box, t_box).reshape(H, W, 3, -1)
            best_iou, best_idx = iou_b.max(dim=-1)

            # Match positive samples
            mask = best_iou > 0.5
            if mask.any():
                iou[b][mask] = best_iou[mask]
                target_box[b][mask] = t_box[best_idx[mask]]
                target_conf[b][mask] = 1.0

                # Keypoint x, y
                target_kpts[b][mask] = targets[b][best_idx[mask], 5:5 + self.num_keypoints * 2].repeat_interleave(3,
                                                                                                                  dim=0).reshape(
                    -1, self.num_keypoints * 2)

                # Keypoint visibility
                target_kpts_v[b][mask] = targets[b][best_idx[mask],
                                         5 + self.num_keypoints * 2:5 + self.num_keypoints * 3].repeat_interleave(3,
                                                                                                                  dim=0).reshape(
                    -1, self.num_keypoints)

        return iou, target_box, target_conf, target_kpts, target_kpts_v

    def compute_iou(self, box1, box2):
        """Calculate IoU"""
        box1 = self.box_center_to_corner(box1)
        box2 = self.box_center_to_corner(box2)
        return self.iou(box1, box2)

    def box_center_to_corner(self, box):
        """(cx,cy,w,h) → (x1,y1,x2,y2)"""
        x1 = box[..., 0] - box[..., 2] / 2
        y1 = box[..., 1] - box[..., 3] / 2
        x2 = box[..., 0] + box[..., 2] / 2
        y2 = box[..., 1] + box[..., 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def box_corner_to_center(self, box):
        """（x1,y1,x2,y2）→（cx,cy,w,h）"""
        cx = (box[..., 0] + box[..., 2]) / 2
        cy = (box[..., 1] + box[..., 3]) / 2
        w = box[..., 2] - box[..., 0]
        h = box[..., 3] - box[..., 1]
        return torch.stack([cx, cy, w, h], dim=-1)

    def iou(self, box1, box2):
        """Calculate IoU (x1,y1,x2,y2)"""
        inter_area = self.intersection(box1, box2)
        union_area = self.union(box1, box2)
        return inter_area / (union_area + 1e-9)

    def intersection(self, box1, box2):
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])
        return (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    def union(self, box1, box2):
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        return area1 + area2 - self.intersection(box1, box2)

    def compute_box_loss(self, pred, target, iou):
        """Calculate the bounding box loss"""
        pred = self.box_center_to_corner(pred)
        target = self.box_center_to_corner(target)
        return (1.0 - self.iou(pred, target)).mean()

    def compute_kpt_loss(self, pred, target, target_v, mask):
        """Calculate the key point loss: Only calculate the loss for the key points where v > 0."""
        # pred: (B, H, W, 3, 2K), target: (B, H, W, 3, 2K), target_v: (B, H, W, 3, K), mask: (B, H, W, 3)
        B, H, W, A, _ = pred.shape

        # Generate keypoint mask: v > 0 and the bounding box is a positive sample (mask = 1)
        kpt_mask = (target_v > 0).float()  # (B, H, W, 3, K)
        kpt_mask = kpt_mask.unsqueeze(-1).repeat(1, 1, 1, 1, 1, 2).reshape(B, H, W, A, -1)  # 扩展到x,y维度
        global_mask = mask.unsqueeze(-1)  # (B, H, W, 3, 1)
        total_mask = kpt_mask * global_mask  # Only the visible keypoints of the positive samples

        # Calculate the L1 loss (only for valid key points)
        if total_mask.sum() > 0:
            return F.l1_loss(pred * total_mask, target * total_mask) / total_mask.sum()
        else:
            return torch.tensor(0.0, device=pred.device)
