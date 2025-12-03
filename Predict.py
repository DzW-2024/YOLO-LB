import torch
import cv2
import os
from YOLO_LB import YOLO_LB


def inference(model, img_path, ann_path=None, device=None, conf_threshold=0.5, iou_threshold=0.5, num_keypoints=17):
    """Inference function: Supports drawing key points based on the "v" in the label (if ann_path is provided)"""
    model.eval()
    # Read image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0.unsqueeze(0).to(device)

    # Read the "v" in the label (used for filtering out invisible keypoints)
    kpt_visibility = None
    if ann_path and os.path.exists(ann_path):
        with open(ann_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 1 + 4 + 3 * num_keypoints:
                    kpt_visibility = [parts[5 + 3 * k + 2] for k in range(num_keypoints)]
                    break

    # Infer
    with torch.no_grad():
        preds = model(img_tensor)[0]  # (B, N, 5 + 2K)

    # Filter out low-confidence predictions
    mask = preds[..., 4] > conf_threshold
    preds = preds[mask]
    if preds.shape[0] == 0:
        return img

    # NMS
    boxes = preds[..., :4] * torch.tensor([w, h, w, h], device=device)
    scores = preds[..., 4]
    keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    preds = preds[keep]

    for pred in preds:
        x1, y1, x2, y2 = map(int, pred[:4].clamp(0, max(w, h)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        kpts = pred[5:].view(-1, 2) * torch.tensor([w, h], device=device)
        for k in range(num_keypoints):
            kx, ky = map(int, kpts[k].clamp(0, max(w, h)))
            if kpt_visibility is None or kpt_visibility[k] > 0:
                cv2.circle(img, (kx, ky), 3, (0, 0, 255), -1)
                connections = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11), (12, 13), (13, 14),
                               (15, 16)]
                if k < len(connections):
                    k1, k2 = connections[k]
                    if k1 < num_keypoints and k2 < num_keypoints:
                        if kpt_visibility is None or (kpt_visibility[k1] > 0 and kpt_visibility[k2] > 0):
                            kx1, ky1 = map(int, kpts[k1].clamp(0, max(w, h)))
                            kx2, ky2 = map(int, kpts[k2].clamp(0, max(w, h)))
                            cv2.line(img, (kx1, ky1), (kx2, ky2), (255, 0, 0), 2)

    return img

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_keypoints = 3

    # Initialize the model
    model = YOLO_LB(num_classes=1, num_keypoints=num_keypoints).to(device)

    img = inference(
        model=model,
        img_path="path/to/test.jpg",
        ann_path="path/to/test.txt",  # Optional: Used for obtaining the visibility of keypoints
        device=device,
        conf_threshold=0.5,
        iou_threshold=0.5,
        num_keypoints=num_keypoints
    )
    cv2.imwrite("result.jpg", img)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()