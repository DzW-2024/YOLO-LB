import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from thop import profile
from thop import clever_format
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics import Precision, Recall, Accuracy
import warnings
from YOLO_LB import YOLO_LB
from PoseDataset import PoseDataset

warnings.filterwarnings('ignore')


def calculate_model_complexity(model, input_shape=(3, 640, 640), device='cuda'):
    """Calculate the parameters and FLOPs"""
    model.eval()
    input_tensor = torch.randn(1, *input_shape).to(device)

    # Calculate the parameters and FLOPs
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "Parameters": f"{total_params / 1e6:.3f}M",
        "Trainable_parameters": f"{trainable_params / 1e6:.3f}M",
        "FLOPs": flops,
        "Input_shape": input_shape
    }


def compute_detection_metrics(model, val_loader, device, conf_threshold=0.5, iou_threshold=0.5):
    """Metrics: Precision, Recall, Accuracy, mAP (at the bounding box level)"""
    model.eval()
    metric_map = MeanAveragePrecision(class_metrics=True).to(device)
    metric_precision = Precision(task="binary").to(device)
    metric_recall = Recall(task="binary").to(device)
    metric_accuracy = Accuracy(task="binary").to(device)

    all_tp = 0
    all_fp = 0
    all_fn = 0

    with torch.no_grad():
        for imgs, targets, orig_w, orig_h in tqdm(val_loader, desc="计算检测指标"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            orig_w = orig_w.to(device)
            orig_h = orig_h.to(device)

            # Model inference
            preds = model(imgs)  # (B, N, 5 + 2K)：x1,y1,x2,y2,conf,kpts...

            for b in range(imgs.shape[0]):
                # Processing the predictions and true labels of individual images
                pred_b = preds[b]
                target_b = targets[b]
                w = orig_w[b].item()
                h = orig_h[b].item()

                # Filter out low-confidence predictions
                pred_valid = pred_b[pred_b[..., 4] > conf_threshold]
                if len(pred_valid) == 0:
                    # There are no prediction boxes, and all the real boxes are false negatives.
                    all_fn += len(target_b)
                    continue

                # Convert the prediction box to pixel coordinates
                pred_boxes = pred_valid[..., :4] * torch.tensor([w, h, w, h], device=device)
                pred_scores = pred_valid[..., 4]
                pred_classes = torch.zeros(len(pred_valid), dtype=torch.int32).to(device)

                # Real box extraction (x1, y1, x2, y2, class_id)
                if len(target_b) > 0:
                    true_boxes = target_b[..., :4]
                    true_classes = target_b[..., 4].int()
                else:
                    true_boxes = torch.empty((0, 4)).to(device)
                    true_classes = torch.empty((0,)).to(device)

                # Construct the input in the format of torchmetrics
                preds_dict = {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_classes
                }
                targets_dict = {
                    "boxes": true_boxes,
                    "labels": true_classes
                }

                metric_map.update([preds_dict], [targets_dict])

                # Calculate TP/FP/FN (based on IoU matching)
                if len(true_boxes) == 0:
                    all_fp += len(pred_valid)
                    continue

                if len(pred_boxes) == 0:
                    all_fn += len(true_boxes)
                    continue

                # Calculate the IoU matrix of the predicted bounding box and the real bounding box
                iou_matrix = torch.zeros((len(pred_boxes), len(true_boxes)), device=device)
                for i, pb in enumerate(pred_boxes):
                    for j, tb in enumerate(true_boxes):
                        iou_matrix[i, j] = box_iou(pb.unsqueeze(0), tb.unsqueeze(0)).item()

                # Match the predicted bounding box with the real bounding box
                matched_true = torch.zeros(len(true_boxes), dtype=torch.bool).to(device)
                tp = 0
                fp = 0
                for i in range(len(pred_boxes)):
                    max_iou, max_j = iou_matrix[i].max(dim=0)
                    if max_iou > iou_threshold and not matched_true[max_j]:
                        tp += 1
                        matched_true[max_j] = True
                    else:
                        fp += 1
                fn = len(true_boxes) - matched_true.sum()

                # Update global statistics
                all_tp += tp
                all_fp += fp
                all_fn += fn

                # Update the Precision/Recall/Accuracy metrics
                # TP=1, FP=0
                y_pred = torch.cat([torch.ones(tp), torch.zeros(fp)]).to(device)
                # Among the samples predicted as positive, the true positive examples = TP, and the negative examples = FP.
                y_true = torch.cat([torch.ones(tp), torch.zeros(fp)]).to(device)
                if len(y_pred) > 0:
                    metric_precision.update(y_pred, y_true)
                    metric_recall.update(y_pred, y_true)
                    metric_accuracy.update(y_pred, y_true)

    # Calculate the final metric
    map_result = metric_map.compute()
    precision = metric_precision.compute().item() if metric_precision.total != 0 else 0.0
    recall = metric_recall.compute().item() if metric_recall.total != 0 else 0.0
    accuracy = metric_accuracy.compute().item() if metric_accuracy.total != 0 else 0.0

    # Calculate precision/recall/accuracy
    manual_precision = all_tp / (all_tp + all_fp + 1e-9)
    manual_recall = all_tp / (all_tp + all_fn + 1e-9)
    manual_accuracy = all_tp / (all_tp + all_fp + all_fn + 1e-9)

    return {
        "mAP@0.5": map_result["map"].item(),
        "mAP@0.5:0.95": map_result["map_50_95"].item(),
        "Precision": max(precision, manual_precision),
        "Recall": max(recall, manual_recall),
        "Accuracy": max(accuracy, manual_accuracy),
        "TP": all_tp,
        "FP": all_fp,
        "FN": all_fn
    }


def compute_keypoint_metrics(model, val_loader, device, conf_threshold=0.5, iou_threshold=0.5, pck_threshold=0.05):
    """Calculate keypoint metrics: PCK@0.05, PCK@0.1"""
    model.eval()
    total_visible_kpts = 0
    correct_kpts_pck5 = 0
    correct_kpts_pck10 = 0

    with torch.no_grad():
        for imgs, targets, orig_w, orig_h in tqdm(val_loader, desc="Calculate keypoint metrics"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            orig_w = orig_w.to(device)
            orig_h = orig_h.to(device)

            preds = model(imgs)  # (B, N, 5 + 2K)

            for b in range(imgs.shape[0]):
                pred_b = preds[b]
                target_b = targets[b]
                w = orig_w[b].item()
                h = orig_h[b].item()
                # Diagonal length of the image (PCK threshold reference)
                img_diag = np.sqrt(w ** 2 + h ** 2)

                # Filter out low-confidence predictions
                pred_valid = pred_b[pred_b[..., 4] > conf_threshold]
                if len(pred_valid) == 0 or len(target_b) == 0:
                    continue

                # Matching of predicted bounding box and true bounding box
                pred_boxes = pred_valid[..., :4] * torch.tensor([w, h, w, h], device=device)
                true_boxes = target_b[..., :4]
                iou_matrix = torch.zeros((len(pred_boxes), len(true_boxes)), device=device)

                for i, pb in enumerate(pred_boxes):
                    for j, tb in enumerate(true_boxes):
                        iou_matrix[i, j] = box_iou(pb.unsqueeze(0), tb.unsqueeze(0)).item()

                # Find the best-matched predicted box and the corresponding real box
                max_iou, pred_idx = iou_matrix.max(dim=0)
                valid_matches = max_iou > iou_threshold

                if not valid_matches.any():
                    continue

                # Extract the matching predicted keypoints and the corresponding real keypoints
                for true_idx in torch.where(valid_matches)[0]:
                    p_idx = pred_idx[true_idx]

                    # Predict keypoints (normalized → pixel coordinates)
                    pred_kpts = pred_valid[p_idx, 5:].view(-1, 2) * torch.tensor([w, h], device=device)
                    # The true keypoints and visibility
                    true_kpts = target_b[true_idx, 5:5 + model.num_keypoints * 2].view(-1, 2)
                    true_kpts_v = target_b[true_idx, 5 + model.num_keypoints * 2:5 + model.num_keypoints * 3]

                    # Only calculate the visible keypoints
                    visible_mask = true_kpts_v > 0
                    if not visible_mask.any():
                        continue

                    total_visible_kpts += visible_mask.sum().item()

                    # Calculate the distance between keypoints
                    dist = torch.norm(pred_kpts - true_kpts, dim=1)

                    # PCK@0.05: Distance < 5% of the diagonal of the image
                    correct_kpts_pck5 += (dist[visible_mask] < img_diag * pck_threshold).sum().item()
                    # PCK@0.1: Distance < 10% of the diagonal of the image
                    correct_kpts_pck10 += (dist[visible_mask] < img_diag * pck_threshold * 2).sum().item()

    # Calculate the PCK index
    pck5 = correct_kpts_pck5 / (total_visible_kpts + 1e-9)
    pck10 = correct_kpts_pck10 / (total_visible_kpts + 1e-9)

    return {
        "PCK@0.05": pck5,
        "PCK@0.1": pck10,
        "Total_visible_kpts": total_visible_kpts,
        "PCK@0.05_correct_kpts": correct_kpts_pck5,
        "PCK@0.1_correct_kpts": correct_kpts_pck10
    }


def box_iou(box1, box2):
    """Calculate the IoU of two bounding boxes (x1, y1, x2, y2)"""
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    return inter_area / (box1_area + box2_area - inter_area + 1e-9)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_keypoints = 17
    img_size = 640

    model = YOLO_LB(num_classes=1, num_keypoints=num_keypoints).to(device)

    val_dataset = PoseDataset(
        img_dir="images",
        ann_dir="labels",
        img_size=img_size,
        num_keypoints=num_keypoints
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Calculate the complexity of the model (number of parameters + FLOPs)
    print("=" * 50)
    complexity = calculate_model_complexity(model, input_shape=(3, img_size, img_size), device=device)
    print("Model complexity metric:")
    for k, v in complexity.items():
        print(f"  {k}: {v}")

    # Calculate the evaluation metrics (Precision, Recall, Accuracy, mAP)
    print("\n" + "=" * 50)
    detection_metrics = compute_detection_metrics(model, val_loader, device, conf_threshold=0.5, iou_threshold=0.5)
    print("Detection metrics (box): ")
    for k, v in detection_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Calculate the keypoint indicators (PCK)
    print("\n" + "=" * 50)
    keypoint_metrics = compute_keypoint_metrics(model, val_loader, device, conf_threshold=0.5, iou_threshold=0.5)
    print("Keypoint metrics (pose):")
    for k, v in keypoint_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 50)