import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import os


class PoseDataset(Dataset):
    # class cx cy w h k1x k1y k1v k2x k2y k2v k3x k3y k3v
    def __init__(self, img_dir, ann_dir, img_size=640, num_keypoints=3):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_size = img_size
        self.num_keypoints = num_keypoints
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Read images
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Read annotation
        ann_path = os.path.join(self.ann_dir, img_name.replace(os.path.splitext(img_name)[1], '.txt'))
        targets = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = list(map(float, line.split()))
                    # Length verification: class(1) + bbox(4) + kpt(x,y,v)*K → 1+4+3K
                    if len(parts) != 1 + 4 + 3 * self.num_keypoints:
                        continue

                        # YOLO: class cx cy w h (Norm) → Convert to pixel coordinates
                        class_id = parts[0]
                        cx = parts[1] * orig_w  # Restore to the original image x-coordinate
                        cy = parts[2] * orig_h  # Restore to the original image y-coordinate
                        w = parts[3] * orig_w  # Restore to the original image width
                        h = parts[4] * orig_h  # Restore to the original image height
                        # Convert to x1, y1, x2, y2 (pixel coordinates)
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2

                        # Keypoints: x and y are normalized and restored to the original image coordinates; v is retained
                        kpts = []
                        for k in range(self.num_keypoints):
                            kx = parts[5 + 3 * k] * orig_w  # Restore x
                            ky = parts[6 + 3 * k] * orig_h  # Restore y
                            kv = parts[7 + 3 * k]  # Remain visibility
                            kpts.extend([kx, ky, kv])

                        # Joined label:x1,y1,x2,y2,class_id,kpt1x,kpt1y,kpt1v,...
                        target = [x1, y1, x2, y2, class_id] + kpts
                        targets.append(target)

        # Convert to tensor (shape: (N, 5 + 3K))
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros(0, 5 + 3 * self.num_keypoints)
        return img, targets, orig_w, orig_h
