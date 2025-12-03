import os
import torch
from Loss import Loss
from YOLO_LB import YOLO_LB
from PoseDataset import PoseDataset
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm



def train(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, total=len(train_loader))
        for imgs, targets in loop:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward
            preds = model(imgs)
            loss = criterion(preds, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item())


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_keypoints = 3
    # Initialize model
    model = YOLO_LB(num_classes=1, num_keypoints=num_keypoints).to(device)
    # Dataset (replace with the actual path)
    train_dataset = PoseDataset(
        img_dir="images",
        ann_dir="labels",
        img_size=640,
        num_keypoints=num_keypoints
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    # Loss function
    criterion = Loss(model.anchors, num_classes=1, num_keypoints=num_keypoints, img_size=640).to(device)
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    train(model, train_loader, criterion, optimizer, epochs=2000, device=device)
