import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import DGCNNSemSeg
from dataset import PointCloudDataset

class LivePlotter:
    def __init__(self, title='Training Progress'):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.suptitle(title)
        self.axes[0].set_xlabel('Epoch'); self.axes[0].set_ylabel('Loss'); self.axes[0].set_title('Loss vs. Epochs')
        self.axes[1].set_xlabel('Epoch'); self.axes[1].set_ylabel('Mean IoU'); self.axes[1].set_title('mIoU vs. Epochs')
        self.train_loss_history, self.val_loss_history = [], []
        self.train_miou_history, self.val_miou_history = [], []

    def update(self, epoch, train_loss, val_loss, train_miou, val_miou):
        self.train_loss_history.append(train_loss); self.val_loss_history.append(val_loss)
        self.train_miou_history.append(train_miou); self.val_miou_history.append(val_miou)
        epochs = range(1, epoch + 2)
        self.axes[0].cla(); self.axes[0].plot(epochs, self.train_loss_history, 'b-', label='Train Loss'); self.axes[0].plot(epochs, self.val_loss_history, 'r-', label='Validation Loss'); self.axes[0].legend(); self.axes[0].grid(True)
        self.axes[1].cla(); self.axes[1].plot(epochs, self.train_miou_history, 'b-', label='Train mIoU'); self.axes[1].plot(epochs, self.val_miou_history, 'r-', label='Validation mIoU'); self.axes[1].legend(); self.axes[1].grid(True)
        plt.draw(); plt.pause(0.1)

def calculate_miou(pred, target, num_classes):
    iou_list = []
    pred, target = pred.cpu().numpy().flatten(), target.cpu().numpy().flatten()
    for cl in range(num_classes):
        intersection = np.sum((pred == cl) & (target == cl))
        union = np.sum((pred == cl) | (target == cl))
        if union == 0: continue
        iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0.0

def main(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    plotter = LivePlotter()

    os.makedirs(args.checkpoint_path, exist_ok=True)

    print("Loading data...")
    train_dataset = PointCloudDataset(root_dir=args.data_path, num_points=args.num_points, split='train')
    val_dataset = PointCloudDataset(root_dir=args.data_path, num_points=args.num_points, split='val', augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    print("Data loaded.")

    model = DGCNNSemSeg(num_classes=args.num_classes, num_features=args.num_features, k=args.k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss, epoch_train_miou = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for points, labels in pbar:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(points)
            pred_flat, labels_flat = pred.view(-1, args.num_classes), labels.view(-1)
            loss = criterion(pred_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            with torch.no_grad():
                miou = calculate_miou(pred.argmax(dim=-1), labels, args.num_classes)
            epoch_train_miou += miou
            pbar.set_postfix({'loss': loss.item(), 'mIoU': miou})

        avg_train_loss, avg_train_miou = epoch_train_loss / len(train_loader), epoch_train_miou / len(train_loader)
        scheduler.step()

        model.eval()
        epoch_val_loss, epoch_val_miou = 0.0, 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for points, labels in pbar_val:
                points, labels = points.to(device), labels.to(device)
                pred = model(points)
                pred_flat, labels_flat = pred.view(-1, args.num_classes), labels.view(-1)
                loss = criterion(pred_flat, labels_flat)
                epoch_val_loss += loss.item()
                miou = calculate_miou(pred.argmax(dim=-1), labels, args.num_classes)
                epoch_val_miou += miou
                pbar_val.set_postfix({'loss': loss.item(), 'mIoU': miou})

        avg_val_loss, avg_val_miou = epoch_val_loss / len(val_loader), epoch_val_miou / len(val_loader)
        plotter.update(epoch, avg_train_loss, avg_val_loss, avg_train_miou, avg_val_miou)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train mIoU: {avg_train_miou:.4f} | Val Loss: {avg_val_loss:.4f}, Val mIoU: {avg_val_miou:.4f}")

        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            print(f"New best model found! Saving to {args.checkpoint_path}/best_model.pth")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, 'best_model.pth'))
            
    plt.savefig('training_progress.png')
    print("Training complete. Final plot saved as training_progress.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DGCNN for semantic segmentation.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the processed data directory.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Path to save model checkpoints.')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes for segmentation.')
    parser.add_argument('--num_features', type=int, default=7, help='Number of input features.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points to sample per block.')
    parser.add_argument('--k', type=int, default=20, help='Number of nearest neighbors for DGCNN.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training.')
    
    args = parser.parse_args()
    main(args)
