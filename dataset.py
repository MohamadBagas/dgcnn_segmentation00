import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pre-processed point cloud blocks.
    """
    def __init__(self, root_dir, num_points=4096, split='train', augment=True):
        self.root_dir = os.path.join(root_dir, split)
        self.file_list = [f for f in os.listdir(self.root_dir) if f.endswith('.npy')]
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        
        data = np.load(file_path)
        
        points = data[:, :-1]
        labels = data[:, -1]

        if len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        
        points = points[choice, :]
        labels = labels[choice]

        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])
            points[:, :2] = points[:, :2].dot(rotation_matrix)

            scale = np.random.uniform(0.95, 1.05)
            points[:, :3] *= scale

            jitter = np.random.normal(0, 0.02, size=points[:, :3].shape)
            points[:, :3] += jitter

        center = np.mean(points[:, :3], axis=0)
        points[:, :3] -= center

        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        return points, labels
