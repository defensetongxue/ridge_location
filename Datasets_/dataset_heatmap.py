import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import json
from .transforms_kit import *
class KeypointDetectionDatasetHeatmap(Dataset):
    def __init__(self, data_path, split='train', heatmap_rate=0.25, sigma=1.5,resize=[600,800]):
        self.data_path = data_path
        self.heatmap_rate = heatmap_rate
        self.sigma = sigma
        self.annotations = json.load(open(os.path.join(data_path, 'ridge', f"{split}.json")))

        if split=='train':
            self.transform=KeypointDetectionTransformHeatmap(mode='train',size=resize)
        elif split=='val' or split=='test':
            self.transform=KeypointDetectionTransformHeatmap(mode='val',size=resize)
        else:
            raise ValueError(
                f"Invalid split: {split}, split should be one of train|valid|test")

    def __len__(self):
        return len(self.annotations)

    def generate_target(self, img, pt, sigma, label_type='Gaussian', img_path=''):
        tmp_size = sigma * 3
        ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
        br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
        if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
            raise ValueError(f"Invalid boundary point {pt} in image {img_path}")

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2

        if label_type == 'Gaussian':
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        else:
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

        g_x = 0, br[0] - ul[0]
        g_y = 0, br[1] - ul[1]
        img_x = ul[0], br[0]
        img_y = ul[1], br[1]

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            img[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        return img

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.data_path, 'images', f"{annotation['image_name']}")
        img = Image.open(img_path).convert('RGB')

        ridge_coordinates = annotation['ridge_coordinate']
        
        if self.transform:
            img,ridge_coordinates=self.transform(img,ridge_coordinates)
        heatmap_width = int(img.shape[1] * self.heatmap_rate)
        heatmap_height = int(img.shape[2] * self.heatmap_rate)
        heatmap = np.zeros((heatmap_width, heatmap_height), dtype=np.float32)
        for ridge_coordinate in ridge_coordinates:
            ridge_coordinate_scaled = (ridge_coordinate[0] * self.heatmap_rate, ridge_coordinate[1] * self.heatmap_rate)
            heatmap = self.generate_target(heatmap, ridge_coordinate_scaled, sigma=self.sigma, img_path=img_path)

        heatmap = heatmap[np.newaxis, :]
        return img, heatmap, img_path



class KeypointDetectionTransformHeatmap:
    def __init__(self, mean=[0.4623, 0.3856, 0.2822],
                 std=[0.2527, 0.1889, 0.1334],
                 size=(800,800),
                 mode='train'):
        self.size = size
        self.mode = mode

        if self.mode == 'train':
            self.transforms = transformsCompose([
                ContrastEnhancement(factor=1.5),
                Resize(size),
                Fix_RandomRotation(),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                Normalize(mean, std)
            ])
        else:
            self.transforms = transformsCompose([
                ContrastEnhancement(factor=1.5),
                Resize(size),
                ToTensor(),
                Normalize(mean, std)
            ])

    def __call__(self, img, keypoints):
        img, keypoints = self.transforms(img, keypoints)
        return img, keypoints
    

    