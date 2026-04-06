# %%
from pathlib import Path
import PIL
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from matplotlib import pyplot as plt


class KittiDataset(Dataset):

    def __init__(self,
        root_dir: str,
        split: str,
        n_steps: int,
        n_pred_steps: int,
        inp_transforms: transforms=None
    ):
        self.root_dir = root_dir
        self.split = split
        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.inp_transforms = inp_transforms

        _dir = Path(f"{root_dir}/{split}")
        subfolders = sorted(_dir.joinpath("image_02").iterdir())
        self.samples, self.gt = [], []
        for folder in subfolders:
            _name = folder.name
            
            calib = self.load_calib(Path(f"{_dir}").joinpath(f"calib/{_name}.txt"))
            images = list(_dir.joinpath(f"image_02/{_name}").rglob("*.png"))
            pcds = list(_dir.joinpath(f"velodyne/{_name}").rglob("*.bin"))
            
            for i in range(0, len(images) - n_steps - n_pred_steps - 1):
                self.samples.append((
                    calib,
                    images[i:i + n_steps + n_pred_steps - 1],
                    pcds[i:i + n_steps + n_pred_steps - 1],
                ))

    @staticmethod
    def load_calib(calib_path):
        """Reads the calibration file and returns a dictionary of matrices."""
        calib_data = {}
        with open(calib_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if not line or line == '\n': continue
                
                if i < 4:
                    key, values = line.split(':', 1)
                else:
                    key, values = line.split(' ', 1)
                calib_data[key] = np.array([float(x) for x in values.split()])
        
        # Reshape matrices based on KITTI format
        calib_data['P2'] = calib_data['P2'].reshape(3, 4)
        calib_data['P3'] = calib_data['P3'].reshape(3, 4)
        calib_data['Tr_velo_cam'] = calib_data['Tr_velo_cam'].reshape(3, 4)
        # Convert Tr_velo_to_cam to 4x4 homogenous for easier projection
        calib_data['Tr_velo_cam'] = np.vstack([calib_data['Tr_velo_cam'], [0, 0, 0, 1]])
        
        # R0_rect is typically 3x3 in file, often used as 4x4 in math
        calib_data['R_rect'] = calib_data['R_rect'].reshape(3, 3)
        
        return calib_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        calib, image_paths, point_cloud_paths = self.samples[idx]

        P2 = calib['P2'].reshape(3, 4)
        R_rect = np.eye(4)
        R_rect[:3, :3] = calib['R_rect'].reshape(3, 3)
        Tr = calib['Tr_velo_cam']
        # print(calib['Tr_velo_cam'].shape)

        rgb_images = [self.inp_transforms(PIL.Image.open(path)) for path in image_paths]
        point_clouds = [np.fromfile(path, dtype=np.float32).reshape((-1, 4)) for path in point_cloud_paths]
        point_clouds = [
            np.column_stack((point_cloud[:, :3], np.ones(len(point_cloud))))
            for point_cloud in point_clouds
        ]

        pts_2ds = [
            (R_rect @ Tr @ point_cloud.T).T
            for point_cloud in point_clouds
        ]
        pts_2ds = [
            pts_2d[pts_2d[:, 2]>0]
            for pts_2d in pts_2ds
        ]
        pts_2ds = [
            (P2 @ pts_2d.T).T
            for pts_2d in pts_2ds
        ]
        
        depth_images = []
        for pts_2d, img in zip(pts_2ds, rgb_images):
            size = img.size
            
            depth_array = np.zeros(size[::-1], dtype=np.float32)
            
            uv = pts_2d[:, :2] / pts_2d[:, 2:3]
            u = np.round(uv[:, 0]).astype(int)
            v = np.round(uv[:, 1]).astype(int)

            valid = (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])
            depth_array[v[valid], u[valid]] = pts_2d[valid, 2] / pts_2d[valid, 2].max()
            
            depth_images.append(self.inp_transforms(PIL.Image.fromarray(depth_array)))

        return {
            'rgb': rgb_images,
            'depth': depth_images
        }
    

if __name__ == "__main__":
    root_dir = "C:/Users/ngoak/data/kitti_tracking"
    n_steps, n_pred_steps = 16, 3

    train_ds = KittiDataset(root_dir, "training", n_steps, n_pred_steps)
    for sample in train_ds:
        rgb_images, depth_images = sample

        # plt.imshow(rgb_images[0])
        # plt.axis("off")
        # plt.show()
        
        plt.imshow(depth_images[0])
        plt.axis("off")
        plt.show()
        break
