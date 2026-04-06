# %%
from pathlib import Path
import PIL
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from matplotlib import pyplot as plt

from pathlib import Path
from datasets import IterableDataset
import numpy as np
from PIL import Image


class KittiHFIterableDataset:
    def __init__(
        self,
        root_dir: str,
        split: str,
        n_steps: int,
        n_pred_steps: int,
        transform=None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.transform = transform

        self.base_dir = self.root_dir / split

    @staticmethod
    def load_calib(calib_path):
        calib_data = {}

        with open(calib_path, "r") as f:
            for i, line in enumerate(f.readlines()):
                if not line.strip():
                    continue

                if i < 4:
                    key, values = line.split(":", 1)
                else:
                    key, values = line.split(" ", 1)

                calib_data[key] = np.array(
                    [float(x) for x in values.split()],
                    dtype=np.float32
                )

        calib_data["P2"] = calib_data["P2"].reshape(3, 4)
        calib_data["P3"] = calib_data["P3"].reshape(3, 4)

        calib_data["Tr_velo_cam"] = calib_data["Tr_velo_cam"].reshape(3, 4)
        calib_data["Tr_velo_cam"] = np.vstack(
            [calib_data["Tr_velo_cam"], [0, 0, 0, 1]]
        )

        calib_data["R_rect"] = calib_data["R_rect"].reshape(3, 3)

        return calib_data

    def _generate_depth(self, point_cloud, calib, image_size):
        P2 = calib["P2"]

        R_rect = np.eye(4, dtype=np.float32)
        R_rect[:3, :3] = calib["R_rect"]

        Tr = calib["Tr_velo_cam"]

        point_cloud = np.column_stack(
            (point_cloud[:, :3], np.ones(len(point_cloud)))
        )

        pts_3d = (R_rect @ Tr @ point_cloud.T).T
        pts_3d = pts_3d[pts_3d[:, 2] > 0]

        pts_2d = (P2 @ pts_3d.T).T

        depth = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

        uv = pts_2d[:, :2] / pts_2d[:, 2:3]
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)

        valid = (
            (u >= 0)
            & (u < image_size[0])
            & (v >= 0)
            & (v < image_size[1])
        )

        if valid.any():
            depth[v[valid], u[valid]] = (
                pts_2d[valid, 2] / pts_2d[valid, 2].max()
            )

        return depth

    def generator(self):
        image_root = self.base_dir / "image_02"
        velo_root = self.base_dir / "velodyne"
        calib_root = self.base_dir / "calib"

        for seq_dir in sorted(image_root.iterdir()):
            seq_name = seq_dir.name

            calib = self.load_calib(calib_root / f"{seq_name}.txt")

            images = sorted(seq_dir.glob("*.png"))
            pcds = sorted((velo_root / seq_name).glob("*.bin"))

            window = self.n_steps + self.n_pred_steps - 1

            for i in range(len(images) - window):
                rgb_list = []
                depth_list = []

                img_paths = images[i:i + window]
                pcd_paths = pcds[i:i + window]

                for img_path, pcd_path in zip(img_paths, pcd_paths):
                    img = Image.open(img_path).convert("RGB")

                    point_cloud = np.fromfile(
                        pcd_path,
                        dtype=np.float32
                    ).reshape(-1, 4)

                    depth = self._generate_depth(
                        point_cloud,
                        calib,
                        img.size
                    )

                    if self.transform:
                        img = self.transform(img)
                        depth = self.transform(Image.fromarray(depth))
                    else:
                        img = np.array(img)
                        depth = depth

                    rgb_list.append(img)
                    depth_list.append(depth)

                yield {
                    "rgb": rgb_list,
                    "depth": depth_list,
                    "calib": {
                        "P2": calib["P2"],
                        "R_rect": calib["R_rect"],
                        "Tr_velo_cam": calib["Tr_velo_cam"]
                    }
                }

    def to_hf_dataset(self):
        return IterableDataset.from_generator(self.generator)
    

if __name__ == "__main__":
    root_dir = "C:/Users/ngoak/data/kitti_tracking"
    n_steps, n_pred_steps = 16, 3

    dataset_builder = KittiHFIterableDataset(
        root_dir="C:/Users/ngoak/data/kitti_tracking",
        split="training",
        n_steps=n_steps,
        n_pred_steps=n_pred_steps,
        # transform=T.Compose([
        #     T.Resize((240, 640)),
        # ])
    )

    hf_dataset = dataset_builder.to_hf_dataset()

    sample = next(iter(hf_dataset))
    print(sample.keys())


# %%
    plt.imshow(
        PIL.Image.fromarray(1000*sample['depth'][0])
    )
