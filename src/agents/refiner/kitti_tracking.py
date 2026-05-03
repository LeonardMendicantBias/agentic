from pathlib import Path
import PIL

from torch.utils.data import Dataset
import torchvision.transforms as transforms


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

        subfolders = sorted(Path(f"{root_dir}/images/{split}").iterdir())
        self.samples, self.gt = [], []
        for folder in subfolders:
            images = list(Path(f"{folder}").rglob("*.png"))
            labels = list(Path(f"{root_dir}/panoptic_maps/{split}/{folder.name}").rglob("*.png"))

            for i in range(0, len(images) - n_steps - n_pred_steps - 1):
                self.samples.append((
                    images[i:i + n_steps + n_pred_steps - 1],
                    labels[i:i + n_steps + n_pred_steps - 1],
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        samples, labels = self.samples[idx]

        images = [PIL.Image.open(sample) for sample in samples]
        gts = [PIL.Image.open(label) for label in labels]

        if self.inp_transforms is not None:
            images = [self.inp_transforms(image) for image in images]
        return images, gts