from pathlib import Path
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def transform(train=False):
    if train:
        ops = [transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
               transforms.RandomHorizontalFlip(), transforms.RandomApply([transforms.RandomRotation(15)], p=.5),
               transforms.ColorJitter(brightness=.25, contrast=.3, saturation=.8, hue=.5),
               transforms.GaussianBlur(3, sigma=(.1, 1.5)),
               transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    else:
        ops = [transforms.Resize(256), transforms.CenterCrop(224), transforms.Grayscale(num_output_channels=3),
               transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class SareeDataset(Dataset):
    def __init__(self, frame, root, train=False):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform(train)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        p = Path(row.path)
        if not p.is_absolute(): p = self.root / p
        with Image.open(p) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(row.label) if 'label' in self.frame.columns else -1, str(row.path)


class IdentityBatchSampler(Sampler):
    """P identities × K distinct images per identity, with replacement if needed."""
    def __init__(self, labels, p=8, k=4, batches=100):
        self.by_label = {}
        for i, y in enumerate(labels): self.by_label.setdefault(int(y), []).append(i)
        if len(self.by_label) < 2: raise ValueError("Need at least two design IDs to train")
        self.p, self.k, self.batches = min(p, len(self.by_label)), k, batches
        self.labels = list(self.by_label)

    def __len__(self): return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            chosen = random.sample(self.labels, self.p)
            batch = []
            for y in chosen:
                ids = self.by_label[y]
                batch.extend(random.choices(ids, k=self.k) if len(ids) < self.k else random.sample(ids, self.k))
            yield batch


def read_manifest(path, root):
    df = pd.read_csv(path)
    need = {"path", "design_id", "colorway_id", "split"}
    if not need.issubset(df.columns): raise ValueError(f"Manifest requires columns: {sorted(need)}")
    df = df.copy()
    df["design_id"] = df.design_id.astype(str)
    df["colorway_id"] = df.colorway_id.astype(str)
    for p in df.path:
        q = Path(p); q = q if q.is_absolute() else Path(root) / q
        if not q.is_file(): raise FileNotFoundError(q)
    return df
