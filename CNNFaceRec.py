
import os
import math
import random
import argparse
from typing import Tuple, List, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ----------------------------
# Dataset and preprocessing
# ----------------------------
class LFWPairsDataset(Dataset):
    """
    Loads pairs from an LFW pairs file and returns (img1, img2, label)
    label: 1 for same person, 0 for different
    """
    def __init__(
        self,
        dataset_dir: str,
        pairs_file: str,
        image_size: Tuple[int, int] = (100, 100),
        max_pairs: Optional[int] = None,
        augment: bool = False,
    ):
        self.dataset_dir = dataset_dir
        self.pairs_file = pairs_file
        self.image_size = image_size
        self.max_pairs = max_pairs
        self.augment = augment

        self.pairs = self._parse_pairs_file(self.pairs_file)
        if self.max_pairs is not None:
            self.pairs = self.pairs[: self.max_pairs]

    def _parse_pairs_file(self, path: str) -> List[Tuple[str, int, str, int, int]]:
        pairs = []
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        # Skip header if present (first line often an integer count)
        start_idx = 0
        if lines and lines[0].split()[0].isdigit():
            start_idx = 1

        for ln in lines[start_idx:]:
            parts = ln.split()
            # same-person format: name i j
            # diff-person format: name1 i name2 j
            if len(parts) == 3:
                name = parts[0]
                i = int(parts[1])
                j = int(parts[2])
                img1 = self._image_path(name, i)
                img2 = self._image_path(name, j)
                label = 1
            elif len(parts) == 4:
                name1 = parts[0]
                i = int(parts[1])
                name2 = parts[2]
                j = int(parts[3])
                img1 = self._image_path(name1, i)
                img2 = self._image_path(name2, j)
                label = 0
            else:
                continue
            pairs.append((img1, img2, label))
        return pairs

    def _image_path(self, person_name: str, image_num: int) -> str:
        fn = f"{person_name}_{image_num:04d}.jpg"
        return os.path.join(self.dataset_dir, person_name, fn)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.image_size)
        if self.augment:
            # Lightweight augmentation
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            if random.random() < 0.2:
                alpha = 1.0 + (random.random() - 0.5) * 0.2
                beta = (random.random() - 0.5) * 10
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img = img.astype(np.float32) / 255.0
        # Standardize per-image lightly for CNN stability
        mean = float(img.mean())
        std = float(img.std())
        if std < 1e-3:
            std = 1e-3
        img = (img - mean) / std
        return img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p1, p2, label = self.pairs[idx]
        img1 = self._load_image(p1)
        img2 = self._load_image(p2)
        # Add channel dimension
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        return torch.from_numpy(img1), torch.from_numpy(img2), torch.tensor(label, dtype=torch.float32)


# ----------------------------
# Model: Siamese CNN
# ----------------------------
class ConvEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        # Input: 1 x 100 x 100
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # -> 16 x 100 x 100
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 16 x 50 x 50

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> 32 x 50 x 50
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32 x 25 x 25

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> 64 x 25 x 25
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 64 x 12 x 12 (floor)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        # Normalize embedding (helps contrastive training and cosine distance)
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.backbone = ConvEmbeddingNet(embedding_dim=embedding_dim)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_once(x1), self.forward_once(x2)


# ----------------------------
# Loss: Contrastive Loss (Hadsell et al. 2006)
# ----------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label_same: torch.Tensor) -> torch.Tensor:
        # label_same: 1 for same, 0 for different
        # Distance in embedding space
        dist = F.pairwise_distance(emb1, emb2, p=2)
        pos_loss = label_same * dist.pow(2)
        neg_loss = (1 - label_same) * F.relu(self.margin - dist).pow(2)
        loss = 0.5 * (pos_loss + neg_loss)
        return loss.mean()


# ----------------------------
# Utilities
# ----------------------------
@torch.no_grad()
def compute_distances(model: SiameseNet, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    dists, labels = [], []
    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        e1, e2 = model(x1, x2)
        dist = F.pairwise_distance(e1, e2, p=2)
        dists.append(dist.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(dists), np.concatenate(labels)


def tune_threshold(dists: np.ndarray, labels: np.ndarray) -> float:
    # Sweep thresholds over observed distances
    mins, maxs = float(np.min(dists)), float(np.max(dists))
    grid = np.linspace(mins, maxs, num=200)
    best_thr, best_f1 = grid[0], -1.0
    for thr in grid:
        preds = (dists <= thr).astype(np.int32)  # same if distance <= thr
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def evaluate_metrics(dists: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[float, float, float, float, Tuple[int, int, int, int]]:
    preds = (dists <= threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return acc, prec, rec, f1, (tp, fp, tn, fn)


# ----------------------------
# Training
# ----------------------------
def train(
    model: SiameseNet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    margin: float = 1.0,
):
    criterion = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x1, x2, y in pbar:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            e1, e2 = model(x1, x2)
            loss = criterion(e1, e2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{running / (pbar.n or 1):.4f}")

        if val_loader is not None:
            dists, labels = compute_distances(model, val_loader, device)
            thr = tune_threshold(dists, labels)
            acc, prec, rec, f1, (tp, fp, tn, fn) = evaluate_metrics(dists, labels, thr)
            print(f"[Val] thr={thr:.3f} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} | TP={tp} FP={fp} TN={tn} FN={fn}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Siamese CNN Face Verification on LFW")
    parser.add_argument("--dataset_dir", type=str, default="FaceRecognitionDset/lfw_funneled")
    parser.add_argument("--pairs_train", type=str, default="FaceRecognitionDset/pairsDevTrain.txt")
    parser.add_argument("--pairs_val", type=str, default="FaceRecognitionDset/pairsDevTest.txt")
    parser.add_argument("--pairs_test", type=str, default="FaceRecognitionDset/pairs.txt")
    parser.add_argument("--image_size", type=int, nargs=2, default=[100, 100])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--max_pairs_train", type=int, default=800)
    parser.add_argument("--max_pairs_eval", type=int, default=400)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_ds = LFWPairsDataset(
        dataset_dir=args.dataset_dir,
        pairs_file=args.pairs_train,
        image_size=tuple(args.image_size),
        max_pairs=args.max_pairs_train,
        augment=True,
    )
    val_ds = LFWPairsDataset(
        dataset_dir=args.dataset_dir,
        pairs_file=args.pairs_val,
        image_size=tuple(args.image_size),
        max_pairs=args.max_pairs_eval,
        augment=False,
    )
    test_ds = LFWPairsDataset(
        dataset_dir=args.dataset_dir,
        pairs_file=args.pairs_test,
        image_size=tuple(args.image_size),
        max_pairs=args.max_pairs_eval,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = SiameseNet(embedding_dim=args.embedding_dim).to(device)

    # Train
    train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, margin=args.margin)

    # Tune threshold on validation
    dists_val, labels_val = compute_distances(model, val_loader, device)
    thr = tune_threshold(dists_val, labels_val)
    print(f"Chosen threshold from validation: {thr:.3f}")

    # Evaluate on test
    dists_test, labels_test = compute_distances(model, test_loader, device)
    acc, prec, rec, f1, (tp, fp, tn, fn) = evaluate_metrics(dists_test, labels_test, thr)
    print("Test Results:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")


if __name__ == "__main__":
    main()