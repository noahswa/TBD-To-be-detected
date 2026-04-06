import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices_stratified(targets, val_ratio: float, seed: int):
    rng = random.Random(seed)
    per_class = {}
    for idx, target in enumerate(targets):
        per_class.setdefault(target, []).append(idx)

    train_indices = []
    val_indices = []
    for _, indices in per_class.items():
        rng.shuffle(indices)
        val_count = int(len(indices) * val_ratio)
        val_count = min(max(val_count, 1), max(len(indices) - 1, 1)) if len(indices) > 1 else 0
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def get_model(num_classes: int, pretrained: bool):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def run_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, targets = batch[:2]
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_seen += batch_size

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, targets = batch[:2]
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_seen += batch_size

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc


def extract_resnet50_features(model, images):
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def export_features(model, loader, device, output_path: Path, class_names):
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []

    for images, targets, paths in loader:
        images = images.to(device)
        features = extract_resnet50_features(model, images)
        all_features.append(features.cpu().numpy())
        all_labels.append(targets.numpy())
        all_paths.extend(paths)

    if not all_features:
        return

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    class_names_np = np.array(class_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features_np,
        labels=labels_np,
        paths=np.array(all_paths),
        class_names=class_names_np,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet-50 baseline for distracted driver classes c0-c9")
    parser.add_argument("--data-dir", type=str, default="imgs/train", help="Path to class folders c0-c9")
    parser.add_argument("--output-dir", type=str, default="outputs/resnet50_baseline")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--save-features", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_transform, eval_transform = build_transforms()

    train_dataset_full = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=str(data_dir), transform=eval_transform)
    feature_dataset_full = ImageFolderWithPaths(root=str(data_dir), transform=eval_transform)

    class_names = train_dataset_full.classes
    num_classes = len(class_names)
    if num_classes != 10:
        print(f"Warning: expected 10 classes (c0-c9), found {num_classes}: {class_names}")

    train_indices, val_indices = split_indices_stratified(train_dataset_full.targets, args.val_ratio, args.seed)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=num_classes, pretrained=not args.no_pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_acc = -1.0
    best_path = output_dir / "best_resnet50.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_val_batches,
        )

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(summary)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_path,
            )

    history_path = output_dir / "metrics.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"Saved best checkpoint to: {best_path}")
    print(f"Saved metrics to: {history_path}")

    if args.save_features:
        feature_loader = DataLoader(
            feature_dataset_full,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        feature_path = Path(args.save_features)
        export_features(model, feature_loader, device, feature_path, class_names)
        print(f"Saved extracted features to: {feature_path}")


if __name__ == "__main__":
    main()