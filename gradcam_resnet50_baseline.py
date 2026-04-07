import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_DESCRIPTIONS = {
    "c0": "safe driving",
    "c1": "texting - right",
    "c2": "talking on the phone - right",
    "c3": "texting - left",
    "c4": "talking on the phone - left",
    "c5": "operating the radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger",
}


def get_model(num_classes: int):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(f"Checkpoint does not include class_names: {checkpoint_path}")

    model = get_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, checkpoint


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze()
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.cpu().numpy(), probabilities.detach().cpu().squeeze(0), target_class

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_preprocess():
    model_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    display_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
        ]
    )
    return model_transform, display_transform


def colorize_heatmap(heatmap):
    x = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def make_overlay(display_image, heatmap, alpha: float):
    image_np = np.asarray(display_image).astype(np.float32) / 255.0
    heatmap_rgb = colorize_heatmap(heatmap).astype(np.float32)
    overlay = (1.0 - alpha) * image_np + alpha * heatmap_rgb
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def parse_target_class(value, class_names):
    if value is None:
        return None
    if value in class_names:
        return class_names.index(value)
    try:
        class_index = int(value)
    except ValueError as exc:
        raise ValueError(f"Target class must be a class name or index, got: {value}") from exc
    if class_index < 0 or class_index >= len(class_names):
        raise ValueError(f"Target class index out of range: {class_index}")
    return class_index


def collect_images(image_paths, image_dir, max_images):
    paths = [Path(path) for path in image_paths]
    if image_dir is not None:
        dir_paths = sorted(
            path
            for path in Path(image_dir).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if max_images is not None:
            dir_paths = dir_paths[:max_images]
        paths.extend(dir_paths)

    if not paths:
        raise ValueError("Provide at least one image path or --image-dir.")
    return paths


def safe_stem(path: Path):
    parent = path.parent.name
    stem = path.stem
    return f"{parent}_{stem}" if parent else stem


def load_existing_summary(summary_path: Path):
    if not summary_path.exists():
        return None

    with open(summary_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_gradcam(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, checkpoint = load_checkpoint(checkpoint_path, device)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    model_transform, display_transform = build_preprocess()
    image_paths = collect_images(args.images, args.image_dir, args.max_images)
    target_class = parse_target_class(args.target_class, class_names)

    summaries = []
    try:
        for image_path in image_paths:
            original_image = Image.open(image_path).convert("RGB")
            input_tensor = model_transform(original_image).unsqueeze(0).to(device)
            display_image = display_transform(original_image)

            heatmap, probabilities, class_index = gradcam(input_tensor, target_class=target_class)
            overlay = make_overlay(display_image, heatmap, args.alpha)

            predicted_index = int(probabilities.argmax().item())
            output_path = output_dir / f"{safe_stem(image_path)}_gradcam_{class_names[class_index]}.jpg"
            overlay.save(output_path, quality=95)

            top_probability, top_index = torch.topk(probabilities, k=min(args.topk, len(class_names)))
            summary = {
                "image": str(image_path),
                "output": str(output_path),
                "target_class": class_names[class_index],
                "target_description": CLASS_DESCRIPTIONS.get(class_names[class_index], ""),
                "predicted_class": class_names[predicted_index],
                "predicted_description": CLASS_DESCRIPTIONS.get(class_names[predicted_index], ""),
                "predicted_probability": float(probabilities[predicted_index].item()),
                "topk": [
                    {
                        "class": class_names[int(idx.item())],
                        "description": CLASS_DESCRIPTIONS.get(class_names[int(idx.item())], ""),
                        "probability": float(prob.item()),
                    }
                    for prob, idx in zip(top_probability, top_index)
                ],
            }
            summaries.append(summary)
            print(
                f"{image_path} -> {output_path} | "
                f"pred={summary['predicted_class']} ({summary['predicted_probability']:.4f}) | "
                f"cam={summary['target_class']}"
            )
    finally:
        gradcam.close()

    summary_path = output_dir / "gradcam_summary.json"
    existing_summary = load_existing_summary(summary_path)
    if existing_summary:
        existing_summary.setdefault("summaries", []).extend(summaries)
        summary = existing_summary
        summary["checkpoint"] = str(checkpoint_path)
        summary["checkpoint_val_acc"] = checkpoint.get("val_acc")
        summary["target_layer"] = "layer4[-1]"
    else:
        summary = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_val_acc": checkpoint.get("val_acc"),
            "target_layer": "layer4[-1]",
            "summaries": summaries,
        }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved Grad-CAM summary to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualizations for the ResNet-50 baseline.")
    parser.add_argument("images", nargs="*", help="One or more image paths to visualize.")
    parser.add_argument("--image-dir", type=str, default=None, help="Optional directory of images to visualize.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images read from --image-dir.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/resnet50_baseline/best_resnet50.pt",
        help="Path to the trained ResNet-50 checkpoint.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/resnet50_baseline/gradcam")
    parser.add_argument("--target-class", type=str, default=None, help="Class name/index for Grad-CAM. Defaults to predicted class.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay opacity.")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to write in the summary.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
    return parser.parse_args()


if __name__ == "__main__":
    run_gradcam(parse_args())
