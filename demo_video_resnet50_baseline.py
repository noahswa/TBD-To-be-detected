import argparse
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms


CLASS_DESCRIPTIONS = {
    "c0": "normal driving",
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
        raise ValueError(f"Checkpoint missing class_names: {checkpoint_path}")

    model = get_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def build_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def infer_true_label(image_path: Path, class_names):
    parent = image_path.parent.name
    if parent in class_names:
        return parent
    return None


def collect_images(image_paths, image_dir, max_images, seed):
    paths = [Path(path) for path in image_paths]

    if image_dir is not None:
        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError(f"Image directory not found: {root}")

        exts = {".jpg", ".jpeg", ".png"}
        dir_paths = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in exts
        ]
        dir_paths = sorted(dir_paths)
        if max_images is not None:
            rng = random.Random(seed)
            rng.shuffle(dir_paths)
            dir_paths = dir_paths[:max_images]
            dir_paths = sorted(dir_paths)
        paths.extend(dir_paths)

    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    if not unique:
        raise ValueError("No images found. Provide explicit paths or --image-dir.")
    return unique


@torch.no_grad()
def predict(model, image: Image.Image, preprocess, device):
    tensor = preprocess(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred_idx = int(probabilities.argmax().item())
    pred_prob = float(probabilities[pred_idx].item())
    return probabilities, pred_idx, pred_prob


def load_font(font_size: int, bold: bool = False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text_block(draw, text, x, y, font, fill):
    draw.text((x, y), text, font=font, fill=fill)
    bbox = draw.textbbox((x, y), text, font=font)
    return bbox[3] + 8


def render_frame(
    image: Image.Image,
    image_path: Path,
    index: int,
    total: int,
    class_names,
    pred_idx: int,
    pred_prob: float,
    topk_pairs,
    true_label,
    image_size,
    panel_width,
    title_font,
    body_font,
    class_font,
):
    image_w, image_h = image_size
    frame_w = image_w + panel_width
    frame_h = image_h

    frame = Image.new("RGB", (frame_w, frame_h), color=(14, 14, 14))
    image_canvas = image.resize((image_w, image_h), Image.Resampling.BILINEAR).convert("RGB")
    frame.paste(image_canvas, (0, 0))

    draw = ImageDraw.Draw(frame)
    draw.rectangle((image_w, 0, frame_w, frame_h), fill=(18, 18, 18))
    draw.line((image_w, 0, image_w, frame_h), fill=(70, 70, 70), width=2)
    draw = ImageDraw.Draw(frame)

    x = image_w + 16
    y = 16

    pred_class = class_names[pred_idx]
    pred_desc = CLASS_DESCRIPTIONS.get(pred_class, "")

    title = f"Sample {index + 1}/{total}"
    y = draw_text_block(draw, title, x, y, title_font, (255, 255, 255))
    y = draw_text_block(draw, f"File: {image_path.name}", x, y, body_font, (235, 235, 235))

    if true_label is not None:
        match = true_label == pred_class
        mark = "✓" if match else "✗"
        gt_text = f"GT: {true_label} ({CLASS_DESCRIPTIONS.get(true_label, '')})"
        y = draw_text_block(draw, gt_text, x, y, body_font, (160, 255, 160) if match else (255, 220, 160))
        y = draw_text_block(draw, f"Match: {mark}", x, y, body_font, (160, 255, 160) if match else (255, 200, 120))

    y = draw_text_block(draw, f"Pred: {pred_class} ({pred_desc})", x, y, body_font, (255, 255, 120))
    y = draw_text_block(draw, f"Confidence: {pred_prob:.1%}", x, y, body_font, (255, 255, 120))

    y = draw_text_block(draw, "Top-k:", x, y, body_font, (220, 220, 255))
    for class_name, probability in topk_pairs:
        y = draw_text_block(draw, f"- {class_name}: {probability:.1%}", x + 10, y, body_font, (220, 220, 255))

    y += 8
    y = draw_text_block(draw, "Classes:", x, y, body_font, (255, 255, 255))

    half = (len(class_names) + 1) // 2
    left_column = class_names[:half]
    right_column = class_names[half:]
    left_x = x + 10
    right_x = x + (panel_width // 2)
    class_y = y

    for idx, class_name in enumerate(left_column):
        label = CLASS_DESCRIPTIONS.get(class_name, "")
        draw.text((left_x, class_y + idx * 22), f"{class_name}: {label}", font=class_font, fill=(210, 210, 210))

    for idx, class_name in enumerate(right_column):
        label = CLASS_DESCRIPTIONS.get(class_name, "")
        draw.text((right_x, class_y + idx * 22), f"{class_name}: {label}", font=class_font, fill=(210, 210, 210))

    return np.asarray(frame)


def parse_args():
    parser = argparse.ArgumentParser(description="Create a short prediction demo video using trained ResNet-50 checkpoint.")
    parser.add_argument("images", nargs="*", help="Optional explicit image paths.")
    parser.add_argument("--image-dir", type=str, default="imgs/test", help="Directory to sample images from.")
    parser.add_argument("--max-images", type=int, default=10, help="Maximum images from --image-dir.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/resnet50_baseline/best_resnet50.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="outputs/resnet50_baseline/demo_prediction.mp4",
        help="Output mp4 path.",
    )
    parser.add_argument("--fps", type=int, default=2, help="Video frames per second.")
    parser.add_argument("--topk", type=int, default=3, help="Top-k predictions shown in overlay.")
    parser.add_argument("--device", type=str, default=None, help="Override device: cpu or cuda.")
    parser.add_argument("--width", type=int, default=960, help="Video frame width.")
    parser.add_argument("--height", type=int, default=540, help="Video frame height.")
    parser.add_argument("--panel-width", type=int, default=620, help="Right-side info panel width.")
    parser.add_argument("--title-font-size", type=int, default=38, help="Title font size for panel text.")
    parser.add_argument("--body-font-size", type=int, default=27, help="Body font size for panel text.")
    parser.add_argument("--class-font-size", type=int, default=18, help="Font size for c0-c9 class list.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint)
    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    model, class_names = load_checkpoint(checkpoint_path, device)
    preprocess = build_preprocess()

    image_paths = collect_images(args.images, args.image_dir, args.max_images, args.seed)
    image_size = (args.width, args.height)
    title_font = load_font(args.title_font_size, bold=True)
    body_font = load_font(args.body_font_size, bold=False)
    class_font = load_font(args.class_font_size, bold=False)

    topk = max(1, min(args.topk, len(class_names)))

    print("Class labels:")
    for class_name in class_names:
        print(f"  {class_name}: {CLASS_DESCRIPTIONS.get(class_name, '')}")

    with imageio.get_writer(output_video, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for index, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            probabilities, pred_idx, pred_prob = predict(model, image, preprocess, device)

            top_probs, top_idx = torch.topk(probabilities, k=topk)
            topk_pairs = [
                (class_names[int(idx.item())], float(prob.item()))
                for prob, idx in zip(top_probs, top_idx)
            ]

            true_label = infer_true_label(image_path, class_names)
            frame = render_frame(
                image=image,
                image_path=image_path,
                index=index,
                total=len(image_paths),
                class_names=class_names,
                pred_idx=pred_idx,
                pred_prob=pred_prob,
                topk_pairs=topk_pairs,
                true_label=true_label,
                image_size=image_size,
                panel_width=args.panel_width,
                title_font=title_font,
                body_font=body_font,
                class_font=class_font,
            )
            writer.append_data(frame)

            print(
                f"[{index + 1:03d}/{len(image_paths):03d}] {image_path} "
                f"-> pred={class_names[pred_idx]} ({pred_prob:.3f})"
            )

    print(f"Saved demo video to: {output_video}")


if __name__ == "__main__":
    main()
