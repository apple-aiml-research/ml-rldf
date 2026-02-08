#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch
from argparse import ArgumentParser
from transformers import CLIPProcessor, CLIPModel
from PIL import Image as PILImage
import os
import sys

parser = ArgumentParser()

parser.add_argument(
    "--image_path", type=str, required=True, help="Path to the UI screenshot"
)
parser.add_argument(
    "--description", type=str, required=True, help="Text description of the UI"
)
parser.add_argument("--model_path", type=str, default="./rldfclip-sketch")
parser.add_argument(
    "--processor_path", type=str, default="openai/clip-vit-base-patch32"
)
parser.add_argument("--dev", type=str, default="cpu")
parser.add_argument(
    "--negative_weight",
    type=float,
    default=0.5,
    help="Weight for negative embedding component",
)
parser.add_argument(
    "--negative1_weight",
    type=float,
    default=0.9,
    help="Weight for generic negative embedding",
)
parser.add_argument(
    "--negative2_weight",
    type=float,
    default=0.1,
    help="Weight for description-specific negative embedding",
)

args = parser.parse_args()


def preresize_image(image, image_size):
    """Resize image maintaining aspect ratio to have the smaller dimension equal to image_size."""
    # Resize image maintaining aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        # Width is greater than height
        image = image.resize((int(aspect_ratio * image_size), image_size))
    else:
        # Height is greater than width
        image = image.resize((image_size, int(image_size / aspect_ratio)))
    return image


def slide_window_over_image(input_image, img_size=224):
    """Create sliding window crops of the image with 50% overlap for CLIP processing."""
    input_image = preresize_image(input_image, img_size)
    # Determine the dimensions of the image
    width, height = input_image.size
    square_size = min(width, height)

    # Use 50% overlap for sliding window
    step_size = square_size // 2

    # Initialize a list to store the cropped images
    cropped_images = []

    # Slide the window over the image
    for y in range(0, height - square_size + 1, step_size):
        for x in range(0, width - square_size + 1, step_size):
            # Define the coordinates of the current window
            left = x
            upper = y
            right = x + square_size
            lower = y + square_size

            # Crop the image and add it to the list
            cropped_image = input_image.crop((left, upper, right, lower))
            cropped_images.append(cropped_image)

    return cropped_images


# Check CUDA availability
if args.dev == "cuda" and not torch.cuda.is_available():
    print("CUDA requested but not available, falling back to CPU")
    args.dev = "cpu"

print("Loading CLIP model and processor...")
try:
    model = (
        CLIPModel.from_pretrained(
            args.model_path,
        )
        .to(args.dev)
        .to(dtype=torch.float16)
        .eval()
    )

    processor = CLIPProcessor.from_pretrained(args.processor_path)
except Exception as e:
    print(f"Error loading CLIP model or processor: {e}")
    sys.exit(1)

print(f"Loading image from {args.image_path}...")
try:
    image = PILImage.open(args.image_path).convert("RGB")
except FileNotFoundError:
    print(f"Image file not found: {args.image_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

print("Processing image with sliding window...")
windowed_images = slide_window_over_image(image)

# Process image
inputs = processor(images=windowed_images, return_tensors="pt")
inputs["pixel_values"] = inputs["pixel_values"].to(model.device).to(dtype=torch.float16)

with torch.no_grad():
    image_features = model.get_image_features(**inputs)
    # Average all window embeddings
    image_embedding = image_features.mean(dim=0)

print("Computing text embeddings...")
description_lower = args.description.strip().lower()

# Positive text embedding
positive_text = "ui screenshot. well-designed. " + description_lower
inputs_positive = processor(text=[positive_text], return_tensors="pt", padding=True)
inputs_positive["input_ids"] = inputs_positive["input_ids"].to(model.device)
inputs_positive["attention_mask"] = inputs_positive["attention_mask"].to(model.device)

with torch.no_grad():
    positive_text_embedding = model.get_text_features(**inputs_positive)[0]

# Negative text embedding 1
negative_text1 = "ui screenshot. poor design. empty screen"
inputs_negative1 = processor(text=[negative_text1], return_tensors="pt", padding=True)
inputs_negative1["input_ids"] = inputs_negative1["input_ids"].to(model.device)
inputs_negative1["attention_mask"] = inputs_negative1["attention_mask"].to(model.device)

with torch.no_grad():
    negative_text_embedding1 = model.get_text_features(**inputs_negative1)[0]

# Negative text embedding 2
negative_text2 = positive_text.replace("well-designed. ", "poor design. ")
inputs_negative2 = processor(text=[negative_text2], return_tensors="pt", padding=True)
inputs_negative2["input_ids"] = inputs_negative2["input_ids"].to(model.device)
inputs_negative2["attention_mask"] = inputs_negative2["attention_mask"].to(model.device)

with torch.no_grad():
    negative_text_embedding2 = model.get_text_features(**inputs_negative2)[0]

# Combine embeddings using contrastive negative weighting
text_embedding = positive_text_embedding - args.negative_weight * (
    args.negative1_weight * negative_text_embedding1
    + args.negative2_weight * negative_text_embedding2
)

# Normalize embeddings
image_embedding = image_embedding / image_embedding.norm()
text_embedding = text_embedding / text_embedding.norm()

# Compute score
score = torch.sum(image_embedding * text_embedding).cpu().item()

print("\n" + "=" * 50)
print(f"UI Quality Score: {score:.4f}")
print("=" * 50)
print(f"\nImage: {args.image_path}")
print(f"Description: {args.description}")
