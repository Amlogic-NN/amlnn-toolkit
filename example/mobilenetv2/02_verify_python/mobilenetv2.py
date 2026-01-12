# -*- coding: utf-8 -*-
"""
MobileNetV2 Quantized Inference Demo with AMLNNLite
Author: xinxin.he
Description:
    This script demonstrates how to perform image classification
    using a quantized MobileNetV2 (224x224) model with AMLNNLite.
"""

import numpy as np
import os
import glob
import argparse
from PIL import Image
from amlnnlite.api import AMLNNLite


def preprocess(image_path: str) -> np.ndarray:
    """
    Preprocess the input image for MobileNetV2 quantized model.

    Steps:
        1. Load image and convert to RGB
        2. Resize to 224x224
        3. Normalize to [-1, 1]
        4. Quantize to uint8 with zero-point = 128, scale = 0.0078125

    Args:
        image_path (str): Path to input image

    Returns:
        np.ndarray: Preprocessed image data with shape (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.array(img, dtype=np.float32)

    # Normalize to [-1, 1]
    img = img / 127.5 - 1.0

    # Expand batch dimension
    data = np.expand_dims(img, axis=0)

    # Quantization (uint8)
    data = data / 0.0078125 + 128
    data = np.clip(data, 0, 255).astype(np.uint8)

    return data


def postprocess_topk(predictions: np.ndarray, labels_path: str, k: int = 5) -> None:
    """
    Postprocess model output and print top-K classification results.

    Args:
        predictions (np.ndarray): Raw model output
        labels_path (str): Path to labels.txt
        k (int): Number of top results to display
    """
    predictions = predictions.reshape(-1)

    # Load labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    # Get top-k indices
    top_indices = np.argsort(predictions)[::-1][:k]

    print(f"\nTop-{k} Classification Results:")
    for rank, idx in enumerate(top_indices, start=1):
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        score = predictions[idx]
        print(f"  {rank}. {label:<20} (score: {score:.6f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='../01_export_model/mobilenet_v2_1.0_224_quant.tflite')
    args = parser.parse_args()
    
    # Initialize AMLNNLite
    amlnn = AMLNNLite()
    amlnn.config(
        model_path=args.model_path           # Model file path, Support ADLD and quantized TFlite models
    )
    amlnn.init()

    # Find all image files in the 01_export_model directory
    image_dir = "../01_export_model"
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print("No image files found in", image_dir)
        amlnn.uninit()
        return
    
    print(f"Found {len(image_files)} image files to process:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    print()

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)
        
        try:
            # Preprocess input
            input_data = preprocess(image_path)

            # Run inference
            outputs = amlnn.inference(
                inputs=[input_data]
            )

            # Postprocess results
            postprocess_topk(outputs[0], "../01_export_model/labels.txt", k=5)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
        
        print()

    # Optional visualization
    amlnn.visualize()

    # Release resources
    amlnn.uninit()


if __name__ == "__main__":
    main()
