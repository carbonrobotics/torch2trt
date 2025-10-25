"""
Multiple Optimization Profiles Example

This example demonstrates how to use multiple TensorRT optimization profiles
with torch2trt to handle different input shape ranges efficiently.

Multiple optimization profiles are useful when your model will process inputs
with significantly different characteristics, such as:
- Small vs. large images
- Different batch sizes for inference vs. training
- Multiple aspect ratios or resolutions

TensorRT will automatically select the best profile for each input at runtime.
"""

import torch
import torch.nn as nn
from torch2trt import torch2trt


def example_single_profile():
    """Example: Single optimization profile (original behavior)"""
    print("\n=== Example 1: Single Optimization Profile ===")

    model = nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda().eval()

    # Single profile - original API remains unchanged
    model_trt = torch2trt(
        model,
        [torch.randn(1, 3, 224, 224).cuda()],
        min_shapes=[(1, 3, 224, 224)],
        opt_shapes=[(1, 3, 224, 224)],
        max_shapes=[(4, 3, 224, 224)]
    )

    print("✓ Created TensorRT engine with 1 optimization profile")
    print("  Shape range: batch [1-4], spatial 224x224")

    # Test inference
    x = torch.randn(2, 3, 224, 224).cuda()
    output = model_trt(x)
    print(f"✓ Inference successful, output shape: {output.shape}")


def example_multiple_resolution_profiles():
    """Example: Multiple profiles for different image resolutions"""
    print("\n=== Example 2: Multiple Resolution Profiles ===")

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU()
    ).cuda().eval()

    # Define multiple optimization profiles for different resolutions
    # Profile 0: Small images (32x32 to 128x128)
    # Profile 1: Medium images (224x224 to 384x384)
    # Profile 2: Large images (512x512 to 1024x1024)

    min_shapes = [
        [(1, 3, 32, 32)],      # Profile 0 min
        [(1, 3, 224, 224)],    # Profile 1 min
        [(1, 3, 512, 512)]     # Profile 2 min
    ]

    opt_shapes = [
        [(1, 3, 64, 64)],      # Profile 0 optimal
        [(1, 3, 320, 320)],    # Profile 1 optimal
        [(1, 3, 768, 768)]     # Profile 2 optimal
    ]

    max_shapes = [
        [(4, 3, 128, 128)],    # Profile 0 max
        [(4, 3, 384, 384)],    # Profile 1 max
        [(2, 3, 1024, 1024)]   # Profile 2 max
    ]

    model_trt = torch2trt(
        model,
        [torch.randn(1, 3, 320, 320).cuda()],
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes
    )

    print("✓ Created TensorRT engine with 3 optimization profiles")
    print("  Profile 0: Small images (32-128 pixels)")
    print("  Profile 1: Medium images (224-384 pixels)")
    print("  Profile 2: Large images (512-1024 pixels)")

    # Test with different resolutions
    # TensorRT automatically selects the best profile for each input

    x_small = torch.randn(1, 3, 64, 64).cuda()
    output_small = model_trt(x_small)
    print(f"✓ Small image inference: {x_small.shape} -> {output_small.shape}")

    x_medium = torch.randn(1, 3, 320, 320).cuda()
    output_medium = model_trt(x_medium)
    print(f"✓ Medium image inference: {x_medium.shape} -> {output_medium.shape}")

    x_large = torch.randn(1, 3, 768, 768).cuda()
    output_large = model_trt(x_large)
    print(f"✓ Large image inference: {x_large.shape} -> {output_large.shape}")


def example_multiple_batch_profiles():
    """Example: Multiple profiles for different batch sizes"""
    print("\n=== Example 3: Multiple Batch Size Profiles ===")

    model = nn.Linear(128, 10).cuda().eval()

    # Define profiles optimized for different batch sizes
    # Profile 0: Single/small batch (1-4)
    # Profile 1: Medium batch (8-16)
    # Profile 2: Large batch (32-64)

    min_shapes = [
        [(1, 128)],    # Profile 0 min
        [(8, 128)],    # Profile 1 min
        [(32, 128)]    # Profile 2 min
    ]

    opt_shapes = [
        [(2, 128)],    # Profile 0 optimal
        [(12, 128)],   # Profile 1 optimal
        [(48, 128)]    # Profile 2 optimal
    ]

    max_shapes = [
        [(4, 128)],    # Profile 0 max
        [(16, 128)],   # Profile 1 max
        [(64, 128)]    # Profile 2 max
    ]

    model_trt = torch2trt(
        model,
        [torch.randn(2, 128).cuda()],
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes
    )

    print("✓ Created TensorRT engine with 3 batch size profiles")
    print("  Profile 0: Small batch (1-4)")
    print("  Profile 1: Medium batch (8-16)")
    print("  Profile 2: Large batch (32-64)")

    # Test with different batch sizes
    x_small = torch.randn(2, 128).cuda()
    output_small = model_trt(x_small)
    print(f"✓ Small batch inference: {x_small.shape} -> {output_small.shape}")

    x_medium = torch.randn(12, 128).cuda()
    output_medium = model_trt(x_medium)
    print(f"✓ Medium batch inference: {x_medium.shape} -> {output_medium.shape}")

    x_large = torch.randn(48, 128).cuda()
    output_large = model_trt(x_large)
    print(f"✓ Large batch inference: {x_large.shape} -> {output_large.shape}")


def example_fp16_int8_with_multiple_profiles():
    """Example: Multiple profiles with FP16 or INT8 mode"""
    print("\n=== Example 4: Multiple Profiles with FP16 ===")

    model = nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda().eval()

    # Multiple profiles work with all TensorRT features
    min_shapes = [
        [(1, 3, 224, 224)],
        [(1, 3, 512, 512)]
    ]
    opt_shapes = [
        [(1, 3, 224, 224)],
        [(1, 3, 512, 512)]
    ]
    max_shapes = [
        [(4, 3, 224, 224)],
        [(4, 3, 512, 512)]
    ]

    model_trt_fp16 = torch2trt(
        model,
        [torch.randn(1, 3, 224, 224).cuda()],
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        fp16_mode=True  # Enable FP16 precision
    )

    print("✓ Created TensorRT engine with 2 profiles and FP16 precision")

    x = torch.randn(1, 3, 512, 512).cuda()
    output = model_trt_fp16(x)
    print(f"✓ FP16 inference successful: {x.shape} -> {output.shape}")


def main():
    """Run all examples"""
    print("=" * 60)
    print("Multiple TensorRT Optimization Profiles Examples")
    print("=" * 60)

    example_single_profile()
    example_multiple_resolution_profiles()
    example_multiple_batch_profiles()
    example_fp16_int8_with_multiple_profiles()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
