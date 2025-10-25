# Multiple Optimization Profiles

TensorRT optimization profiles define the range of input shapes that an engine can handle. Starting with this update, torch2trt supports creating multiple optimization profiles for a single engine, allowing you to optimize performance across different input shape ranges.

## What are Optimization Profiles?

An optimization profile defines three key shapes for each input:
- **Minimum shape**: The smallest input shape the engine can handle
- **Optimal shape**: The shape for which the engine is most optimized
- **Maximum shape**: The largest input shape the engine can handle

TensorRT uses these profiles to optimize kernel selection and memory allocation for different input sizes.

## Why Use Multiple Profiles?

Multiple optimization profiles are beneficial when your model processes inputs with significantly different characteristics:

1. **Different image resolutions**: Small thumbnails vs. high-resolution images
2. **Varying batch sizes**: Single-image inference vs. batch processing
3. **Multiple aspect ratios**: Square images vs. wide panoramas
4. **Mixed workloads**: Different use cases requiring different input ranges

Without multiple profiles, a single profile must span the entire shape range, which may result in suboptimal performance. With multiple profiles, TensorRT can select the most appropriate profile for each input at runtime, improving performance.

## Basic Usage

### Single Profile (Original Behavior)

The original API remains unchanged for single profiles:

```python
import torch
from torch2trt import torch2trt

model = MyModel().cuda().eval()

# Single optimization profile
model_trt = torch2trt(
    model,
    [torch.randn(1, 3, 224, 224).cuda()],
    min_shapes=[(1, 3, 224, 224)],
    opt_shapes=[(1, 3, 224, 224)],
    max_shapes=[(4, 3, 224, 224)]
)
```

### Multiple Profiles (New Feature)

To create multiple profiles, provide lists of shape specifications:

```python
import torch
from torch2trt import torch2trt

model = MyModel().cuda().eval()

# Define multiple optimization profiles
min_shapes = [
    [(1, 3, 32, 32)],      # Profile 0: Small images
    [(1, 3, 224, 224)],    # Profile 1: Medium images
    [(1, 3, 512, 512)]     # Profile 2: Large images
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

# Create engine with 3 optimization profiles
model_trt = torch2trt(
    model,
    [torch.randn(1, 3, 320, 320).cuda()],
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes
)

# TensorRT automatically selects the best profile at runtime
output_small = model_trt(torch.randn(1, 3, 64, 64).cuda())    # Uses profile 0
output_medium = model_trt(torch.randn(1, 3, 320, 320).cuda()) # Uses profile 1
output_large = model_trt(torch.randn(1, 3, 768, 768).cuda())  # Uses profile 2
```

## Use Cases

### Example 1: Different Image Resolutions

Optimize for processing images at multiple resolutions:

```python
# Profile for small images (detection on thumbnails)
# Profile for medium images (standard inference)
# Profile for large images (high-quality processing)

min_shapes = [
    [(1, 3, 64, 64)],
    [(1, 3, 224, 224)],
    [(1, 3, 512, 512)]
]

opt_shapes = [
    [(4, 3, 96, 96)],
    [(4, 3, 320, 320)],
    [(2, 3, 768, 768)]
]

max_shapes = [
    [(8, 3, 128, 128)],
    [(8, 3, 448, 448)],
    [(4, 3, 1024, 1024)]
]

model_trt = torch2trt(
    model,
    [torch.randn(1, 3, 224, 224).cuda()],
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes
)
```

### Example 2: Different Batch Sizes

Optimize separately for single-image and batch inference:

```python
# Profile 0: Single/small batch (online inference)
# Profile 1: Large batch (batch processing)

min_shapes = [
    [(1, 3, 224, 224)],
    [(16, 3, 224, 224)]
]

opt_shapes = [
    [(4, 3, 224, 224)],
    [(32, 3, 224, 224)]
]

max_shapes = [
    [(8, 3, 224, 224)],
    [(64, 3, 224, 224)]
]

model_trt = torch2trt(
    model,
    [torch.randn(1, 3, 224, 224).cuda()],
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes
)
```

### Example 3: Multiple Input Tensors

For models with multiple inputs, specify shapes for each input:

```python
# Model with 2 inputs
min_shapes = [
    [(1, 3, 224, 224), (1, 128)],    # Profile 0
    [(1, 3, 512, 512), (1, 128)]     # Profile 1
]

opt_shapes = [
    [(2, 3, 320, 320), (2, 128)],    # Profile 0
    [(2, 3, 768, 768), (2, 128)]     # Profile 1
]

max_shapes = [
    [(4, 3, 448, 448), (4, 128)],    # Profile 0
    [(4, 3, 1024, 1024), (4, 128)]   # Profile 1
]

model_trt = torch2trt(
    model,
    [torch.randn(1, 3, 320, 320).cuda(), torch.randn(1, 128).cuda()],
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes
)
```

## Compatibility with Other Features

Multiple optimization profiles work seamlessly with all torch2trt features:

### FP16 Mode

```python
model_trt = torch2trt(
    model,
    [torch.randn(1, 3, 224, 224).cuda()],
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes,
    fp16_mode=True  # Enable FP16 precision
)
```

### INT8 Mode

When using INT8 mode with multiple profiles, you can specify which profile to use for calibration:

```python
# Define profiles for different use cases
min_shapes = [
    [(1, 3, 224, 224)],    # Profile 0: Small batch, standard resolution
    [(8, 3, 224, 224)],    # Profile 1: Large batch, standard resolution
    [(1, 3, 512, 512)]     # Profile 2: Small batch, high resolution
]

# If your primary workload is high-resolution inference,
# calibrate using profile 2 for best accuracy on that workload
model_trt = torch2trt(
    model,
    [torch.randn(1, 3, 224, 224).cuda()],
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes,
    int8_mode=True,
    int8_calib_dataset=calibration_dataset,
    int8_calib_profile_index=2  # Use profile 2 for calibration
)
```

**Important**: Choose the calibration profile that best represents your primary use case:
- Different input shapes can produce different activation ranges
- Batch size and spatial resolution affect calibration statistics
- Using the wrong profile may reduce accuracy on your main workload
- Default is profile 0 (first profile) if not specified

### ONNX Export

```python
model_trt = torch2trt(
    model,
    [torch.randn(1, 3, 224, 224).cuda()],
    min_shapes=min_shapes,
    opt_shapes=opt_shapes,
    max_shapes=max_shapes,
    use_onnx=True
)
```

## INT8 Calibration with Multiple Profiles

### Does the Calibration Profile Matter?

**Yes, it can significantly matter.** INT8 calibration determines quantization scales by analyzing activation ranges during inference. Different input shapes can produce different activation statistics, which affects the quality of quantization:

#### Why Different Profiles Produce Different Calibration Results:

1. **Batch size effects**:
   - Larger batches may have different statistical properties
   - Batch normalization layers behave differently with different batch sizes
   - Aggregation operations (mean, max pooling) can produce different ranges

2. **Spatial resolution effects**:
   - Different image sizes activate different numerical ranges
   - Downsampling operations produce different intermediate values
   - Edge cases at boundaries may appear only at certain resolutions

3. **Numerical precision**:
   - Larger inputs may accumulate different floating-point errors
   - Different computation patterns can affect final activation ranges

### Choosing the Right Calibration Profile

**Rule of thumb**: Calibrate using the profile that represents your **primary or most accuracy-critical workload**.

#### Example Scenarios:

**Scenario 1: High-resolution inference is critical**
```python
min_shapes = [
    [(1, 3, 224, 224)],    # Profile 0: Fast preview
    [(1, 3, 1024, 1024)]   # Profile 1: High-quality (primary use case)
]

model_trt = torch2trt(
    model, [x],
    min_shapes=min_shapes, opt_shapes=opt_shapes, max_shapes=max_shapes,
    int8_mode=True,
    int8_calib_dataset=calib_dataset,
    int8_calib_profile_index=1  # Optimize INT8 for high-resolution
)
```

**Scenario 2: Batch processing is most common**
```python
min_shapes = [
    [(1, 3, 224, 224)],     # Profile 0: Single-image serving
    [(16, 3, 224, 224)]     # Profile 1: Batch processing (primary use case)
]

model_trt = torch2trt(
    model, [x],
    min_shapes=min_shapes, opt_shapes=opt_shapes, max_shapes=max_shapes,
    int8_mode=True,
    int8_calib_dataset=calib_dataset,
    int8_calib_profile_index=1  # Optimize INT8 for batch workloads
)
```

### Calibration Dataset Considerations

The calibration dataset should contain inputs matching the shape range of your chosen profile:

```python
# If calibrating with profile 1 (high-resolution profile)
calib_dataset = ListDataset()
for img in calibration_images:
    # Use high-resolution images matching profile 1's range
    calib_dataset.insert((img.resize(1024, 1024).cuda(),))

model_trt = torch2trt(
    model, [x],
    min_shapes=min_shapes, opt_shapes=opt_shapes, max_shapes=max_shapes,
    int8_mode=True,
    int8_calib_dataset=calib_dataset,
    int8_calib_profile_index=1
)
```

### Validating Your Choice

To verify your calibration profile choice:

1. **Measure accuracy** on your primary workload with different calibration profiles
2. **Compare quantization errors** between profiles
3. **Test edge cases** specific to each profile's shape range

If accuracy differs significantly, choose the profile that gives best results for your most important use case.

## Best Practices

1. **Choose non-overlapping ranges**: While profiles can overlap, non-overlapping ranges often perform better as each profile is highly optimized for its specific range.

2. **Use optimal shapes wisely**: Set the optimal shape to the most common input size in each profile's range for best performance.

3. **Consider memory vs. performance trade-offs**: More profiles mean more optimization but also increased engine size and build time.

4. **Test with representative inputs**: Ensure your min/max ranges cover all expected inputs, or you'll get runtime errors.

5. **Profile count recommendations**:
   - 1-2 profiles: Most use cases
   - 3-4 profiles: Complex scenarios with very different input distributions
   - 5+ profiles: Rarely needed, may increase build time significantly

## Runtime Behavior

- TensorRT automatically selects the most appropriate profile for each input
- Profile selection is based on which profile's range best fits the input shape
- If an input falls outside all profile ranges, TensorRT will return an error
- Profile selection overhead is minimal and typically negligible

## Limitations

1. **Dynamic axes consistency**: Dynamic axes (dimensions that vary between min and max) must be consistent across all profiles
2. **DLA devices**: Multiple profiles are not supported when using DLA (Deep Learning Accelerator) devices
3. **Build time**: More profiles increase engine build time proportionally

## See Also

- [Basic Usage](basic_usage.md)
- [Reduced Precision](reduced_precision.md)
- [TensorRT Optimization Profiles Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles)
