import torch
import torch.nn as nn
import tensorrt as trt
from torch2trt import torch2trt


def test_multiple_optimization_profiles_conv2d():
    """Test multiple optimization profiles with different shape ranges"""

    torch.manual_seed(0)

    module = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1).cuda().eval()

    # Define multiple optimization profiles
    # Profile 1: Small images (32-128 pixels)
    # Profile 2: Large images (224-512 pixels)
    min_shapes = [
        [(1, 3, 32, 32)],   # Profile 1 min
        [(1, 3, 224, 224)]   # Profile 2 min
    ]
    opt_shapes = [
        [(1, 3, 64, 64)],   # Profile 1 optimal
        [(1, 3, 320, 320)]   # Profile 2 optimal
    ]
    max_shapes = [
        [(4, 3, 128, 128)], # Profile 1 max
        [(4, 3, 512, 512)]  # Profile 2 max
    ]

    module_trt = torch2trt(
        module,
        [torch.randn(1, 3, 64, 64).cuda()],
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        log_level=trt.Logger.INFO
    )

    # Test with inputs from profile 1 range (small images)
    x_small_1 = torch.randn(1, 3, 32, 32).cuda()
    assert torch.allclose(module(x_small_1), module_trt(x_small_1), rtol=1e-3, atol=1e-3)

    x_small_2 = torch.randn(2, 3, 64, 64).cuda()
    assert torch.allclose(module(x_small_2), module_trt(x_small_2), rtol=1e-3, atol=1e-3)

    x_small_3 = torch.randn(4, 3, 128, 128).cuda()
    assert torch.allclose(module(x_small_3), module_trt(x_small_3), rtol=1e-3, atol=1e-3)

    # Test with inputs from profile 2 range (large images)
    x_large_1 = torch.randn(1, 3, 224, 224).cuda()
    assert torch.allclose(module(x_large_1), module_trt(x_large_1), rtol=1e-3, atol=1e-3)

    x_large_2 = torch.randn(2, 3, 320, 320).cuda()
    assert torch.allclose(module(x_large_2), module_trt(x_large_2), rtol=1e-3, atol=1e-3)

    x_large_3 = torch.randn(4, 3, 512, 512).cuda()
    assert torch.allclose(module(x_large_3), module_trt(x_large_3), rtol=1e-3, atol=1e-3)


def test_multiple_optimization_profiles_sequential():
    """Test multiple optimization profiles with a more complex model"""

    torch.manual_seed(0)

    module = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    ).cuda().eval()

    # Define 3 optimization profiles for different use cases
    min_shapes = [
        [(1, 3, 32, 32)],    # Profile 1: Tiny images
        [(1, 3, 128, 128)],  # Profile 2: Medium images
        [(1, 3, 256, 256)]   # Profile 3: Large images
    ]
    opt_shapes = [
        [(1, 3, 48, 48)],
        [(1, 3, 160, 160)],
        [(1, 3, 384, 384)]
    ]
    max_shapes = [
        [(2, 3, 64, 64)],
        [(2, 3, 192, 192)],
        [(2, 3, 512, 512)]
    ]

    module_trt = torch2trt(
        module,
        [torch.randn(1, 3, 160, 160).cuda()],
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        log_level=trt.Logger.INFO
    )

    # Test each profile range
    x1 = torch.randn(1, 3, 32, 32).cuda()
    assert torch.allclose(module(x1), module_trt(x1), rtol=1e-3, atol=1e-3)

    x2 = torch.randn(1, 3, 128, 128).cuda()
    assert torch.allclose(module(x2), module_trt(x2), rtol=1e-3, atol=1e-3)

    x3 = torch.randn(1, 3, 256, 256).cuda()
    assert torch.allclose(module(x3), module_trt(x3), rtol=1e-3, atol=1e-3)


def test_multiple_optimization_profiles_batch_sizes():
    """Test multiple optimization profiles with different batch size ranges"""

    torch.manual_seed(0)

    module = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1).cuda().eval()

    # Define profiles with different batch size ranges
    min_shapes = [
        [(1, 3, 224, 224)],   # Profile 1: batch 1-4
        [(8, 3, 224, 224)]    # Profile 2: batch 8-32
    ]
    opt_shapes = [
        [(2, 3, 224, 224)],
        [(16, 3, 224, 224)]
    ]
    max_shapes = [
        [(4, 3, 224, 224)],
        [(32, 3, 224, 224)]
    ]

    module_trt = torch2trt(
        module,
        [torch.randn(1, 3, 224, 224).cuda()],
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        log_level=trt.Logger.INFO
    )

    # Test small batch sizes (profile 1)
    x_batch1 = torch.randn(1, 3, 224, 224).cuda()
    assert torch.allclose(module(x_batch1), module_trt(x_batch1), rtol=1e-3, atol=1e-3)

    x_batch4 = torch.randn(4, 3, 224, 224).cuda()
    assert torch.allclose(module(x_batch4), module_trt(x_batch4), rtol=1e-3, atol=1e-3)

    # Test large batch sizes (profile 2)
    x_batch8 = torch.randn(8, 3, 224, 224).cuda()
    assert torch.allclose(module(x_batch8), module_trt(x_batch8), rtol=1e-3, atol=1e-3)

    x_batch16 = torch.randn(16, 3, 224, 224).cuda()
    assert torch.allclose(module(x_batch16), module_trt(x_batch16), rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    test_multiple_optimization_profiles_conv2d()
    test_multiple_optimization_profiles_sequential()
    test_multiple_optimization_profiles_batch_sizes()
    print("All tests passed!")
