from torch2trt.torch2trt import *


@tensorrt_converter('torch.Tensor.contiguous')
@tensorrt_converter('torch.nn.functional.dropout')
@tensorrt_converter('torch.nn.functional.dropout2d')
@tensorrt_converter('torch.nn.functional.dropout3d')
def convert_functional_identity(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    output._trt = input_trt


@tensorrt_converter('torch.nn.Dropout.forward')
@tensorrt_converter('torch.nn.Dropout2d.forward')
@tensorrt_converter('torch.nn.Dropout3d.forward')
def convert_identity(ctx):
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    output._trt = input_trt

@tensorrt_converter('torch.Tensor.to')
def convert_identity(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    torch_type = ctx.method_args[1]
    tensorrt_type = trt.DataType.FLOAT
    if torch_type == torch.float16:
        tensorrt_type = trt.DataType.HALF
    elif torch_type == torch.int8:
        tensorrt_type = trt.DataType.INT8

    layer = ctx.network.add_identity(input_trt)
    layer.set_output_type(0, tensorrt_type)
    output = ctx.method_return[0]
    output._trt = layer.get_output(0)
