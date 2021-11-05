# reference: https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/, https://github.com/RizhaoCai/PyTorch_ONNX_TensorRT/blob/master/dynamic_shape_example.py

import os
import sys

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def save_engine(engine, path):
    buf = engine.serialize()
    with open(path, 'wb') as f:
        f.write(buf)


def load_engine(path):
    with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def build_engine(onnx_file_path, input_shape, max_batch_size):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = max_batch_size

        config = builder.create_builder_config()
        config.max_workspace_size = (1 << 30)

        with open(onnx_file_path, 'rb') as model:
            parsing_succeed = parser.parse(model.read())
            if not parsing_succeed:
                raise AssertionError('Failed to parse the ONNX model')

        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1,) + input_shape, (max_batch_size,) +
                          input_shape, (max_batch_size,) + input_shape)
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config=config)
        if not engine:
            raise AssertionError('Failed to build the engine')
    return engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, dynamic_batch=False):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        slice_ = slice(1, None) if dynamic_batch else slice(None)
        size = trt.volume(engine.get_binding_shape(
            binding)[slice_]) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        binding_index = engine.get_binding_index(binding)

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        # Only bytes, no need for size
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    success_flag = context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)  # Bug
    # success_flag = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle) # Bug

    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()

    return [out.host for out in outputs]


def do_inference_sync(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]

    # Run inference.
    success_flag = context.execute_v2(bindings=bindings)
    if not success_flag:
        raise AssertionError('trt execute failed')

    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]

    return [out.host for out in outputs]
