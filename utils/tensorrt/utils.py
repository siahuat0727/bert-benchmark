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

# def do_inference(engine, inputs, h_input_1, d_input_1, h_output_1, h_output_2, d_output_1, d_output_2, stream, batch_size):
#    """
#    This is the function to run the inference
#    Args:
#       engine : Path to the TensorRT engine
#       inputs : Input ids to the model.
#       h_input_1: Input in the host
#       d_input_1: Input in the device
#       h_output_1: Output in the host
#       d_output_1: Output in the device
#       stream: CUDA stream
#       batch_size : Batch size for execution time
#
#    Output:
#       The list of output images
#
#    """
#
#    load_inputs_to_buffer(inputs, h_input_1)
#
#    with engine.create_execution_context() as context:
#        # Transfer input data to the GPU.
#        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)
#
#        # Run inference.
#
#        context.profiler = trt.Profiler()
#        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output_1), int(d_output_2)])
#
#        # Transfer predictions back from the GPU.
#        cuda.memcpy_dtoh_async(h_output_1, d_output_1, stream)
#        cuda.memcpy_dtoh_async(h_output_2, d_output_2, stream)
#        # Synchronize the stream
#        stream.synchronize()
#        # Return the host output.
#        out_1 = h_output_1.reshape((batch_size,-1))
#        out_2 = h_output_2.reshape((batch_size,-1))
#        return out_1, out_2


def do_inference_sync(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]

    # Run inference.
    success_flag = context.execute_v2(bindings=bindings)
    if not success_flag:
        raise AssertionError('trt execute failed')

    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]

    return [out.host for out in outputs]


# def do_inference_sync(engine, inputs, h_input_1, d_input_1, h_output_1, h_output_2, d_output_1, d_output_2, stream, batch_size):
#    """
#    This is the function to run the inference
#    Args:
#       engine : Path to the TensorRT engine
#       inputs : Input ids to the model.
#       h_input_1: Input in the host
#       d_input_1: Input in the device
#       h_output_1: Output in the host
#       d_output_1: Output in the device
#       stream: CUDA stream
#       batch_size : Batch size for execution time
#
#    Output:
#       The list of output images
#
#    """
#
#    load_inputs_to_buffer(inputs, h_input_1)
#
#    with engine.create_execution_context() as context:
#        # Transfer input data to the GPU.
#        cuda.memcpy_htod(d_input_1, h_input_1)
#
#        # Run inference.
#
#        context.set_binding_shape(0, (2, 512))
#        context.profiler = trt.Profiler()
#        context.execute_v2(bindings=[int(d_input_1), int(d_output_1), int(d_output_2)])
#
#        # Transfer predictions back from the GPU.
#        cuda.memcpy_dtoh(h_output_1, d_output_1)
#        cuda.memcpy_dtoh(h_output_2, d_output_2)
#
#        # Return the host output.
#        out_1 = h_output_1
#        out_2 = h_output_2
#        return out_1, out_2


# FIXME save_engine=True
def get_engine_fix_batch(max_batch_size=2, onnx_file_path="", engine_file_path="", save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    input_shape = [512]

    def build_engine(max_batch_size):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = max_batch_size

            # network.add_input("input", trt.int32, (-1, 512))

            # pdb.set_trace()
            config = builder.create_builder_config()
            config.max_workspace_size = (1 << 30)

            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found'.format(onnx_file_path))
                sys.exit(0)
            with open(onnx_file_path, 'rb') as model:
                parsing_succeed = parser.parse(model.read())

                if not parsing_succeed:
                    sys.exit('Failed to parse the ONNX model')
            print('Building an engine from file {}; this may take a while...'.format(
                onnx_file_path))

            # Static input
            engine = builder.build_engine(network, config=config)

            if not engine:
                sys.exit('Failed to build the engine')

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                    print("Completed creating Engine")
            return engine
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size)
