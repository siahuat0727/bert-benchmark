#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script uses a prebuilt TensorRT BERT QA Engine to answer a question
based on the provided passage. It additionally includes an interactive mode
where multiple questions can be asked.
"""

import time
import json
import ctypes
import argparse
import collections
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch


TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--engine',
            help='Path to BERT TensorRT engine')
    parser.add_argument("-b", "--batch-size", default=1, help="Batch size for inference.", type=int)
    parser.add_argument('-p', '--passage', nargs='*',
            help='Text for paragraph/passage for BERT QA',
            default='')
    parser.add_argument('-pf', '--passage-file',
            help='File containing input passage',
            default='')
    parser.add_argument('-q', '--question', nargs='*',
            help='Text for query/question for BERT QA',
            default='')
    parser.add_argument('-qf', '--question-file',
            help='File containing input question',
            default='')
    parser.add_argument('-sq', '--squad-json',
            help='SQuAD json file',
            default='')
    parser.add_argument('-o', '--output-prediction-file',
            help='Output prediction file for SQuAD evaluation',
            default='./predictions.json')
    parser.add_argument('-v', '--vocab-file',
            help='Path to file containing entire understandable vocab')
    parser.add_argument('-s', '--sequence-length',
            help='The sequence length to use. Defaults to 128',
            default=128, type=int)
    parser.add_argument('--max-query-length',
            help='The maximum length of a query in number of tokens. Queries longer than this will be truncated',
            default=64, type=int)
    parser.add_argument('--max-answer-length',
            help='The maximum length of an answer that can be generated',
            default=30, type=int)
    parser.add_argument('--n-best-size',
            help='Total number of n-best predictions to generate in the nbest_predictions.json output file',
            default=20, type=int)
    parser.add_argument('--doc-stride',
            help='When splitting up a long document into chunks, what stride to take between chunks',
            default=128, type=int)
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    max_seq_length = args.sequence_length

    # Import necessary plugins for BERT TensorRT
    handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    if not handle:
        raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

    # The first context created will use the 0th profile. A new context must be created
    # for each additional profile needed. Here, we only use batch size 1, thus we only need the first profile.
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        for binding in engine:
            print(engine.get_binding_shape(binding), engine.max_batch_size)

        # select engine profile
        selected_profile = -1
        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles
        for idx in range(engine.num_optimization_profiles):
            profile_shape = engine.get_profile_shape(profile_index = idx, binding = idx * num_binding_per_profile)
            if profile_shape[0][0] <= args.batch_size and profile_shape[2][0] >= args.batch_size and profile_shape[0][1] <= max_seq_length and profile_shape[2][1] >= max_seq_length:
                selected_profile = idx
                break
        if selected_profile == -1:
            raise RuntimeError("Could not find any profile that can run batch size {}.".format(args.batch_size))

        context.active_optimization_profile = selected_profile
        binding_idx_offset = selected_profile * num_binding_per_profile

        # Specify input shapes. These must be within the min/max bounds of the active profile
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        input_shape = (args.batch_size, max_seq_length)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
        for binding in range(3):
            context.set_binding_shape(binding_idx_offset + binding, input_shape)
        assert context.all_binding_shapes_specified

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Allocate device memory for inputs.
        d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        # h_output_qkv = cuda.pagelocked_empty(tuple(context.get_binding_shape(binding_idx_offset + 3)), dtype=np.float32)
        h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(binding_idx_offset + 3)), dtype=np.float32)
        # h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(binding_idx_offset + 4)), dtype=np.float32)
        # d_output_qkv = cuda.mem_alloc(h_output_qkv.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        def inference(input_ids_batch):
            global h_output

            # Copy inputs
            # input_ids_batch = np.repeat(np.expand_dims(feature.input_ids, 0), args.batch_size, axis=0)
            input_ids = cuda.register_host_memory(np.ascontiguousarray(input_ids_batch.ravel()))
            eval_start_time = time.time()
            cuda.memcpy_htod_async(d_inputs[0], input_ids, stream)
            # Run inference
            context.execute_async_v2(bindings=[0 for i in range(binding_idx_offset)] + [int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
            # Synchronize the stream
            stream.synchronize()

            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # cuda.memcpy_dtoh_async(h_output_qkv, d_output_qkv, stream)
            stream.synchronize()
            # Only retrieve and post-process the first batch
            output = h_output
            print(f'trt {output.shape=} {output.flatten()[:20]=}')

        inference(torch.arange(512).unsqueeze(0))

