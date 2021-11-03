import os
from typing import Callable

import numpy as np
import torch

from .base_benchmark import BaseBenchmark
from utils import assert_equality


class TensorRTBenchmark(BaseBenchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16 = 'fp16' in self.runtime_method.split('-')
        self.use_plugin = 'plugin' in self.runtime_method.split('-')

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)

        # TODO DRY
        model.cpu()
        input_ids = input_ids.cpu()

        onnx_model_path = f'{model_name}.onnx'
        self._export_onnx_model(model, input_ids, onnx_model_path)

        trt_engine_path = f'{model_name}.engine'

        self._export_tensorrt_engine(
            model, input_ids, onnx_model_path, trt_engine_path)

        return self._do_prepare_trt_inference_func(trt_engine_path, input_ids)

    def _do_prepare_trt_inference_func(self, trt_engine_path, input_ids):
        from trt_utils import load_engine, allocate_buffers
        import pycuda.driver as cuda

        batch_size = input_ids.size(0)

        engine = load_engine(trt_engine_path)

        context = engine.create_execution_context()

        context.set_binding_shape(0, (512,))
        context.set_binding_shape(1, (512,))
        context.set_binding_shape(2, (2,))
        context.set_binding_shape(3, (512,))
        inputs, outputs, bindings, stream = allocate_buffers(
            engine, dynamic_batch=self.dynamic_batch)
        inputs[0].host[:input_ids.nelement()] = np.asarray(
            input_ids).ravel()
        # inputs[1].host[:input_ids.nelement()] = np.asarray(
        #     torch.ones_like(input_ids)).ravel()
        inputs[2].host[:input_ids.nelement()] = np.asarray(
            torch.ones_like(input_ids)).ravel()

        if self.use_plugin:
            import ctypes
            handle = ctypes.CDLL("libnvinfer_plugin.so",
                                 mode=ctypes.RTLD_GLOBAL)

        def encoder_forward():

            [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]

            # success = context.execute(batch_size=batch_size, bindings=bindings)
            success = context.execute_v2(bindings=bindings)
            assert success, "Not exec success"

            if self.check_equal:
                [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]

            if self.dynamic_batch:
                return [
                    out.host.reshape(self.max_batch_size, -1)[:batch_size]
                    for out in outputs
                ]
            return [out.host for out in outputs]

        return encoder_forward

    def _export_tensorrt_engine(self, model, input_ids, onnx_model_path, trt_engine_path):
        if os.path.exists(trt_engine_path):
            return

        batch_size = input_ids.size(0)

        if self.use_plugin:
            # TODO no magic number
            cmd = (
                # f'python builder.py -x {onnx_model_path}'
                f'python builder_varseqlen.py -x {onnx_model_path}'
                f' -o {trt_engine_path}'
                f' -b {batch_size} -s 512 -c config/'
            )
            if self.fp16:
                cmd += ' --fp16'
            err = os.system(cmd)
            assert not err, f'{cmd} exit with errno {err}'
        else:
            from trt_utils import build_engine, save_engine
            engine = build_engine(onnx_model_path,
                                  input_ids.size()[1:],
                                  self.max_batch_size)

            save_engine(engine, trt_engine_path)

        if self.check_equal:
            self._assert_trt_valid(model, input_ids, trt_engine_path)

    def _assert_trt_valid(self, model, input_ids, trt_engine_path):

        trt_forward = self._do_prepare_trt_inference_func(
            trt_engine_path, input_ids)
        trt_output = trt_forward()

        pytorch_output = self._get_pytorch_output(model, input_ids)

        atol = 1e-5
        if self.use_plugin:
            atol = 5e-3
        if self.fp16:
            atol = 5e-2

        print(assert_equality(pytorch_output, trt_output, atol=atol))
        print(f'TensorRT {trt_engine_path} is valid!')
