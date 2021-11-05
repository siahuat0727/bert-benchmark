import os
from typing import Callable

import numpy as np

from .base_benchmark import BaseBenchmark


class TensorRTBenchmark(BaseBenchmark):

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
        from utils.tensorrt.utils import load_engine, allocate_buffers
        import pycuda.driver as cuda

        batch_size = input_ids.size(0)

        engine = load_engine(trt_engine_path)

        context = engine.create_execution_context()
        context.set_binding_shape(0, input_ids.size())
        inputs, outputs, bindings, stream = allocate_buffers(
            engine, dynamic_batch=self.dynamic_batch)
        inputs[0].host[:input_ids.nelement()] = np.asarray(
            input_ids).ravel()

        if self.use_plugin:
            import ctypes
            handle = ctypes.CDLL("libnvinfer_plugin.so",
                                 mode=ctypes.RTLD_GLOBAL)

        def encoder_forward():

            if self.check_equal:
                [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]

            # success = context.execute(batch_size=batch_size, bindings=bindings)
            success = context.execute_v2(bindings=bindings)
            if not success:
                raise AssertionError("Not exec success")

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
        if self.sequence_length is None:
            raise AssertionError

        if self.use_plugin:
            # TODO no magic number
            cmd = (
                f'python utils/tensorrt/builder.py -x {onnx_model_path}'
                f' -o {trt_engine_path}'
                f' -b {batch_size} -s {self.sequence_length} -c config/'
            )
            if self.fp16:
                cmd += ' --fp16'
            err = os.system(cmd)
            if err:
                raise AssertionError(f'{cmd} exit with errno {err}')
        else:
            from utils.tensorrt.utils import build_engine, save_engine
            engine = build_engine(onnx_model_path,
                                  input_ids.size()[1:],
                                  self.max_batch_size)

            save_engine(engine, trt_engine_path)

    def extract_output(self, output):
        return output
