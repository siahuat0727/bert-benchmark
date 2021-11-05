import os
import timeit
from pathlib import Path
from typing import Callable

from .base_benchmark import BaseBenchmark


class NNFusionBenchmark(BaseBenchmark):

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)

        # TODO DRY
        model.cpu()
        input_ids = input_ids.cpu()

        onnx_model_path = f'{model_name}.onnx'
        self._export_onnx_model(model, input_ids, onnx_model_path)

        nnfusion_path = f'nnfusion_rt/cuda_codegen/main_test'

        self._export_nnfusion_engine(
            model, input_ids, onnx_model_path, nnfusion_path)
        return self._do_prepare_nnfusion_inference_func(model, input_ids, nnfusion_path)

    def _measure_speed(self, func) -> float:
        _ignore_result = self.do_measure_speed(func, 1, 1, False)

        with open('nnfusion_result.txt') as f:
            nnfusion_result = f.readlines()[-1]
        if not nnfusion_result.startswith('Summary'):
            raise AssertionError(nnfusion_result)

        nnfusion_mintime = float(nnfusion_result.split('[')[
                                 2].split(',')[0]) / 1000
        return nnfusion_mintime

    def _export_nnfusion_engine(self, model, input_ids, onnx_model_path, nnfusion_path):
        os.system(f'rm -rf nnfusion_rt')
        os.system(
            f'LD_LIBRARY_PATH=/usr/local/lib nnfusion {onnx_model_path} -f onnx')
        os.system(f'cd nnfusion_rt/cuda_codegen && cmake . && make -j')
        if not os.path.exists(nnfusion_path):
            raise AssertionError

    def _do_prepare_nnfusion_inference_func(self, model, input_ids, nnfusion_path):

        if self.check_equal:
            print('Warning: No implementation of nnfusion correctness check')

        filename = Path(nnfusion_path).name
        dirname = Path(nnfusion_path).parent

        def encoder_forward():
            os.system(
                f'cd {dirname} && ./{filename} > ../../nnfusion_result.txt')

        return encoder_forward
