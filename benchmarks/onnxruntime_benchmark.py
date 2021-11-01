from typing import Callable


from .base_benchmark import BaseBenchmark
from utils import assert_equality


class ONNXRuntimeBenchmark(BaseBenchmark):

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)
        model.cpu()
        input_ids = input_ids.cpu()

        onnx_model_path = f'{model_name}.onnx'
        self._export_onnx_model(model, input_ids, onnx_model_path)

        return self._do_prepare_onnx_inference_func(onnx_model_path, input_ids)
