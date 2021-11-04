from typing import Callable

import torch

from .base_benchmark import BaseBenchmark


class DeepSpeedBenchmark(BaseBenchmark):

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)

        dtype = torch.half if self.fp16 else torch.float

        import deepspeed
        ds_engine = deepspeed.init_inference(
            model, mp_size=1, dtype=dtype, replace_method='auto')
        ds_model = ds_engine.module

        return self._do_prepare_deepspeed_inference_func(ds_model, input_ids)

    def _do_prepare_deepspeed_inference_func(self, model, input_ids):

        def encoder_forward():
            with torch.no_grad():
                return model(input_ids)

        return encoder_forward

    def extract_output(self, output):
        return [output.cpu().numpy()]
