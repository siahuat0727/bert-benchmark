from typing import Callable

import torch

from .base_benchmark import BaseBenchmark
from utils import assert_equality


class DeepSpeedBenchmark(BaseBenchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16 = 'fp16' in self.runtime_method.split('-')

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)

        return self._do_prepare_deepspeed_inference_func(model, input_ids)

    def _do_prepare_deepspeed_inference_func(self, model, input_ids):
        import deepspeed

        dtype = torch.half if self.fp16 else torch.float
        ds_engine = deepspeed.init_inference(
            model, mp_size=1, dtype=dtype, replace_method='auto')
        ds_model = ds_engine.module

        if self.check_equal:
            pytorch_output = self._get_pytorch_output(model, input_ids)
            ds_output = self._get_pytorch_output(ds_model, input_ids)
            atol = 1e-3 if self.fp16 else 1e-5
            print(assert_equality(pytorch_output, ds_output, atol=atol))

        def encoder_forward():
            return ds_model(input_ids)

        return encoder_forward
