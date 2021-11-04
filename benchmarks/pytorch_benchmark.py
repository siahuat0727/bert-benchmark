from typing import Callable

import torch

from .base_benchmark import BaseBenchmark


class PyTorchBenchmark(BaseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.runtime_method == 'pytorch-jit':
            assert hasattr(self.args, 'torchscript')
            self.args.torchscript = True

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        model, input_ids = self._shared_prepare_inference_preprocessing(model_name, batch_size, sequence_length)

        if self.args.fp16:
            logger.info("Running training in Mixed Precision...")
            if not self.args.is_gpu:
                raise ValueError("Mixed precision is possible only for GPU.")
            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        if self.args.torchscript:
            with torch.no_grad():
                inference_model = torch.jit.trace(model, input_ids)
        else:
            inference_model = model

        def encoder_forward():
            with torch.no_grad():
                outputs = inference_model(input_ids)
            return outputs

        return encoder_forward

    def extract_output(self, output):
        return [output.cpu().numpy()]
