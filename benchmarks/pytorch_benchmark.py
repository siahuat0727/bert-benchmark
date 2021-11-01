from typing import Callable

from .base_benchmark import BaseBenchmark


class PyTorchBenchmark(BaseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.runtime_method == 'pytorch-jit':
            assert hasattr(self.args, 'torchscript')
            self.args.torchscript = True

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        return super()._prepare_inference_func(model_name, batch_size, sequence_length)
