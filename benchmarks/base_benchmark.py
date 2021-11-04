import os
from typing import Callable
import timeit
import functools
from functools import partial

import numpy as np
import torch
import torch.onnx
from transformers import PyTorchBenchmark
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers import BertConfig

from utils import assert_equality
from models import BertModel


def get_encoder_output(forward):
    @functools.wraps(forward)
    def wrapper(data):
        return forward(data).last_hidden_state
    return wrapper


class BaseBenchmark(PyTorchBenchmark):
    """
    Base class of GPU inference speed test for pytorch, pytorch-jit, onnx, tensorrt, deepspeed and nnfusion
    """

    def __init__(self, *args, **kwargs):
        self.max_batch_size = kwargs.pop('max_batch_size')
        self.runtime_method = kwargs.pop('runtime_method')
        self.check_equal = kwargs.pop('check_equal')
        self.dynamic_batch = kwargs.pop('dynamic_batch')
        self.do_constant_folding = kwargs.pop('do_constant_folding')
        self.sequence_length = None
        self.pytorch_output = None

        super().__init__(*args, **kwargs)

    def do_measure_speed(self, func, repeat, number, is_warmup=True):

        if is_warmup:
            timeit.repeat(
                func,
                repeat=1,
                number=number,
            )

        # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
        runtimes = timeit.repeat(
            func,
            repeat=repeat,
            number=number,
        )
        print(f'{self.runtime_method} {runtimes}')

        return min(runtimes) / number

    # TODO not start with _, since some runtime (nnfusion) may override it
    def _measure_speed(self, func) -> float:
        return self.do_measure_speed(func, self.args.repeat, 10)

    def _shared_prepare_inference_preprocessing(self, model_name: str, batch_size: int, sequence_length: int):
        """Shared preprocessing for _prepare_xxx_inference_func"""
        # reference: super()._prepare_inference_func
        self.sequence_length = sequence_length

        config = self.config_dict[model_name]

        if self.args.torchscript:
            config.torchscript = True

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(
            config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(
            vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        config.max_position_embeddings = max(config.max_position_embeddings, sequence_length)
        if self.runtime_method == 'nnfusion':
            # input_shape is needed for onnx to generate node without op 'Where'
            # 'Where' op is not supported in NNFusion v0.3
            model = BertModel(config, input_shape=input_ids.size())
        else:
            model = BertModel(config)

        model.eval()
        model.to(self.args.device)

        if self.check_equal:
            self.pytorch_output = self._get_pytorch_output(model, input_ids)

        return model, input_ids

    def _export_onnx_model(self, model, input_ids, onnx_model_path):
        if os.path.exists(onnx_model_path):
            return

        kwargs = {}
        if self.dynamic_batch:
            kwargs['dynamic_axes'] = {
                'input': {0: 'batch_size'},
                'output1': {0: 'batch_size'},
            }

        torch.onnx.export(model,
                          input_ids,
                          onnx_model_path,
                          export_params=True,
                          opset_version=13,
                          verbose=False,
                          # when using trt with plugin, uncomment this line
                          do_constant_folding=self.do_constant_folding,
                          input_names=['input'],
                          output_names=['output1'],
                          **kwargs,
                          # output_names=['output1', 'output2'],
                          # dynamic_axes={'input': {0: 'batch_size'},
                          #               'output1': {0: 'batch_size'},
                          #               'output2': {0: 'batch_size'}},
                          )
        if self.check_equal:
            self._assert_onnx_valid(model, input_ids, onnx_model_path)

    def _do_prepare_onnx_inference_func(self, onnx_model_path, input_ids):

        import onnxruntime
        ort_session = onnxruntime.InferenceSession(onnx_model_path)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        input_ids = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}

        def encoder_forward():
            return ort_session.run(None, input_ids)

        return encoder_forward

    def _assert_onnx_valid(self, model, input_ids, onnx_model_path):
        assert self.check_equal

        import onnx
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)  # TODO What happend if fail?
        onnx_forward = self._do_prepare_onnx_inference_func(
            onnx_model_path, input_ids)
        onnx_output = onnx_forward()

        pytorch_output = self.pytorch_output

        print(assert_equality(pytorch_output, onnx_output))
        print(f'ONNX {onnx_model_path} is valid!')

    # TODO may call by deepspeed, don't start with _
    def _get_pytorch_output(self, model, input_ids):
        def extract_pytorch_output(tensor):
            # FIXME del
            return [tensor.cpu().numpy()]
            # return tensor[0].numpy(), tensor[1].numpy()
            return [
                tensor.last_hidden_state.numpy(),
                tensor.pooler_output.numpy(),
            ]

        with torch.no_grad():
            pytorch_output = model(input_ids)
        return extract_pytorch_output(pytorch_output)
