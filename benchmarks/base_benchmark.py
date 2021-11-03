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

        config = self.config_dict[model_name]

        if self.args.torchscript:
            config.torchscript = True

        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                model_class = config.architectures[0]
                transformers_module = __import__(
                    "transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                model = model_cls(config)
            except ImportError:
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            model = MODEL_MAPPING[config.__class__](config)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(
            config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(
            vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)

        if self.runtime_method == 'nnfusion':
            # input_shape is needed for onnx to generate node without op 'Where'
            # 'Where' op is not supported in NNFusion v0.3
            model = BertModel(BertConfig(), input_shape=input_ids.size())
        else:
            model = BertModel(BertConfig(num_hidden_layers=1))

        model.eval()
        model.to(self.args.device)
        return model, input_ids

    def _export_onnx_model(self, model, input_ids, onnx_model_path):
        if os.path.exists(onnx_model_path):
            return

        attention_mask = torch.ones_like(input_ids)
        # token_ids = torch.zeros_like(input_ids).long()

        inputs = (input_ids, attention_mask)
        # inputs = (input_ids, attention_mask, token_ids)

        n_input = 2
        n_output = 1

        input_names = [f'input{i+1}' for i in range(n_input)]
        output_names = [f'output{i+1}' for i in range(n_output)]


        kwargs = {}
        if self.dynamic_batch:
            kwargs['dynamic_axes'] = {
                k: {0: 'batch_size'}
                for k in input_names + output_names
            }

        torch.onnx.export(model,
                          inputs,
                          onnx_model_path,
                          export_params=True,
                          opset_version=13,
                          verbose=False,
                          do_constant_folding=self.do_constant_folding,
                          input_names=input_names,
                          output_names=output_names,
                          **kwargs,
                          )
        if self.check_equal:
            self._assert_onnx_valid(model, input_ids, onnx_model_path)

    def _do_prepare_onnx_inference_func(self, onnx_model_path, input_ids):

        import onnxruntime
        ort_session = onnxruntime.InferenceSession(onnx_model_path)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # input_ids2 = {ort_session.get_inputs()[0].name: to_numpy(torch.ones_like(input_ids))}
        ort_input = {
                ort_session.get_inputs()[0].name: to_numpy(input_ids),
                ort_session.get_inputs()[1].name: to_numpy(torch.ones_like(input_ids)),
                # ort_session.get_inputs()[2].name: to_numpy(torch.zeros_like(input_ids).long()),
        }

        def encoder_forward():
            return ort_session.run(None, ort_input)

        return encoder_forward

    def _assert_onnx_valid(self, model, input_ids, onnx_model_path):

        import onnx
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)  # TODO What happend if fail?
        onnx_forward = self._do_prepare_onnx_inference_func(
            onnx_model_path, input_ids)
        onnx_output = onnx_forward()

        pytorch_output = self._get_pytorch_output(model, input_ids)

        print(assert_equality(pytorch_output, onnx_output))
        print(f'ONNX {onnx_model_path} is valid!')

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
