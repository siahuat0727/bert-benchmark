import os
from typing import Callable
from pathlib import Path
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


# TODO refactor: split each runtime into individual class
class MyPyTorchBenchmark(PyTorchBenchmark):
    """
    Gpu inference speed test for pytorch, pytorch-jit, onnx and tensorrt
    """

    def __init__(self, *args, **kwargs):
        self.max_batch_size = kwargs.pop('max_batch_size')
        self.runtime_method = kwargs.pop('runtime_method')
        self.check_equal = kwargs.pop('check_equal')
        self.dynamic_batch = kwargs.pop('dynamic_batch')
        self.do_constant_folding = kwargs.pop('do_constant_folding')

        super().__init__(*args, **kwargs)

        if self.runtime_method == 'pytorch-jit':
            assert hasattr(self.args, 'torchscript')
            self.args.torchscript = True

    def _measure_speed(self, func) -> float:
        # try:
        # if self.runtime_method != 'pytorch':
        # if self.args.is_tpu or self.args.torchscript:
        # run additional 10 times to stabilize compilation for tpu and torchscript
        # logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
        number = 10

        if self.runtime_method != 'nnfusion':
            timeit.repeat(
                func,
                repeat=1,
                number=number,
            )

        repeat = 1 if self.runtime_method == 'nnfusion' else self.args.repeat
        number = 1 if self.runtime_method == 'nnfusion' else number

        # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
        runtimes = timeit.repeat(
            func,
            repeat=repeat,
            number=number,
        )
        print(f'{self.runtime_method} {runtimes}')

        if self.args.is_tpu and self.args.torch_xla_tpu_print_metrics:
            import torch_xla.debug.metrics as met

            self.print_fn(met.metrics_report())

        if self.runtime_method == 'nnfusion':
            with open('nnfusion_result.txt') as f:
                nnfusion_result = f.readlines()[-1]
            assert nnfusion_result.startswith('Summary'), nnfusion_result
            nnfusion_mintime = float(nnfusion_result.split('[')[2].split(',')[0]) / 1000
            return nnfusion_mintime
        return min(runtimes) / number
        # except RuntimeError as e:
        #     self.print_fn(f"Doesn't fit on GPU. {e}")
        #     return "N/A"

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        return {
            'pytorch': self._prepare_pytorch_inference_func,
            'pytorch-jit': self._prepare_pytorch_inference_func,
            'onnxruntime': self._prepare_onnx_inference_func,
            'tensorrt': self._prepare_tensorrt_inference_func,
            'deepspeed': self._prepare_deepspeed_inference_func,
            'nnfusion': self._prepare_nnfusion_inference_func,
        }[self.runtime_method](model_name, batch_size, sequence_length)

    def _prepare_pytorch_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        return super()._prepare_inference_func(model_name, batch_size, sequence_length)

    def _prepare_onnx_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:

        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)
        model.cpu()
        input_ids = input_ids.cpu()

        onnx_model_path = f'{model_name}.onnx'
        self._export_onnx_model(model, input_ids, onnx_model_path)

        return self._do_prepare_onnx_inference_func(onnx_model_path, input_ids)

    def _prepare_tensorrt_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:

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

    def _prepare_deepspeed_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:

        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)

        return self._do_prepare_deepspeed_inference_func(model, input_ids)

    def _prepare_nnfusion_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:

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
            model = BertModel(BertConfig())

        model.eval()
        model.to(self.args.device)
        return model, input_ids

    def _do_prepare_onnx_inference_func(self, onnx_model_path, input_ids):

        import onnxruntime
        ort_session = onnxruntime.InferenceSession(onnx_model_path)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        input_ids = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}

        def encoder_forward():
            return ort_session.run(None, input_ids)

        return encoder_forward

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
                          do_constant_folding=self.do_constant_folding,  # when using trt with plugin, uncomment this line
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

    def _do_prepare_trt_inference_func(self, trt_engine_path, input_ids):
        from trt_utils import load_engine, allocate_buffers
        import pycuda.driver as cuda

        batch_size = input_ids.size(0)

        engine = load_engine(trt_engine_path)

        context = engine.create_execution_context()
        context.set_binding_shape(0, input_ids.size())
        # FIXME add dynamic args
        inputs, outputs, bindings, stream = allocate_buffers(engine, dynamic_batch=self.dynamic_batch)
        inputs[0].host[:input_ids.nelement()] = np.asarray(
                input_ids).ravel()

        [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]

        def sync_output(outputs):
            # FIXME del 1:
            return [
                out.host.reshape(self.max_batch_size, -1)[:batch_size]
                for out in outputs
            ]

        def encoder_forward():

            # success = context.execute(batch_size=batch_size, bindings=bindings)
            success = context.execute_v2(bindings=bindings)
            assert success, "Not exec success"
            # [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
            return [
                out.host.reshape(self.max_batch_size, -1)[:batch_size]
                for out in outputs
            ]

        return encoder_forward

    def _export_tensorrt_engine(self, model, input_ids, onnx_model_path, trt_engine_path):
        if os.path.exists(trt_engine_path):
            return

        from trt_utils import build_engine, save_engine
        engine = build_engine(onnx_model_path,
                              input_ids.size()[1:],
                              self.max_batch_size)

        save_engine(engine, trt_engine_path)
        # FIXME not valid!
        # if self.check_equal:
        #     self._assert_trt_valid(model, input_ids, trt_engine_path)

    def _export_nnfusion_engine(self, model, input_ids, onnx_model_path, nnfusion_path):
        os.system(f'rm -rf nnfusion_rt')
        os.system(f'LD_LIBRARY_PATH=/usr/local/lib NNFUSION_HOME=/workspace/nnfusion nnfusion {onnx_model_path} -f onnx')
        os.system(f'cd nnfusion_rt/cuda_codegen && NNFUSION_HOME=/workspace/nnfusion cmake . && NNFUSION_HOME=/workspace/nnfusion make -j')
        assert os.path.exists(nnfusion_path)


    def _do_prepare_deepspeed_inference_func(self, model, input_ids):
        # TODO check correctness
        import deepspeed
        ds_engine = deepspeed.init_inference(model, mp_size=1, dtype=torch.half, replace_method='auto')
        ds_model = ds_engine.module

        if self.check_equal:
            pytorch_output = self._get_pytorch_output(model, input_ids)
            ds_output = self._get_pytorch_output(ds_model, input_ids)
            print(assert_equality(pytorch_output, ds_output))

        def encoder_forward():
            return ds_model(input_ids)

        return encoder_forward

    def _do_prepare_nnfusion_inference_func(self, model, input_ids, nnfusion_path):

        if self.check_equal:
            print('Warning: No implementation of nnfusion correctness check')

        filename = Path(nnfusion_path).name
        dirname = Path(nnfusion_path).parent

        def encoder_forward():
            os.system(f'cd {dirname} && ./{filename} > ../../nnfusion_result.txt')

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

    def _assert_trt_valid(self, model, input_ids, trt_engine_path):

        trt_forward = self._do_prepare_trt_inference_func(
            trt_engine_path, input_ids)
        trt_output = trt_forward()
        # sync_output = trt_forward()
        # trt_output = sync_output()

        pytorch_output = self._get_pytorch_output(model, input_ids)

        print(assert_equality(pytorch_output, trt_output))
        print(f'TensorRT {trt_engine_path} is valid!')

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
