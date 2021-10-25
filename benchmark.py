from transformers import BertConfig, BertModel
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

from utils import assert_equality


def get_encoder_output(forward):
    @functools.wraps(forward)
    def wrapper(data):
        return forward(data).last_hidden_state
    return wrapper


class BertModelSub(BertModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        return sequence_output
        # pooled_output = self.pooler(
        #     sequence_output) if self.pooler is not None else None

        # if not return_dict:
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return sequence_output, pooled_output
        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )


class MyPyTorchBenchmark(PyTorchBenchmark):
    """
    Gpu inference speed test for pytorch, pytorch-jit, onnx and tensorrt
    """

    def __init__(self, *args, **kwargs):
        self.max_batch_size = kwargs.pop('max_batch_size')
        self.runtime_method = kwargs.pop('runtime_method')

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
        timeit.repeat(
            func,
            repeat=1,
            number=10,
        )

        number = 10
        # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
        runtimes = timeit.repeat(
            func,
            repeat=self.args.repeat,
            number=number,
        )
        print(f'{self.runtime_method} {runtimes}')

        if self.args.is_tpu and self.args.torch_xla_tpu_print_metrics:
            import torch_xla.debug.metrics as met

            self.print_fn(met.metrics_report())

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

        self._assert_onnx_valid(model, input_ids, onnx_model_path)
        return self._do_prepare_onnx_inference_func(onnx_model_path, input_ids)

    def _prepare_tensorrt_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:

        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)

        # TODO DRY
        model.cpu()
        # model.forward = get_encoder_output(model.forward)
        input_ids = input_ids.cpu()

        # from net import Net
        # model = Net()
        # input_ids = torch.rand((batch_size, sequence_length))

        onnx_model_path = f'{model_name}.onnx'
        self._export_onnx_model(model, input_ids, onnx_model_path)

        trt_engine_path = f'{model_name}.engine'

        self._export_tensorrt_engine(
            model, input_ids, onnx_model_path, trt_engine_path)
        self._assert_trt_valid(model, input_ids, trt_engine_path)
        return self._do_prepare_trt_inference_func(trt_engine_path, input_ids)

    def _prepare_deepspeed_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:

        model, input_ids = self._shared_prepare_inference_preprocessing(
            model_name, batch_size, sequence_length)

        # TODO DRY
        # model.cpu()
        # input_ids = input_ids.cpu()

        return self._do_prepare_deepspeed_inference_func(model, input_ids)

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

        model = BertModelSub(BertConfig())
        # model = BertModel(BertConfig())
        model.eval()
        model.to(self.args.device)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(
            config, "vocab_size") else config.encoder.vocab_size
        input_ids = torch.randint(
            vocab_size, (batch_size, sequence_length), dtype=torch.long, device=self.args.device)
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

        torch.onnx.export(model,
                          input_ids,
                          onnx_model_path,
                          export_params=True,
                          opset_version=13,
                          verbose=False,
                          # do_constant_folding=False,  # when using trt with plugin, uncomment this line
                          input_names=['input'],
                          output_names=['output1'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output1': {0: 'batch_size'}},
                          # output_names=['output1', 'output2'],
                          # dynamic_axes={'input': {0: 'batch_size'},
                          #               'output1': {0: 'batch_size'},
                          #               'output2': {0: 'batch_size'}},
                          )
        self._assert_onnx_valid(model, input_ids, onnx_model_path)

    def _do_prepare_trt_inference_func(self, trt_engine_path, input_ids):
        from trt_utils import load_engine, allocate_buffers
        import pycuda.driver as cuda

        batch_size = input_ids.size(0)

        engine = load_engine(trt_engine_path)

        # context.set_binding_shape(0, input_ids.size())

        def sync_output(outputs):
            # FIXME del 1:
            return [
                out.host.reshape(self.max_batch_size, -1)[:batch_size]
                for out in outputs
            ]

        def encoder_forward():

            context = engine.create_execution_context()
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            inputs[0].host[:input_ids.nelement()] = np.asarray(
                input_ids).ravel()

            [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]

            success = context.execute(batch_size=batch_size, bindings=bindings)
            # success = context.execute_v2(bindings=bindings)
            assert success, "Not exec success"
            [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
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
        self._assert_trt_valid(model, input_ids, trt_engine_path)


    def _do_prepare_deepspeed_inference_func(self, model, input_ids):
        import deepspeed
        ds_engine = deepspeed.init_inference(model, mp_size=1, dtype=torch.half, replace_method='auto')

        model = ds_engine.module

        def encoder_forward():
            return model(input_ids)

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
            return [tensor.numpy()]
            # return tensor[0].numpy(), tensor[1].numpy()
            return [
                tensor.last_hidden_state.numpy(),
                tensor.pooler_output.numpy(),
            ]

        with torch.no_grad():
            pytorch_output = model(input_ids)
        return extract_pytorch_output(pytorch_output)
