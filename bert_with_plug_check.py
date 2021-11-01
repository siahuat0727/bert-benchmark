import torch
import onnxruntime
import numpy as np

from utils import assert_equality

import ctypes
handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
assert handle, handle


def main():

    onnx_model_path = 'bert-with-plug.onnx'
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    input_ids = torch.randint(28996, (1, 512), dtype=torch.long)
    input_ids = torch.arange(512).unsqueeze(0)

    onnx_output = ort_session.run(
        None, {ort_session.get_inputs()[0].name: input_ids.cpu().numpy()})

    from trt_utils import load_engine, allocate_buffers
    import pycuda.driver as cuda

    batch_size = input_ids.size(0)

    trt_engine_path = 'bert-with-plug.engine'
    engine = load_engine(trt_engine_path)

    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    inputs[0].host[:input_ids.nelement()] = np.asarray(input_ids).ravel()

    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]

    success = context.execute(batch_size=batch_size, bindings=bindings)
    # success = context.execute_v2(bindings=bindings)
    print(f'{success=}')
    assert success, "Not exec success"
    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
    trt_output = [
        out.host
        for out in outputs
    ]
    max_diff = assert_equality(onnx_output, trt_output[-1:], atol=1e-1)
    print(max_diff)


main()
