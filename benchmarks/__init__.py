from .pytorch_benchmark import PyTorchBenchmark
from .tensorrt_benchmark import TensorRTBenchmark
from .nnfusion_benchmark import NNFusionBenchmark
from .deepspeed_benchmark import DeepSpeedBenchmark
from .onnxruntime_benchmark import ONNXRuntimeBenchmark

BENCHMARKS = {
    'pytorch': PyTorchBenchmark,
    'tensorrt': TensorRTBenchmark,
    'nnfusion': NNFusionBenchmark,
    'deepspeed': DeepSpeedBenchmark,
    'onnxruntime': ONNXRuntimeBenchmark,
}
