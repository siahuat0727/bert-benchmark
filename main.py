import dataclasses

from transformers import (
    HfArgumentParser,
    PyTorchBenchmarkArguments,
    BertConfig,
)

from benchmark_args import BenchmarkArgumentsSubset
from benchmarks import BENCHMARKS


def get_benchmark(runtime_method):
    return BENCHMARKS[runtime_method.split('-')[0]]


def bert_infer_speed(pt_benchmark_subargs, runtime_method, max_batch_size, n_layer, check_equal, dynamic_batch, do_constant_folding):
    pt_benchmark_subdict = dataclasses.asdict(pt_benchmark_subargs)

    if len(pt_benchmark_subargs.batch_sizes) != 1:
        raise AssertionError(
            pt_benchmark_subargs.batch_sizes
        )
    batch_size = pt_benchmark_subargs.batch_sizes[0]

    output_csv = f'speed#{runtime_method}#{batch_size}.csv'
    pt_benchmark_subdict['inference_time_csv_file'] = output_csv
    output_csv = f'memory#{runtime_method}#{batch_size}.csv'
    pt_benchmark_subdict['inference_memory_csv_file'] = output_csv
    pt_benchmark_subdict['env_info_csv_file'] = '/tmp/env.csv'

    # args = PyTorchBenchmarkArguments(**pt_benchmark_subdict, memory=False)
    args = PyTorchBenchmarkArguments(
        **pt_benchmark_subdict, multi_process=False, memory=False)

    config = BertConfig(num_hidden_layers=n_layer)

    benchmark_cls = get_benchmark(runtime_method)

    benchmark = benchmark_cls(args=args,
                              configs=[config],
                              runtime_method=runtime_method,
                              max_batch_size=max_batch_size,
                              check_equal=check_equal,
                              dynamic_batch=dynamic_batch,
                              do_constant_folding=do_constant_folding,
                              )
    benchmark.run()


def main():
    parser = HfArgumentParser(BenchmarkArgumentsSubset)
    parser.add_argument('--runtime-method',
                        default='pytorch',
                        choices=['pytorch',
                                 'pytorch-jit',
                                 'onnxruntime',
                                 'tensorrt',
                                 'tensorrt-plugin',
                                 'tensorrt-plugin-fp16',
                                 'deepspeed',
                                 'deepspeed-fp16',
                                 'nnfusion'],
                        help="Runtime selected to run inference"
                             "the option 'pytorch-jit' will replace")
    parser.add_argument('--n_layer',
                        default=12,
                        type=int,
                        help="Number of hidden layers of Bert")
    parser.add_argument('--max_batch_size',
                        default=16,
                        type=int,
                        help="For building trt engine")
    parser.add_argument('--check_equal',
                        action='store_true',
                        help="Whether to check equality")
    parser.add_argument('--dynamic_batch',
                        action='store_true',
                        help="Whether to use dynamic batch (for onnx and trt)")
    parser.add_argument('--do_constant_folding',
                        action='store_true',
                        help="Whether to enable constant folding (for onnx)")

    pt_benchmark_subargs, args = parser.parse_args_into_dataclasses()

    if args.runtime_method.startswith('tensorrt'):
        if 'plugin' in args.runtime_method:
            args.do_constant_folding = False
            args.dynamic_batch = False
        else:
            args.do_constant_folding = True
            args.dynamic_batch = True

    if not all(
        batch_size <= args.max_batch_size
        for batch_size in pt_benchmark_subargs.batch_sizes
    ):
        raise AssertionError("Batch sizes is too large (increase --max_batch_size)")

    bert_infer_speed(pt_benchmark_subargs, args.runtime_method,
                     args.max_batch_size, args.n_layer, args.check_equal,
                     args.dynamic_batch, args.do_constant_folding)


if __name__ == "__main__":
    main()
