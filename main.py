import dataclasses

from transformers import (
    HfArgumentParser,
    PyTorchBenchmarkArguments,
    BertConfig,
)

from benchmark_args import BenchmarkArgumentsSubset
from benchmark import MyPyTorchBenchmark


def bert_infer_speed(pt_benchmark_subargs, runtime_method, max_batch_size, n_layer):
    pt_benchmark_subdict = dataclasses.asdict(pt_benchmark_subargs)

    output_csv = f'speed-{runtime_method}.csv'
    pt_benchmark_subdict['inference_time_csv_file'] = output_csv
    output_csv = f'memory-{runtime_method}.csv'
    pt_benchmark_subdict['inference_memory_csv_file'] = output_csv

    args = PyTorchBenchmarkArguments(**pt_benchmark_subdict, memory=False)
    # args = PyTorchBenchmarkArguments(**pt_benchmark_subdict, multi_process=False)
    # args = PyTorchBenchmarkArguments(**pt_benchmark_subdict)

    config = BertConfig(num_hidden_layers=n_layer)

    benchmark = MyPyTorchBenchmark(args=args,
                                   configs=[config],
                                   runtime_method=runtime_method,
                                   max_batch_size=max_batch_size)
    benchmark.run()


def main():
    parser = HfArgumentParser(BenchmarkArgumentsSubset)
    parser.add_argument('--runtime-method',
                        default='pytorch',
                        choices=['pytorch', 'pytorch-jit',
                                 'onnxruntime', 'tensorrt', 'deepspeed'],
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

    pt_benchmark_subargs, args = parser.parse_args_into_dataclasses()

    assert all(
        batch_size <= args.max_batch_size
        for batch_size in pt_benchmark_subargs.batch_sizes
    ), "Batch sizes is too large (increase --max_batch_size)"

    bert_infer_speed(pt_benchmark_subargs, args.runtime_method,
                     args.max_batch_size, args.n_layer)


if __name__ == "__main__":
    main()
