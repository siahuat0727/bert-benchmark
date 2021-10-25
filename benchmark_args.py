from dataclasses import dataclass, field
from typing import List

from transformers.benchmark.benchmark_args_utils import list_field


# reference: transformers.benchmark.benchmark_args.BenchmarkArguments
@dataclass
class BenchmarkArgumentsSubset:
    """
    A subset of transformers.benchmark.benchmark_args.BenchmarkArguments
    """
    # TODO Check how to reuse transformers.benchmark.benchmark_args.BenchmarkArguments gracefully

    models: List[str] = list_field(
        default=[],
        metadata={
            "help": "Model checkpoints to be provided to the AutoModel classes. Leave blank to benchmark the base version of all available models"
        },
    )

    batch_sizes: List[int] = list_field(
        default=[8], metadata={"help": "List of batch sizes for which memory and time performance will be evaluated"}
    )

    sequence_lengths: List[int] = list_field(
        default=[8, 32, 128, 512],
        metadata={
            "help": "List of sequence lengths for which memory and time performance will be evaluated"},
    )

    save_to_csv: bool = field(default=False, metadata={
                              "help": "Save result to a CSV file"})

    env_print: bool = field(default=False, metadata={
                            "help": "Whether to print environment information"})
    repeat: int = field(default=3, metadata={
                        "help": "Times an experiment will be run."})

    @property
    def model_names(self):
        assert (
            len(self.models) > 0
        ), "Please make sure you provide at least one model name / model identifier, *e.g.* `--models bert-base-cased` or `args.models = ['bert-base-cased']."
        return self.models
