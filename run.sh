#!/bin/bash

benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes 1 2 4 8 16 --repeat 3 --save_to_csv"
# python main.py $benchmark_settings --runtime-method pytorch
# python main.py $benchmark_settings --runtime-method pytorch-jit
# python main.py $benchmark_settings --runtime-method onnxruntime
# python main.py $benchmark_settings --runtime-method tensorrt

benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes 1 --repeat 3 --save_to_csv --max_batch_size 1"
python main.py $benchmark_settings --runtime-method tensorrt

# benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes 1 --repeat 3 --save_to_csv"
# python main.py $benchmark_settings --runtime-method pytorch --env_print

# python main.py --models bert-base-cased --sequence_lengths 512 --batch_sizes 1 --repeat 2 --save_to_csv --runtime-method onnx
# python main.py --models bert-base-cased --sequence_lengths 512 --batch_sizes 1 2 4 8 --repeat 2 --save_to_csv --runtime-method tensorrt
#
# python main.py --models bert-base-cased --sequence_lengths 512 --batch_sizes 8 --repeat 2 --save_to_csv --runtime-method onnx
# python main.py --models bert-base-cased --sequence_lengths 512 --batch_sizes 8 --repeat 2 --save_to_csv --runtime-method tensorrt
