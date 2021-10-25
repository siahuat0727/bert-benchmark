#!/bin/bash

nvprof_args="--csv --print-gpu-summary  --log-file nvprof%p.log --profile-child-processes"

rm nvprof*.log
for batch in 1 2 4 8 16
do
	for runtime in pytorch pytorch-jit onnxruntime tensorrt deepspeed
	do
		benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes $batch --repeat 3 --save_to_csv"
		nvprof $nvprof_args python main.py $benchmark_settings --runtime-method $runtime && \
		ls nvprof*.log | xargs sed '/^==/d'> "nvprof#${runtime}#${batch}.csv" && \
		rm nvprof*.log
		rm *.onnx
	done
done

benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes 1 2 4 8 16 --repeat 3 --save_to_csv"
# python main.py $benchmark_settings --runtime-method pytorch
benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes 2 --repeat 3 --save_to_csv"
# python main.py $benchmark_settings --runtime-method pytorch
# python main.py $benchmark_settings --runtime-method pytorch-jit
# python main.py $benchmark_settings --runtime-method onnxruntime
# python main.py $benchmark_settings --runtime-method tensorrt
# python main.py $benchmark_settings --runtime-method deepspeed


# rm nvprof*.log
# nvprof $nvprof_args python main.py $benchmark_settings --runtime-method pytorch && \
# ls nvprof*.log|xargs sed '/^==/d'> nvprof_pytorch.csv

benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes 1 --repeat 3 --save_to_csv"
# python main.py $benchmark_settings --runtime-method onnxruntime

benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes 1 --repeat 3 --save_to_csv --max_batch_size 1"
# python main.py $benchmark_settings --runtime-method tensorrt

# benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes 1 --repeat 3 --save_to_csv"
# python main.py $benchmark_settings --runtime-method pytorch --env_print

# python main.py --models bert-base-cased --sequence_lengths 512 --batch_sizes 1 --repeat 2 --save_to_csv --runtime-method onnx
# python main.py --models bert-base-cased --sequence_lengths 512 --batch_sizes 1 2 4 8 --repeat 2 --save_to_csv --runtime-method tensorrt
#
# python main.py --models bert-base-cased --sequence_lengths 512 --batch_sizes 8 --repeat 2 --save_to_csv --runtime-method onnx
# python main.py --models bert-base-cased --sequence_lengths 512 --batch_sizes 8 --repeat 2 --save_to_csv --runtime-method tensorrt
