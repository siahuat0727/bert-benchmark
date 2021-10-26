#!/bin/bash

nvprof_args="--csv --print-gpu-summary  --log-file nvprof%p.log --profile-child-processes"

rm *.onnx *.engine &> /dev/null
for batch in 1
do
	for runtime in tensorrt deepspeed onnxruntime pytorch pytorch-jit
	do
		benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes $batch --repeat 3 --save_to_csv --check_equal"
		python main.py $benchmark_settings --runtime-method $runtime
		rm *.onnx *.engine &> /dev/null
	done
done
