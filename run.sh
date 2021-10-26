#!/bin/bash

nvprof_args="--csv --print-gpu-summary  --log-file nvprof%p.log --profile-child-processes"

rm nvprof*.log &> /dev/null
rm -r log_files &> /dev/null
mkdir -p log_files
for batch in 1 2 4 8 16
do
	for runtime in pytorch pytorch-jit onnxruntime tensorrt deepspeed
	do
		benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes $batch --repeat 12 --save_to_csv"
		rm *.onnx &> /dev/null
		nvprof $nvprof_args python main.py $benchmark_settings --runtime-method $runtime
		ls -S nvprof*.log | xargs sed '/^==/d'> "nvprof#${runtime}#${batch}.csv"
		rename.ul nvprof "nvprof#${runtime}#${batch}#" nvprof*.log
		mv nvprof*.log log_files/
	done
done
