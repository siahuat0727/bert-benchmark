#!/bin/bash

nvprof_args="--csv --print-gpu-summary  --log-file nvprof%p.log --profile-child-processes"

repeat=5

log_dir="log_run_files"

rm nvprof*.log &> /dev/null
rm -r $log_dir &> /dev/null
mkdir $log_dir

for batch in 1 2 4 8 16
do
	for runtime in pytorch pytorch-jit onnxruntime tensorrt deepspeed nnfusion
	do
		benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes $batch --repeat $repeat --save_to_csv"
		rm *.onnx *.engine &> /dev/null
		nvprof $nvprof_args python main.py $benchmark_settings --runtime-method $runtime
		ls -S nvprof*.log | xargs sed '/^==/d'> "nvprof#${runtime}#${batch}.csv"
		rename.ul nvprof "nvprof#${runtime}#${batch}#$(($repeat + 1))#" nvprof*.log
		mv nvprof*.log $log_dir
	done
done
