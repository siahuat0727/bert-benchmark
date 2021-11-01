#!/bin/bash

nvprof_args="--csv --print-gpu-summary  --log-file nvprof%p.log --profile-child-processes"

repeat=12

log_dir="log_run_files"

rm nvprof*.log &> /dev/null
rm -r $log_dir &> /dev/null
mkdir $log_dir

for batch in 1 2 4 8 16
do
	for runtime in pytorch pytorch-jit onnxruntime tensorrt deepspeed nnfusion
	do
		benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes $batch --repeat $repeat --save_to_csv"
		log_file="${log_dir}/${runtime}_${batch}.log"


		rm *.onnx *.engine &> /dev/null
		# TODO fix bug: when using nvprof, if it need to build engine then there will no csv output for python benchmark
		if [[ "$runtime" == "tensorrt" ]]; then
			python main.py $benchmark_settings --runtime-method $runtime  --repeat 1 &> /dev/null
		fi

		echo \n\nRun $runtime with $batch batch, save log at $log_file
		nvprof $nvprof_args python main.py $benchmark_settings --runtime-method $runtime &> $log_file
		tail $log_file

		# Generate nvprof csv from log
		ls -S nvprof*.log | xargs sed '/^==/d'> "nvprof#${runtime}#${batch}#$(($repeat + 1))#.csv"

		# Move log to log_dir
		rename.ul nvprof "nvprof#${runtime}#${batch}#$(($repeat + 1))#" nvprof*.log
		mv nvprof*.log $log_dir
	done
done
