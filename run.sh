#!/bin/bash

nvprof_args="--csv --print-gpu-summary  --log-file nvprof%p.log --profile-child-processes"

repeat=12
model="bert-base-cased"
seq_len=576

log_dir="log_run_files"

rm nvprof*.log &> /dev/null
rm -r $log_dir &> /dev/null
mkdir $log_dir

for batch in 1 2 4 8 16
do
	for runtime in deepspeed deepspeed-fp16 onnxruntime nnfusion pytorch pytorch-jit tensorrt tensorrt-plugin tensorrt-plugin-fp16
	do
		benchmark_settings="--models ${model} --sequence_lengths ${seq_len} --batch_sizes $batch --repeat $repeat --save_to_csv"
		log_file="${log_dir}/${runtime}_${batch}.log"

		rm *.onnx *.engine &> /dev/null
		# TODO fix bug: when using nvprof, if it need to build engine then there will no csv output for python benchmark
		if [[ "$runtime" == tensorrt* ]]; then
			python main.py $benchmark_settings --runtime-method $runtime  --repeat 1 &> /dev/null
		fi

		echo
		echo Run $runtime with $batch batch, save log at $log_file

		nvprof $nvprof_args python main.py $benchmark_settings --runtime-method $runtime &> $log_file
		tail $log_file

		nvprof_prefix="nvprof#${runtime}#${batch}#$(($repeat + 1))#${model}#${seq_len}#"

		# Generate nvprof csv from log
		ls -S nvprof*.log | xargs sed '/^==/d'> "${nvprof_prefix}.csv"

		# Move log to log_dir
		rename.ul nvprof $nvprof_prefix nvprof*.log
		mv nvprof*.log $log_dir
	done
done
