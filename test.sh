#!/bin/bash

test_dir="log_test_files"

run () {
	python3 main.py $1 --runtime-method $2 &> $3
}

rm -rf $test_dir &> /dev/null
mkdir $test_dir

rm *.onnx *.engine &> /dev/null
for batch in 1
do
	for runtime in nnfusion deepspeed tensorrt onnxruntime pytorch pytorch-jit
	do
		benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes $batch --repeat 2 --save_to_csv --check_equal"
		log_file="${test_logs}/_${runtime}.log"

		rm *.onnx *.engine &> /dev/null
		echo Testing: $runtime

		if run "$benchmark_settings" $runtime $log_file ; then
			echo  $runtime Passed!
		else
			cat $log_file
			echo  $runtime Failed...
		fi
	done
done
