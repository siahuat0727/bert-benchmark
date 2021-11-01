#!/bin/bash

test_dir="log_test_files"

run () {
	echo Testing: $2, save log at $3
	python3 main.py $1 --runtime-method $2 &> $3
}

rm *.onnx *.engine &> /dev/null
rm -rf $test_dir &> /dev/null
mkdir $test_dir

echo Start tesing...\n

for batch in 1
do
	for runtime in deepspeed tensorrt onnxruntime nnfusion pytorch pytorch-jit
	do
		benchmark_settings="--models bert-base-cased --sequence_lengths 512 --batch_sizes $batch --repeat 2 --save_to_csv --check_equal"
		log_file="${test_dir}/${runtime}.log"

		rm *.onnx *.engine &> /dev/null

		if run "$benchmark_settings" $runtime $log_file ; then
			echo  $runtime Passed!
		else
			cat $log_file
			echo  $runtime Failed...
		fi
		echo
	done
done
