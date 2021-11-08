#!/bin/bash

test_dir="log_test_files"

run () {
	echo Testing: $2, save log at $3
	python3 main.py $1 --runtime-method $2 &> $3
}

rm *.onnx *.engine &> /dev/null
rm max_abs_error_*.txt &> /dev/null
rm -rf $test_dir &> /dev/null
mkdir $test_dir

echo
echo Start tesing...
echo

for batch in 1
do
	for runtime in deepspeed deepspeed-fp16 onnxruntime nnfusion pytorch pytorch-fp16 pytorch-jit tensorrt tensorrt-plugin tensorrt-plugin-fp16
	do
		benchmark_settings="--models bert-base-cased --sequence_lengths 576 --batch_sizes $batch --repeat 2 --save_to_csv --check_equal"
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
