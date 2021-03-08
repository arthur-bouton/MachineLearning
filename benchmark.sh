#!/bin/bash

if [ $# -ge 1 ]; then
	training_script=$1
else
	>&2 echo Please specify the training script to use.
	exit 1
fi

data_file=${training_script%.py}.dat

n_runs=100


trap 'echo;exit' INT
for i in $( seq $n_runs ) ; do
	echo [$training_script] Run $i
	python $training_script 2> /dev/null >> $data_file
done
