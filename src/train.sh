#!/bin/sh
if test "$#" -ne 5; then
	echo "Usage: ./train.sh batch-size dropout task-id memory hops"
	exit 1
fi

batch_size=$1
dropout=$2
task=$3
mem=$4
hops=$5

mkdir task$task

for mode in GREEDY GT MAPO HYBRID
do
	for pgen_weight in 1 1.5
	do
		for emb_size in 64 128
		do
			for lr in 0.001 0.0005
			do
				jbsub -queue x86_24h -cores 1x1+1 -proj S${task}d${dropout}${mode} -name e${emb_size}lr${lr}p${pgen_weight} -mem $mem -out task$task/train.${mode}.lr.$lr.e.$emb_size.h.$hops.d.$dropout.pw.${pgen_weight}.out -err task$task/train.${mode}.lr.$lr.e.$emb_size.h.$hops.d.$dropout.pw.${pgen_weight}.err /u/diraghu1/anaconda3/envs/tf/bin/python single_dialog.py --train True --task_id $task --learning_rate $lr --hops $hops --embedding_size $emb_size --batch_size $batch_size --p_gen_loss True  --word_drop_prob 0.${dropout} --p_gen_loss_weight ${pgen_weight} --rl_mode ${mode}
			done
		done
	done
done
