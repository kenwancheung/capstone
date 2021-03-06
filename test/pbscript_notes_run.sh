#!/bin/bash

#PBS -l nodes=1:ppn=16:gpus=1 -q gpu
#PBS -l mem=128Gb
#PBS -l walltime=72:00:00
#PBS -j oe

module load singularity tensorflow/tensorflow:1.4.0-rc0-gpu-py3
$SINGULARITY_EXEC $CONTAINER_PATH python3 /gpfs/data/ildproject-share/capstone//src/othersamples/seq2seq/seq2seq_notes_gpu.py

exit
