#!/bin/bash

#PBS -l nodes=1:ppn=28
#PBS -l mem=124
#PBS -j oe

module load singularity tensorflow/tensorflow:1.4.0-rc0-gpu-py3
$SINGULARITY_EXEC $CONTAINER_PATH python3 /gpfs/data/ildproject-share/capstone/src/othersamples/seq2seq/seq2seq_notes.py

exit
