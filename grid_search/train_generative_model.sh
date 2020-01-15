#!/bin/bash
source /data/bats/envs/esafranc/bin/activate
export PYTHON=/data/bats/envs/esafranc/lib/python3.7
export DATA_PATH=../tutorials/introduction/output/tmp


for MODEL in 'link_hmm'
do
    for ACC_PRIOR in 0.5 1 5 10 50 100 500
    do
        for BALANCE_PRIOR in 0.5 1 5 10 50 100 500
        do
            export MODEL
            export ACC_PRIOR
            export BALANCE_PRIOR

            export OUT=/output/${MODEL}/${ACC_PRIOR}_${BALANCE_PRIOR}

            # Add your code here to submit the train_generative_model script to a grid cluster (e.g., SGE cluster)
            # train_generative_model.sh
            qsub -V -cwd -j y -o $OUT -b y -l vf=4G,long train_generative_model.sh
        done
	done
done
