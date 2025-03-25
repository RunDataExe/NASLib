#!/bin/bash

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python naslib/optimizers/oneshot/gsparsity/runner.py --config-file naslib/optimizers/oneshot/gsparsity/config.yaml

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
