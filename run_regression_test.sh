#!/usr/bin/env bash
#$ -q RAM.q
#$ -cwd
#$ -N regression_events
#$ -o tests/regression_tests.log
source ~/.bashrc
cd ~/event_creation
rm *.png
source activate event_creation

python -m event_creation.tests.regression_tests --db-root=/scratch/db_root
