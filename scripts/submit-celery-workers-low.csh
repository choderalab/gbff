#!/bin/tcsh
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=36:31:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify queue
#PBS -q lowpriority
#
# nodes: number of 8-core nodes
#   ppn: how many cores per node to use (1 through 8)
#       (you are always charged for the entire node)
#PBS -l mem=10gb,nodes=1:ppn=1
##PBS -l nodes=4,tpn=1,gpus=1:shared
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N workers-gbff-subset
#
# specify email
#PBS -M pgrinaway@gmail.com
#
# mail settings
#PBS -m n
#
# filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
#PBS -o /cbio/jclab/home/pgrinaway/

#cd "$PBS_O_WORKDIR"

#source /cbio/jclab/projects/musashi/musashi.tcsh

#echo | grep PYTHONPATH

setenv CELERY_CONFIG /cbio/jclab/projects/pgrinaway/gbff/hydration_energies/config.yaml
#setenv yank /cbio/jclab/home/pgrinaway/yank/yank
#setenv musashi /cbio/jclab/home/pgrinaway/musashi/yank/R82
pwd
#rm /cbio/jclab/home/pgrinaway/spark-1.0.2/logs/*master*
date
#sleep 8h
cd /cbio/jclab/projects/pgrinaway/gbff
celery -A hydration_energies worker -l info -c 1 --app=hydration_energies.app:app
#python $PROJECTS/gbff/gbff/parameterize-using-database.py --types $PROJECTS/gbff/gbff/parameters/gbsa-amber-mbondi2.types --parameters $PROJECTS/gbff/gbff/parameters/gbsa-amber-mbondi2.parameters --iterations 10000 -o $PROJECTS/gbff/gbff/output_subset/$PBS_ARRAYID-MCMC.h5 --database $FREESOLV_PATH/database.pickle --mol2 $FREESOLV_PATH/tripos_mol2 --subset 300
date

