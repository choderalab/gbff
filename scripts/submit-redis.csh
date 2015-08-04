#!/bin/tcsh
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=160:31:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify queue
#PBS -q batch
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
#PBS -N redis-gbff
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
echo $HOSTNAME
setenv CELERY_CONFIG /cbio/jclab/projects/pgrinaway/gbff/hydration_energies/config.yaml
#setenv yank /cbio/jclab/home/pgrinaway/yank/yank
#setenv musashi /cbio/jclab/home/pgrinaway/musashi/yank/R82
pwd
#rm /cbio/jclab/home/pgrinaway/spark-1.0.2/logs/*master*
date
#sleep 8h
redis-server
date

