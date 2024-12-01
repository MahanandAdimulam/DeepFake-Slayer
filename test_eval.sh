#!/bin/bash -l

#SBATCH --job-name=FaceForensicsTestAndEval		# Name for your job
#SBATCH --comment="Testing and Evaluation of the model on binary mask"		# Comment for your job

#SBATCH --account=defake		# Project account to run your job under
#SBATCH --partition=tier3		# Partition to run your job on

#SBATCH --output=%x_%j.out		# Output file
#SBATCH --error=%x_%j.err		# Error file

#SBATCH --mail-user=slack:@U03V0JPFMCZ	# Slack username to notify
#SBATCH --mail-type=ALL			# Type of slack notifications to send

#SBATCH --time=2-00:00:00		# Time limit
#SBATCH --nodes=4			# How many nodes to run on
#SBATCH --ntasks=4			# How many tasks per node
#SBATCH --cpus-per-task=2		# Number of CPUs per task
#SBATCH --mem-per-cpu=24g		# Memory per CPU
#SBATCH --gres=gpu:a100:1

source /shared/rc/defake/Deepfake-Slayer/deepfake/bin/activate	# Run the command hostname
# spack env activate default-ml-23110801
python /shared/rc/defake/Deepfake-Slayer/scripts/DeepFake-Slayer/test.py
python /shared/rc/defake/Deepfake-Slayer/scripts/DeepFake-Slayer/eval.py
deactivate
# spack env deactivate