#!/bin/bash

# Job name
#SBATCH --job-name=train_gpt_neo

# Output and error files
#SBATCH --output=train_output.log
#SBATCH --error=train_error.log

# Resources: number of nodes, tasks, and GPUs
#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 1
##SBATCH --gres=gpu:1
##SBATCH -t 23:00:00
#SBATCH --mem 64G
#SBATCH --job-name="Generating synthetic data"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kjmetzler@wpi.edu

# Activate your virtual environment
source /home/kjmetzler/DS552-1/venv/bin/activate

# Print the current environment for debugging
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"
echo "Pip packages:"
pip list

# Run the training script
python /home/kjmetzler/DS552-1/Project/train_script.py