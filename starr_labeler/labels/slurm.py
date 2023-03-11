from python_submit import python_submit

python_submit("python main.py")

'''
import os

def python_submit(command, node = None):
    # if ./slurm folder does not exist, create it
    if not os.path.exists('./slurm'):
        os.makedirs('./slurm')
    bash_file = open("./slurm.sh","w")
    bash_file.write(f'#!/bin/bash\n{command}')
    bash_file.close()
    os.system('sbatch --ntasks=8 --ntasks-per-node=8 --output ./slurm/slurm-%j.out --mem=512000 --time=30-00:00:00 slurm.sh')
    os.remove("./slurm.sh")

python_submit("python main.py")
'''

