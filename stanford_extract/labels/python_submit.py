import os

def python_submit(command, node = None, ngpus = 1):
    bash_file = open("./slurm.sh","w")
    bash_file.write(f'#!/bin/bash\n{command}')
    bash_file.close()
    if node == None:
        os.system(f'sbatch --ntasks=8 --ntasks-per-node=8 --gres=gpu:{ngpus} --output ./slurm/slurm-%j.out --mem=256000 --time=100-00:00:00 slurm.sh ')
    else:
        os.system(f'sbatch --ntasks=8 --ntasks-per-node=8 --gres=gpu:{ngpus} --output ./slurm/slurm-%j.out --mem=256000 --nodelist={node} --time=100-00:00:00 slurm.sh')
    os.remove("./slurm.sh")