import os


def python_submit(command, node=None, ngpus=0):
    if not os.path.exists("./slurm"):
        os.makedirs("./slurm")
    bash_file = open("./slurm.sh", "w")
    bash_file.write(f"#!/bin/bash\n{command}")
    bash_file.close()
    if node is None:
        os.system(
            f"sbatch --ntasks=4 --ntasks-per-node=4 --gres=gpu:{ngpus}"
            f" --output ./slurm/slurm-%j.out --time=100-00:00:00 slurm.sh "
        )
    else:
        os.system(
            f"sbatch --ntasks=4 --ntasks-per-node=4 --gres=gpu:{ngpus}"
            f" --output ./slurm/slurm-%j.out --nodelist={node} --time=100-00:00:00 slurm.sh"
        )
    os.remove("./slurm.sh")
