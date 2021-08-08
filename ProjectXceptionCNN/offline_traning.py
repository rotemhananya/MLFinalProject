import os
import sys


os.system("nohup " + sys.executable + " Training.py")
# nohup bash -c'/home/dorpi/.aconda/envs/my_env/bin/python run_kl.py' &
# os.system("#SBATCH --cpus-per-task=1")
# os.system("#SBATCH --mem=120G")
# os.system("#SBATCH --ntasks-per-node=2")
# os.system(sys.executable + " run_kl.py")
# print(sys.executable)
# os.system("")
# os.system("nohup bash -c '" + sys.executable + " run_kl.py >result.txt' &")