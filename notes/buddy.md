# Organisational Structure
- Is Nikis group in Theis lab?
- There are 100 people affiliated wiht Theis lab, 20 core team
- Mo is the team leader of the perturbation team

If I want to reach out, everyone should be chill
**For administrative questions:** Nina Monika Fischer

# IT setup
- IT channel on mattermost for quick questions
- If there are bigger issues about the cluster, raise an issue on gitlab
- Change PW here & some stuff: https://citrix.helmholtz-muenchen.de/Citrix/HMGUWeb/

## cluster
Should have been set up automatically
**Documentation** https://ascgitlab.helmholtz-muenchen.de/ICB/ICB_IT/-/wikis/ICB-compute-cluster
The cluster is a separate one, it's not that crowded.

### VPN setup
**Some Info** https://nip.helmholtz-muenchen.de/ict/netzwerk/vpn/index.html
**Request Access** https://spit.helmholtz-muenchen.de/ssc/app#/

**VPN server address**
https://vpnportal.helmholtz-muenchen.de

**Step-by-step**
fill in VPN and computer registration form for IT
get Cisco AnyConnect and connect to vpnportal.helmholtz-muenchen.de
open Terminal and execute ssh -o TCPKeepAlive=no -o ServerAliveInterval=15 -L 8888:localhost:8889 andreas.uhlmann@icb-sarah.scidom.de
copy submit_interactive_gpu.sh and run_jupyter.sh from oksana.bilous directory executing cp submit_interactive_gpu.sh /home/icb/andreas.uhlmann and cp run_jupyter.sh /home/icb/andreas.uhlmann
download Miniconda3 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
install Miniconda3 sh Miniconda3-latest-Linux-x86_64.sh
reopen ssh session using source .bashrc
create Conda environment conda create --name py38 python=3.8 (conda activate py38)
make directory mkdir logs
generate keys as described here https://ascgitlab.helmholtz-muenchen.de/ICB/ICB_IT/-/wikis/public%20key%20authentication
run ssh andreas.uhlmann@vicb-submit-01
run sbatch submit_interactive_gpu.sh
go to logs and run head interactive_XXX.err gives link to Jupiter notebook

**Folders**
There should be a folder in my name, if there is none I should ask for it

### Computing Resources
No CPU limit, might just get busy

Dear All,

Following changes will be made based on feedback I received from users:

1) 

To Increase CPU from 4 => 8 cores on Login/Submit nodes.
To Increase RAM from 8GB => 16GB on Login/Submit nodes.
Please Note: Login/Submit nodes are used ONLY for login and submitting jobs. Heavy-duty tasks should run/submitted on compute nodes in cluster.
2) Enforcing/Limiting resources on GPU nodes for maximum utilization:

2.1) On interactive_gpu_p :

06 CPU limit
16 GB RAM
12 hours max job duration.
New Quality of service(QOS) will be created for this:
qos name: interactive_gpu
To submit job, user need to specify --qos=interactive_gpu
2.2) On gpu_p :

06 CPU limit
90 GB RAM
02 Days max job duration.
New Quality of service (QOS) will be created for this:
qos name: gpu
To submit job user, user need to specify --qos=gpu
IMP:
User needs to be part of above qos to submit jobs in gpu nodes. If your account is not associated with gpu qos OR missed out during these changes, please let me know.


## software
- There are some classic object types in single cell data
- Python: Scanpy, R: Seurat

## Data
There are some datasets already included in scanpy
For new data, one would do the preprocessing themselves, typically


# TODO:
Reach out to Nina about access to Gitlab for cluster documentation