import torch
import numpy asn np
import os
import shutil

def init():
    torch.set_default_tensor_type('torch.FloatTensor')

def set_seed():
    torch.manual_seed(0)
    np.random.seed(0)

def snap(opt):
    exp_dir = os.path.join('src', 'experiments') 
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    fig_path = os.path.join(exp_dir,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.mkdir(fig_path)
    opt.SAVEPATH = fig_path
    with open(os.path.join(opt.SAVEPATH, 'options.txt'), 'w') as f:
        f.write(str(opt))

def create(path):
    if not os.path.exists(path):
        os.mkdir(path)

def overwrite():
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)