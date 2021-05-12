import os
from datetime import datetime

def snap(opt):
    if not os.path.exists(opt.exp_dir):
        os.mkdir(opt.exp_dir)
    fig_path = os.path.join(opt.exp_dir,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.mkdir(fig_path)
    opt.FIGPATH = fig_path
    with open(os.path.join(opt.FIGPATH, 'options.txt'), 'w') as f:
        f.write(str(opt))