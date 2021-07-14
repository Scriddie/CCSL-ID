""" full sachs data """
from data_generation_rwd import *

if __name__ == '__main__':
    utils.create_folder('src/seq/data')
    utils.overwrite_folder('src/seq/data/sachs_custom')
    utils.overwrite_folder('src/seq/data/sachs_custom/figures')

    opt = Namespace()
    opt.base_path = './data/sachs/Data Files/'
    # activation PKA led to good results so far, think about it!
    opt.int_files = [
        # TODO they only use inhibitions! Is this the way to go?
        'activation-PKA', ##
        'inhibition-AKT',
        'inhibition-MEK',
        'activation-PKC', ##
        'inhibition-PKC',
        'inhibition-PIP2',
        'inhibition2-PIP3',  # TODO are we sure about this target?
    ]
    opt.int_vars = list(set([i.split('-')[-1] for i in opt.int_files]))
    opt.obs_files = ['general1', 'general2']
    opt.obs_vars = ['ERK', 'RAF', 'JNK', 'P38', 'PLCG']
    # TODO rename PLCG to PLC?
    opt.out_dir = 'src/seq/data/sachs_custom'
    opt.vars_ord = opt.int_vars + opt.obs_vars
    opt.standardize = True
    opt.standardize_globally = True
    opt.demean = True
    opt.log_transform = False
    opt.remove_outliers = False

    run(opt)