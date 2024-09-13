import numpy as np
import torch

CONFIG_SUPERNET = {
    'dataloading' : {
        'batch_size' :256,
        'w_share_in_train' : 0.8
    },
    'train_settings' : {
        'cnt_epochs' : 30,
        'train_thetas_from_the_epoch': 5,
        'print_freq' : 200,
        'path_to_save_model' : './best_model_Supernet.pth',
        # for Gumbel Softmax
        'init_temperature' : 1.0,
        'exp_anneal_rate' : np.exp(-0.045)
    },
    'logging' : {
        'path_to_log_file' : './supernet_logs/logger/',
        'path_to_tensorboard_logs' : 'ADMIN/supernet_logs/tb'
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 0.1,                  # default 0.1
        'w_momentum' : 0.9,
        'w_weight_decay' : 1e-4,
        # Adam parameters for thetas
        'thetas_lr' : 0.9,            # default 0.01
        'thetas_weight_decay' : 5 * 1e-4            # default 5 * 1e-4
    }
    
}

CONFIG_FBNET = {
    'logging' : {
        'path_to_log_file' : '/home/fding/ADMIN/FBNet_logs/logger/',
        'path_to_tensorboard_logs' : '/home/fding/ADMIN/FBNet_logs/tb'
    },
    'dataloading' : {
        'batch_size' : 10,
        'train_portion' : 0.8,
    },
    'optimizer' : {
        'lr' : 0.001,                     # default 0.1
        'momentum' : 0.6,
        'weight_decay' : 6*1e-4
    },
    'train_settings' : {
        'print_freq' : 100, # show logging information
        # 'path_to_save_model' : 'ADMIN/best_model_FBNet.pth',
        'cnt_epochs' : 30, #
        # YOU COULD USE 'CosineAnnealingLR' or 'MultiStepLR' scheduler
        'scheduler' : 'MultiStepLR',
        ## CosineAnnealingLR settings
        'eta_min' : 0.001,
        ## MultiStepLR settings
        'milestones' : [60, 90, 120], # [90, 180, 270], # decay 10x at 90, 180, and 270 epochs
        'lr_decay' : 0.1
    }
}