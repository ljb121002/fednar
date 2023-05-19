def get_hyper_round(decay, local_lr, epsilon, mu, weight_decay, global_lr, max_norm, dataset, t):
    local_lr = decay*local_lr
    
    epsilon = decay*decay*epsilon


    args_hyperparameters = {'mu': mu, 'eta_l':local_lr, 'decay': decay, 'weight_decay': weight_decay, 
                            'eta_g': global_lr,  
                            'max_norm': max_norm, 'epsilon': epsilon, 'use_augmentation':True}
    
    if(dataset=='CIFAR10' or dataset=='CIFAR100' or dataset=='CINIC10'):
        args_hyperparameters['use_augmentation'] = True
    else:
        args_hyperparameters['use_augmentation'] = False

    return args_hyperparameters, local_lr, epsilon