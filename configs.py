global eta_l_fedavg
global eta_l_fedexp
global eta_l_scaffold
global eta_l_scaffold_exp
global eta_l_fedadagrad
global eta_l_fedprox
global eta_l_fedprox_exp
global eta_l_fedadam
global eta_l_fedavgm
global eta_l_fedavgm_exp

global eta_g_fedavg
global eta_g_scaffold
global eta_g_fedprox
global eta_g_fedadagrad
global eta_g_fedadam
global eta_g_fedavgm

global psilon_fedexp
global epsilon_scaffold_exp
global epsilon_fedprox_exp
global epsilon_fedavgm_exp

global epsilon_fedadagrad
global epsilon_fedadam

def get_configs(dataset):

    if(dataset=='CIFAR10'):
        eta_l_fedavg = 0.01
        eta_l_fedexp = 0.01
        eta_l_scaffold = 0.01
        eta_l_scaffold_exp = 0.01
        eta_l_fedadagrad = 0.01
        eta_l_fedprox = 0.01
        eta_l_fedprox_exp = 0.01
        eta_l_fedadam = 0.01
        eta_l_fedavgm = 0.01
        eta_l_fedavgm_exp = 0.01

        eta_g_fedavg = 1
        eta_g_scaffold = 1
        eta_g_fedprox = 1
        eta_g_fedadagrad = 0.1
        eta_g_fedadam = 0.1
        eta_g_fedavgm = 1
    
    

        epsilon_fedexp = 0.001
        epsilon_scaffold_exp = 0.001
        epsilon_fedprox_exp = 0.001
        epsilon_fedavgm_exp = 0.001
    
    elif(dataset=='CINIC10'):
        eta_l_fedavg = 0.01
        eta_l_fedexp = 0.01
        eta_l_scaffold = 0.01
        eta_l_scaffold_exp = 0.01
        eta_l_fedadagrad = 0.01
        eta_l_fedprox = 0.01
        eta_l_fedprox_exp = 0.01
        eta_l_fedadam = 0.01
        eta_l_fedavgm = 0.01
        eta_l_fedavgm_exp = 0.01

        eta_g_fedavg = 1
        eta_g_scaffold = 1
        eta_g_fedprox = 1
        eta_g_fedadagrad = 0.1
        eta_g_fedadam = 0.1
        eta_g_fedavgm = 1
        
    


        epsilon_fedexp =  0.001
        epsilon_scaffold_exp = 0.001
        epsilon_fedprox_exp = 0.001
        epsilon_fedavgm_exp = 0.001

    elif(dataset=='CIFAR100'):
        eta_l_fedavg = 0.01
        eta_l_fedexp = 0.01
        eta_l_scaffold = 0.01
        eta_l_scaffold_exp = 0.01
        eta_l_fedadagrad = 0.01
        eta_l_fedprox = 0.01
        eta_l_fedprox_exp = 0.01
        eta_l_fedadam = 0.01
        eta_l_fedavgm = 0.01
        eta_l_fedavgm_exp = 0.01

        eta_g_fedavg = 1
        eta_g_scaffold = 1
        eta_g_fedadagrad = 0.1
        eta_g_fedadam = 0.1
        eta_g_fedprox = 1
        eta_g_fedavgm = 1
        

        epsilon_fedexp = 0.001
        epsilon_scaffold_exp = 0.001
        epsilon_fedprox_exp = 0.001
        epsilon_fedavgm_exp = 0.001


    elif(dataset=='EMNIST'):
        eta_l_fedavg = 0.1
        eta_l_fedexp = 0.1
        eta_l_scaffold = 0.1
        eta_l_scaffold_exp = 0.1
        eta_l_fedadagrad = 0.1
        eta_l_fedprox = 0.1
        eta_l_fedprox_exp = 0.1
        eta_l_fedadam = 0.1
        eta_l_fedavgm = 0.316
        eta_l_fedavgm_exp = 0.316

        eta_g_fedavg = 1
        eta_g_scaffold = 1
        eta_g_fedadagrad = 0.316
        eta_g_fedadam = 0.316
        eta_g_fedprox = 1
        eta_g_fedavgm = 1
        

        epsilon_fedexp = 0.1
        epsilon_scaffold_exp = 0.1
        epsilon_fedprox_exp = 0.1
        epsilon_fedavgm_exp = 0.1


        epsilon_fedadagrad = 0.0316
        epsilon_fedadam = 0.0316
        


    mu_fedprox = 0

    if(dataset=='CIFAR10'):
        mu_fedprox = 0.1
    
    if(dataset=='CINIC10'):
        mu_fedprox = 1
    
    if(dataset=='EMNIST'):
        mu_fedprox = 0.001
    
    if (dataset=='CIFAR100'):
        mu_fedprox = 0.001



    eta_l_algs = {'fedavgm(exp)': eta_l_fedavgm_exp, 
                'fedavgm': eta_l_fedavgm,
                'fedadam':eta_l_fedadam, 
                'fedprox':eta_l_fedprox, 
                'fedprox(exp)': eta_l_fedprox_exp, 
                'fedavg': eta_l_fedavg, 
                'fedadagrad': eta_l_fedadagrad, 
                'fedexp': eta_l_fedexp, 
                'scaffold': eta_l_scaffold, 
                'scaffold(exp)': eta_l_scaffold_exp,

                'feddyk': 0.01,
                'fedavg(mom)': 0.01,
                'fedcm': 0.01}

    eta_g_algs = {'fedavgm(exp)': 'adaptive', 
                'fedavgm': eta_g_fedavgm, 
                'fedadam':eta_g_fedadam, 
                'fedprox':eta_g_fedprox, 
                'fedprox(exp)': 'adaptive', 
                'fedavg':eta_g_fedavg, 
                'fedadagrad': eta_g_fedadagrad, 
                'fedexp': 'adaptive', 
                'scaffold': eta_g_scaffold, 
                'scaffold(exp)': 'adaptive', 
                
                'feddyk': 1.0,
                'fedavg(mom)': 1.0,
                'fedcm': 1.0}

    epsilon_algs = {'fedavgm(exp)': epsilon_fedavgm_exp, 
                    'fedavgm': 0, 
                    'fedadam': 0, 
                    'fedprox':0, 
                    'fedprox(exp)':epsilon_fedprox_exp, 
                    'fedavg': 0, 
                    'fedadagrad':0, 
                    'fedexp':epsilon_fedexp, 
                    'scaffold': 0, 
                    'scaffold(exp)': epsilon_scaffold_exp,

                    'feddyk': 0,
                    'fedavg(mom)': 0,
                    'fedcm': 0}

    mu_algs = {'fedavgm(exp)': 0, 
            'fedavgm': 0, 
            'fedadam':0, 
            'fedprox': mu_fedprox, 
            'fedprox(exp)': mu_fedprox, 
            'fedavg':0, 
            'fedadagrad':0, 
            'fedexp':0, 
            'scaffold':0, 
            'scaffold(exp)':0,
            
            'feddyk': 0,
            'fedavg(mom)': 0,
            'fedcm': 0}
    
    return eta_l_algs, eta_g_algs, epsilon_algs, mu_algs


def get_hyper_round(decay, local_lr, epsilon, mu, weight_decay, global_lr, use_gradient_clipping, max_norm, dataset, t):
    local_lr = decay*local_lr
    # local_lr = decay ** t * 0.01 / (6.5 * t / 2000 + 1)
    epsilon = decay*decay*epsilon

    # if t < 500:
        # weight_decay = 0.1
    # elif t < 1000:
    #     weight_decay = 0.01
    # else:
    #     weight_decay = 0.01

    # ============ weight_decay ==================
    # weight_decay = decay * weight_decay
    # ============ weight_decay ==================

    args_hyperparameters = {'mu': mu, 'eta_l':local_lr, 'decay': decay, 'weight_decay': weight_decay, 
                            'eta_g': global_lr, 'use_gradient_clipping': use_gradient_clipping, 
                            'max_norm': max_norm, 'epsilon': epsilon, 'use_augmentation':True}
    
    if(dataset=='CIFAR10' or dataset=='CIFAR100' or dataset=='CINIC10'):
        args_hyperparameters['use_augmentation'] = True
    else:
        args_hyperparameters['use_augmentation'] = False

    return args_hyperparameters, local_lr, epsilon