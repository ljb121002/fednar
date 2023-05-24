
##################################
### Main: Server side optimization and testing
#################################
# Initiate the NN

from util_data import *
from util_models import *
from util_general import *
from configs import *
import time




parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--num_clients', type=int, required=True)
parser.add_argument('--num_participating_clients', type=int, required=True)
parser.add_argument('--num_rounds', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--cp', type=float, required=False, default=20)
parser.add_argument('--filename', type=str, required=True)
parser.add_argument('--no_wandb', default=False, action='store_true')


parser.add_argument('--eta_l', default=0.01, type=float, required=True)
parser.add_argument('--eta_g', default=1.0, required=True)
parser.add_argument('--epsilon', default=0.0, type=float, required=True)
parser.add_argument('--mu', default=0.0, type=float, required=True)

parser.add_argument('--weight_decay', default=1e-4, type=float, required=True)
parser.add_argument('--l2_reg', default=0.0, type=float, required=False, help='l2 loss added on loss')
parser.add_argument('--max_norm', default=10.0, type=float, required=False)
parser.add_argument('--decay', default=0.998, type=float, required=False)
parser.add_argument('--batch_size', default=50, type=int)

parser.add_argument('--use_grid', default=False, action='store_true')
parser.add_argument('--no_model_avg', default=False, action='store_true')
parser.add_argument('--use_nar', default=False, action='store_true')

# shakespeare
parser.add_argument('--num_layer', default=6, type=int)
parser.add_argument('--drop_out', default=0.1, type=float)

args_required = parser.parse_args()

if args_required.use_nar:
    args_required.l2_reg = args_required.weight_decay
    args_required.weight_decay = 0.0

seed = args_required.seed
dataset = args_required.dataset
algorithm = args_required.algorithm
model = args_required.model
num_clients = args_required.num_clients
num_participating_clients = args_required.num_participating_clients
num_rounds = args_required.num_rounds
alpha = args_required.alpha

print_every_test = 1
print_every_train = 1



if args_required.filename == None:
  filename = "results_"+str(seed)+"_"+algorithm+"_"+dataset+"_"+model+"_"+str(num_clients)+"_"+str(num_participating_clients)+"_"+str(num_rounds)+"_"+str(alpha)
else:
  filename = args_required.filename
filename_txt = filename + ".txt"


if (dataset == 'shakespeare'):
  n_c = 80
else: n_c = 10


np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

dict_results = {} ###dictionary to store results for all algorithms

###Default training parameters for all algorithms

args={
"bs":args_required.batch_size,   ###batch size
"cp":args_required.cp,   ### number of local steps
"device":'cuda',
"rounds":num_rounds, 
"num_clients": num_clients,
"num_participating_clients":num_participating_clients,
"use_wandb": not args_required.no_wandb,
"l2_reg": args_required.l2_reg,
"dataset": args_required.dataset,
"drop_out": args_required.drop_out,
"num_layer": args_required.num_layer
}

if args['use_wandb']:
    import wandb
    wandb.init(config=args_required, entity='entity', name=filename, project='fednar')


if dataset == 'shakespeare':
    data_obj = ShakespeareObjectCrop_noniid('./data/shakespeare/', 'shakepeare_nonIID')
    dataset_train = data_obj.clnt_data
    dataset_test_global = data_obj.tst_data

    net_glob_org = shake_transf(args).to(args['device'])
else:
    dataset_train, dataset_test_global = get_dataset(num_clients, n_c, alpha, True)
    net_glob_org = get_model(model,n_c).to(args['device'])


d = parameters_to_vector(net_glob_org.parameters()).numel()


decay =  args_required.decay
max_norm = args_required.max_norm
weight_decay = args_required.weight_decay

n = len(dataset_train)
print ("No. of clients", n)

# ratio of local dataset
p = np.zeros((n))
for i in range(n):
    p[i] = len(dataset_train[i])
p = p/np.sum(p)

algs = [algorithm]
for alg in algs:

    net_glob = copy.deepcopy(net_glob_org)
    net_glob.train()

    filename_model_alg = filename + "-" + alg + ".pt"
    dict_results[alg] = {}
    loss_t = []
    train_loss_algo_tmp = []
    train_acc_algo_tmp = []
    test_loss_algo_tmp = []
    test_acc_algo_tmp = []
    eta_g_tmp = []
    
    w_vec_estimate = torch.zeros(d).to(args['device'])

    grad_mom = torch.zeros(d).to(args['device'])  # fedadam / fedeavgm
    mem_mat = None
    if dataset != 'personachat':
        mem_mat = torch.zeros((n, d)).to(args['device'])  # scaffold
    delta = torch.zeros(d).to(args['device']) # fedadam
    grad_norm_avg_running = 0 # fedavgm (exp)


    local_lr = args_required.eta_l
    global_lr = args_required.eta_g
    epsilon = args_required.epsilon
    mu = args_required.mu
    
    grad_square_avg = torch.ones(d).to(args['device'])
    for t in range(0,args['rounds']+1):        
        print ("Algo ", alg, " Round No. " , t)
        args_hyperparameters, local_lr, epsilon = get_hyper_round(decay, local_lr, epsilon, mu, weight_decay, 
                                                                  global_lr, max_norm, dataset, t)      
        
        S = args['num_participating_clients']
        ind = np.random.choice(n,S,replace=False)

        grad_avg = torch.zeros(d).to(args['device'])
        grad_square_avg_curr = torch.zeros(d).to(args['device'])
        w_init = parameters_to_vector(net_glob.parameters()).to(args['device'])
        
        grad_norm_sum = 0
        p_sum = 0

        c = torch.zeros((d,)).to(args['device']) # scaffold  
        
        if(alg=='scaffold' or alg=='scaffold(exp)'):
            for i in range(n):
                c = c+ p[i]*mem_mat[i]

        all_norms = []
        # local traininig
        for (ii, i) in enumerate(ind):
            grad = get_grad(copy.deepcopy(net_glob),args, args_hyperparameters, dataset_train[i], alg, i,  mem_mat, c, t, grad_square_avg)
            grad, norms = grad
            for norm in norms:
              all_norms.append(norm) 
            grad_norm_sum += p[i]*torch.linalg.norm(grad)**2
            grad_avg = grad_avg + p[i]*grad
            grad_square_avg_curr = grad_square_avg_curr + p[i]*torch.square(grad)
            p_sum += p[i]
        
        grad_square_avg = copy.deepcopy(grad_square_avg_curr)

        # momentum, parameter update 
        with torch.no_grad():
            grad_avg = grad_avg/p_sum
            grad_norm_avg = grad_norm_sum/p_sum
            
            if(alg=='fedavgm' or alg=='fedavgm(ep)'):
              grad_norm_avg_running = grad_norm_avg +0.9*0.5*grad_norm_avg_running # fedavgm
              grad_avg = grad_avg + 0.9*grad_mom
              grad_mom = grad_avg

            if(alg=='fedadam'):
              grad_avg = 0.1*grad_avg + 0.9*grad_mom
              grad_mom = grad_avg
              delta = 0.01*grad_avg**2 + 0.99*delta
              grad_avg_normalized = grad_avg/(0.1)
              delta_normalized = delta/(0.01)
              grad_avg = (grad_avg_normalized/torch.sqrt(delta_normalized + epsilon))
            
            grad_avg_norm = torch.linalg.norm(grad_avg)**2

            # global lr
            eta_g = args_hyperparameters['eta_g']
            if(eta_g == 'adaptive'):
                if(alg!='fedavgm(exp)'):
                    eta_g = (0.5*grad_norm_avg/(grad_avg_norm + S*epsilon)).cpu()
                else:
                    eta_g = (0.5*grad_norm_avg_running/(grad_avg_norm + S*epsilon)).cpu()
                eta_g = max(1,eta_g)
            else:
               eta_g = float(eta_g)

            eta_g_tmp.append(eta_g)
            # parameter update
            w_vec_prev = w_vec_estimate
            w_vec_estimate = parameters_to_vector(net_glob.parameters()) + eta_g*grad_avg
            vector_to_parameters(w_vec_estimate,net_glob.parameters())

            # eval 
            net_eval = copy.deepcopy(net_glob)

            if(t>0):
              w_vec_avg = (w_vec_estimate + w_vec_prev)/2
            else:
              w_vec_avg = w_vec_estimate


            if(alg=='fedexp' or alg=='scaffold(exp)' or alg=='fedprox(exp)' or alg=='fedavgm(exp)'):
                if not args_required.no_model_avg:
                    vector_to_parameters(w_vec_avg, net_eval.parameters())  
            
            if(t%print_every_test==0):
                if t%print_every_train==0:
                    # average training loss/acc of the global model on clients
                    sum_loss_train = 0
                    sum_acc_train = 0
                    train_ind = list(range(n))
                    for i in train_ind:
                      test_acc_i, test_loss_i = test_all(net_eval, dataset_train[i],args, is_train_set=True)
                      sum_loss_train += test_loss_i
                      sum_acc_train += test_acc_i

                    sum_loss_train = sum_loss_train/len(train_ind)
                    sum_acc_train = sum_acc_train/len(train_ind)
                    print ("Training Loss ", sum_loss_train, "Training Accuracy ", sum_acc_train)        
                    train_loss_algo_tmp.append(sum_loss_train)
                    train_acc_algo_tmp.append(sum_acc_train)
                
                # test loss / acc
                sum_loss_test = 0
                sum_acc_test = 0
                test_acc_i, test_loss_i = test_all(net_eval, dataset_test_global,args)
                sum_loss_test = test_loss_i
                sum_acc_test = test_acc_i
                print ("Test Loss", sum_loss_test, "Test Accuracy ", sum_acc_test)

                test_loss_algo_tmp.append(sum_loss_test)
                test_acc_algo_tmp.append(sum_acc_test)
          

        if args['use_wandb']:
            if len(all_norms) != 0:
                norm_mean = sum(all_norms) / len(all_norms)
            else:
                norm_mean = max_norm
            wandb.log({'train_loss': float(sum_loss_train),
                      'train_acc': float(sum_acc_train),
                      'test_loss': float(sum_loss_test),
                      'test_acc': float(sum_acc_test),
                      'global_lr': float(eta_g),
                      'grad_norm_mean': float(norm_mean),
                      'grad_clip_num': len(all_norms),
                      'grad_square_max': float(torch.max(grad_square_avg)),
                      'grad_square_sum': float(torch.sum(grad_square_avg))})

        dict_results[alg][alg+"_training_loss"] = train_loss_algo_tmp
        dict_results[alg][alg+"_training_accuracy"] = train_acc_algo_tmp
        dict_results[alg][alg+"_test_loss"] = test_loss_algo_tmp
        dict_results[alg][alg+"_testing_accuracy"] = test_acc_algo_tmp
        dict_results[alg][alg+"_global_learning_rate"] = eta_g_tmp

        try:
          torch.save(net_glob, filename_model_alg)
        except:
          torch.save(net_glob.state_dict(), filename_model_alg)

        with open(filename_txt, 'w') as f:    
          for i in dict_results.keys():
            for key, value in dict_results[i].items():
              f.write(key+" ")
              f.write(str(value))
              f.write("\n")