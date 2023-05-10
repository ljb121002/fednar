
##################################
### Main: Server side optimization and testing
#################################
# Initiate the NN

from sys import float_info
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
parser.add_argument('--l2_reg', default=0.0, type=float, required=True, help='l2 loss added on loss, use with grad clip')
parser.add_argument('--adjustLR_coef', default=1.0, type=float, required=False)
parser.add_argument('--max_norm', default=10.0, type=float, required=False)
parser.add_argument('--eps_elewd', default=1e-5, type=float, required=False)
parser.add_argument('--decay', default=0.998, type=float, required=True)
parser.add_argument('--local_momentum', default=0.0, type=float)
parser.add_argument('--batch_size', default=50, type=int)

parser.add_argument('--use_grid', default=False, action='store_true')
parser.add_argument('--use_log_etag', default=False, action='store_true')
parser.add_argument('--no_model_avg', default=False, action='store_true')
parser.add_argument('--use_gradient_clipping', default=False, action='store_true')
parser.add_argument('--adjust_lr', default=False, action='store_true')

# personachat
parser.add_argument('--lm_coef', default=1.0, type=float)
parser.add_argument('--mc_coef', default=1.0, type=float)
parser.add_argument('--use_debug_max', default=False, action='store_true')
parser.add_argument('--num_candidates', default=2, action='store_true')
parser.add_argument('--max_history', default=2, action='store_true')

# shakespeare
parser.add_argument('--num_layer', default=6, type=int)
parser.add_argument('--drop_out', default=0.1, type=float)

args_required = parser.parse_args()

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


if(dataset=='CIFAR100'):
  n_c = 100
elif (dataset == 'EMNIST'):
  n_c = 62
elif (dataset == 'personachat'):
  n_c = None
elif (dataset == 'shakespeare'):
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
"adjust_lr": args_required.adjust_lr,
"l2_reg": args_required.l2_reg,
"adjustLR_coef": args_required.adjustLR_coef,
"eps_elewd": args_required.eps_elewd,
"local_momentum": args_required.local_momentum,
"dataset": args_required.dataset,
"lm_coef": args_required.lm_coef,
"mc_coef": args_required.mc_coef,
"use_debug_max": args_required.use_debug_max,
"drop_out": args_required.drop_out,
"num_layer": args_required.num_layer
}

if args['use_wandb']:
    import wandb
    wandb.login(key='d945a40dead9fbb1b24ee2f35cd6a14babf01fa9')
    wandb.init(config=args_required, entity='ljb', name=filename, project='fedexp')


if dataset == 'personachat':
    from pytorch_transformers import GPT2DoubleHeadsModel, GPT2Tokenizer, CONFIG_NAME
    tokenizer_class = GPT2Tokenizer
    model_class = GPT2DoubleHeadsModel
    net_glob_org = model_class.from_pretrained('gpt2').to(args['device'])
    getattr(net_glob_org, 'module', net_glob_org).config.to_json_file(
            os.path.join('./log/pc', CONFIG_NAME)
        )
    tokenizer = tokenizer_class.from_pretrained('gpt2')
    tokenizer.save_pretrained('./log/pc')
    add_special_tokens_(net_glob_org, tokenizer)
    dataset_train = FedPERSONA('./data/personachat', tokenizer, num_candidates=args_required.num_candidates, \
                               max_history=args_required.max_history, personality_permutations=1, do_iid=False, train=True, download=True)
    dataset_test_global = FedPERSONA('./data/personachat', tokenizer, num_candidates=args_required.num_candidates, \
                                     max_history=args_required.max_history, personality_permutations=1, do_iid=False, train=False, download=True, \
                                      use_debug_max=args['use_debug_max']).dataset
elif dataset == 'shakespeare':
    data_obj = ShakespeareObjectCrop_noniid('./data/shakespeare/', 'shakepeare_nonIID')
    dataset_train = data_obj.clnt_data
    dataset_test_global = data_obj.tst_data

    net_glob_org = shake_transf(args).to(args['device'])
else:
    dataset_train, dataset_test_global = get_dataset(dataset, num_clients, n_c, alpha, True)
    if model == 'resnet20':
      from util_models_res20 import *
      net_glob_org = resnet20().to(args['device'])
    else:
      net_glob_org = get_model(model,n_c).to(args['device'])


d = parameters_to_vector(net_glob_org.parameters()).numel()


decay =  args_required.decay
# decay = 0.998
max_norm = args_required.max_norm
use_gradient_clipping = args_required.use_gradient_clipping
# weight_decay = 0.0001
weight_decay = args_required.weight_decay

# eta_l_algs, eta_g_algs, epsilon_algs, mu_algs = get_configs(dataset)
n = len(dataset_train)
print ("No. of clients", n)

# ratio of local dataset
p = np.zeros((n))
if args['dataset'] != 'personachat':
    for i in range(n):
        p[i] = len(dataset_train[i])
else:
    for i in range(n):
        p[i] = dataset_train.train_utterances_per_dialog[i]
p = p/np.sum(p)


# ============================== grid search best ratio ======================================
@torch.no_grad()
def get_best_ratio(w_curr, w_prev, net_eval_grid, grid=0.1, start=0.0, end=1.0):
    max_acc = -1
    max_ratio = None
    acc_ratio1 = None
    min_acc = 200
    min_ratio = None
    ratio = start
    while ratio <= end:
        vector_to_parameters((1-ratio)*w_prev + ratio*w_curr, net_eval_grid.parameters())
        test_acc_grid, test_loss_grid = test_img(net_eval_grid, dataset_test_global, args)
        if test_acc_grid > max_acc:
            max_acc = test_acc_grid
            max_ratio = ratio
        if test_acc_grid < min_acc:
            min_acc = test_acc_grid
            min_ratio = ratio
        ratio = grid + ratio

    vector_to_parameters(w_curr, net_eval_grid.parameters())
    acc_ratio1, _ = test_img(net_eval_grid, dataset_test_global, args)
    return max_ratio, min_ratio, min_acc, max_acc - acc_ratio1, max_acc - min_acc
# ==============================================================================================


net_glob_org = torch.nn.DataParallel(net_glob_org)
net_glol_org = torch.compile(net_glob_org)


algs = [algorithm]
for alg in algs:

    net_glob = copy.deepcopy(net_glob_org)
    net_glob.train()

    # w_glob = net_glob.state_dict()
    filename_model_alg = filename + "-" + alg + ".pt"
    dict_results[alg] = {}
    loss_t = []
    train_loss_algo_tmp = []
    train_acc_algo_tmp = []
    test_loss_algo_tmp = []
    test_acc_algo_tmp = []
    eta_g_tmp = []

    # ===================== grid ======================================
    ratio_best_tmp = []
    acc_gap_tmp = []
    acc_gap1_tmp = []
    # =================================================================
    
    w_vec_estimate = torch.zeros(d).to(args['device'])

    grad_mom = torch.zeros(d).to(args['device'])  # fedadam / fedeavgm
    mem_mat = None
    if dataset != 'personachat':
        mem_mat = torch.zeros((n, d)).to(args['device'])  # scaffold / fedavg (mom)
    delta = torch.zeros(d).to(args['device']) # fedadagrad / fedadam
    grad_norm_avg_running = 0 # fedavgm (exp)


    local_lr = args_required.eta_l
    global_lr = args_required.eta_g
    epsilon = args_required.epsilon
    mu = args_required.mu
    
    grad_square_avg = torch.ones(d).to(args['device'])
    for t in range(0,args['rounds']+1):        
        print ("Algo ", alg, " Round No. " , t)
        args_hyperparameters, local_lr, epsilon = get_hyper_round(decay, local_lr, epsilon, mu, weight_decay, 
                                                                  global_lr, use_gradient_clipping, max_norm, dataset, t)      
        
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

        all_ratios = []
        all_norms = []
        # local traininig
        # curr_time = time.time()
        for (ii, i) in enumerate(ind):
            # print('begin client: ', i)
            # print('time cost: ', time.time() - curr_time)
            # curr_time = time.time()
            grad = get_grad(copy.deepcopy(net_glob),args, args_hyperparameters, dataset_train[i], alg, i,  mem_mat, c, t, grad_square_avg)
            if args['adjust_lr']:
               grad, ratios = grad
               for ratio in ratios:
                  all_ratios.append(ratio)
            if use_gradient_clipping:
               grad, norms = grad
               for norm in norms:
                  all_norms.append(norm) 
            grad_norm_sum += p[i]*torch.linalg.norm(grad)**2
            grad_avg = grad_avg + p[i]*grad
            grad_square_avg_curr = grad_square_avg_curr + p[i]*torch.square(grad)
            p_sum += p[i]
            if ii % (len(ind)//5) == 0:
                print('finished clients ', ii)
        
        grad_square_avg = copy.deepcopy(grad_square_avg_curr)

        # momentum, parameter update 
        with torch.no_grad():
            grad_avg = grad_avg/p_sum
            grad_norm_avg = grad_norm_sum/p_sum
            
            if(alg=='fedavgm' or alg=='fedavgm(ep)'):
              grad_norm_avg_running = grad_norm_avg +0.9*0.5*grad_norm_avg_running # fedavgm
              grad_avg = grad_avg + 0.9*grad_mom
              grad_mom = grad_avg

            if(alg=='fedadagrad'):
              delta = delta + grad_avg**2
              # grad_avg = grad_avg/(torch.sqrt(delta+epsilon_fedadagrad))
              grad_avg = grad_avg/(torch.sqrt(delta+epsilon))

            if(alg=='fedadam'):
              grad_avg = 0.1*grad_avg + 0.9*grad_mom
              grad_mom = grad_avg
              delta = 0.01*grad_avg**2 + 0.99*delta
              grad_avg_normalized = grad_avg/(0.1)
              delta_normalized = delta/(0.01)
              # grad_avg = (grad_avg_normalized/torch.sqrt(delta_normalized + epsilon_fedadam))
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

            if args_required.use_log_etag:
                eta_g = math.log(t/300+1)+1
            eta_g_tmp.append(eta_g)
            # parameter update
            w_vec_prev = w_vec_estimate
            w_vec_estimate = parameters_to_vector(net_glob.parameters()) + eta_g*grad_avg
            vector_to_parameters(w_vec_estimate,net_glob.parameters())

            # eval 
            net_eval = copy.deepcopy(net_glob)

            # ==================== grid search best ratio ==========================================
            if args_required.use_grid:
                ratio_best, ratio_worst, min_acc, acc_gap1, acc_gap = get_best_ratio(copy.deepcopy(w_vec_estimate), copy.deepcopy(w_vec_prev), net_eval, grid=0.1, start=0.0, end=2.0)
                if (t>0):
                  w_vec_avg = (1-ratio_best) * w_vec_prev + ratio_best * w_vec_estimate
                else:
                  w_vec_avg = w_vec_estimate
            
            # ======================================================================================

            else:
                if(t>0):
                  w_vec_avg = (w_vec_estimate + w_vec_prev)/2
                else:
                  w_vec_avg = w_vec_estimate


            if(alg=='fedexp' or alg=='scaffold(exp)' or alg=='fedprox(exp)' or alg=='fedavgm(exp)' or args_required.use_grid or args_required.use_log_etag):
                if not args_required.no_model_avg:
                    vector_to_parameters(w_vec_avg, net_eval.parameters())  
            
            if(t%print_every_test==0):
                if t%print_every_train==0:
                    # average training loss/acc of the global model on clients
                    sum_loss_train = 0
                    sum_acc_train = 0
                    train_ind = ind if dataset =='personachat' else list(range(n))
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
          

        if args_required.use_grid:
            ratio_best_tmp.append(ratio_best)
            acc_gap_tmp.append(acc_gap)
            acc_gap1_tmp.append(acc_gap1)
            dict_results[alg][alg+"_ratio_best"] = ratio_best_tmp
            dict_results[alg][alg+"_acc_gap"] = acc_gap_tmp
            dict_results[alg][alg+"_acc_gap1"] = acc_gap1_tmp
            if args['use_wandb']:
                wandb.log({'train_loss': float(sum_loss_train),
                          'train_acc': float(sum_acc_train),
                          'test_loss': float(sum_loss_test),
                          'test_acc': float(sum_acc_test),
                          'global_lr': float(eta_g),
                          'best_ratio': float(ratio_best),
                          'best_lr': float(ratio_best * eta_g),
                          'worst_ratio': float(ratio_worst),
                          'min_acc': float(min_acc),
                          'acc_gap1': float(acc_gap1),
                          'acc_gap': float(acc_gap)})
        else:
            if args['use_wandb']:
                if len(all_ratios) != 0:
                    ratio_mean = sum(all_ratios) / len(all_ratios)
                else:
                    ratio_mean = 1.0
                if len(all_norms) != 0:
                    norm_mean = sum(all_norms) / len(all_norms)
                else:
                    norm_mean = max_norm
                wandb.log({'train_loss': float(sum_loss_train),
                          'train_acc': float(sum_acc_train),
                          'test_loss': float(sum_loss_test),
                          'test_acc': float(sum_acc_test),
                          'global_lr': float(eta_g),
                          'local_lr_mean': float(ratio_mean),
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
