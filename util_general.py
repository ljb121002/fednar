from util_libs import *
import wandb

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.numpy(), test_loss

def test_all(net_g, dataset, args, is_train_set=False):
    return test_img(net_g, dataset, args)



class LocalUpdate(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        norms = []
        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              log_probs = net(images)
              loss = self.loss_func(log_probs, labels)

              local_par_list = parameters_to_vector(net.parameters())
              loss += 0.5 * self.args['l2_reg'] * torch.norm(local_par_list) ** 2
              loss.backward()
                
              total_norm = torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
              if total_norm >= self.max_norm:
                  norms.append(total_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
        
        return model_to_return, norms

class LocalUpdate_scaffold(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
    
        
    def train_and_sketch(self, net, idx, mem_mat, c):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum = 0, weight_decay = self.weight_decay)


        prev_net = copy.deepcopy(net)
        
        eta = self.lr

        batch_loss = []
        step_count = 0

        norms = []
        while(True):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.use_data_augmentation == True):
                  images = self.transform_train(images)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)


                state_params_diff = c-mem_mat[idx]
                local_par_list = parameters_to_vector(net.parameters())
                
                loss_algo = torch.sum(local_par_list * state_params_diff)
                loss = loss + loss_algo

                loss += 0.5 * self.args['l2_reg'] * torch.norm(local_par_list) ** 2

                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
                if total_norm >= self.max_norm:
                    norms.append(total_norm)

                optimizer.step()
                batch_loss.append(loss.item())
                step_count=step_count+1

            
            
                if(step_count >= self.args['cp']):
                    break

            if(step_count >= self.args['cp']):
              break

        with torch.no_grad():
                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                mem_mat[idx] = (mem_mat[idx]-c) - params_delta_vec/(step_count*eta)
                model_to_return = params_delta_vec
            
        return model_to_return, norms

class LocalUpdate_fedprox(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.mu = args_hyperparameters['mu']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum = 0, weight_decay = self.weight_decay)

        prev_net = copy.deepcopy(net)
        prev_net_vec = parameters_to_vector(prev_net.parameters())
        
        eta = self.lr
        mu = self.mu

        batch_loss = []
        step_count = 0

        norms = []
        while(True):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.use_data_augmentation == True):
                  images = self.transform_train(images)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                local_par_list = parameters_to_vector(net.parameters())
                loss_algo = torch.linalg.norm(local_par_list-prev_net_vec)**2
                loss = loss + mu*0.5*loss_algo

                loss += 0.5 * self.args['l2_reg'] * torch.norm(local_par_list) ** 2

                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
                if total_norm >= self.max_norm:
                    norms.append(total_norm)

                optimizer.step()
                batch_loss.append(loss.item())
                step_count=step_count+1
                
                if(step_count >= self.args['cp']):
                    break
                    
            if(step_count >= self.args['cp']):
                break

        with torch.no_grad():
                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return, norms



def get_grad(net_glob, args, args_hyperparameters,  dataset, alg, idx, mem_mat, c, t, grad_square_avg=None):

    if(alg=='fedadam' or alg == 'fedexp' or alg =='fedavg' or alg=='fedavgm' or alg=='fedavgm(exp)' or alg=='fedadam'):
        local = LocalUpdate(args, args_hyperparameters, dataset=dataset)
        grad = local.train_and_sketch(copy.deepcopy(net_glob))
        return grad

    elif(alg=='scaffold' or alg=='scaffold(exp)'):
         local = LocalUpdate_scaffold(args, args_hyperparameters, dataset=dataset)
         grad = local.train_and_sketch(copy.deepcopy(net_glob),idx,mem_mat,c)
         return grad
    
    elif(alg=='fedprox' or alg=='fedprox(exp)'):            
         local = LocalUpdate_fedprox(args, args_hyperparameters, dataset=dataset)
         grad = local.train_and_sketch(copy.deepcopy(net_glob))
         return grad
