from util_libs import *

import wandb


def _check_shape(y_pred, y):
    if y.ndimension() > 1 and y.shape[1] == 1:
        # (N, 1, ...) -> (N, ...)
        y = y.squeeze(dim=1)
    if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
        # (N, 1, ...) -> (N, ...)
        y_pred = y_pred.squeeze(dim=1)
    if not (y.ndimension() == y_pred.ndimension()
            or y.ndimension() + 1 == y_pred.ndimension()):
        raise ValueError("y must have shape of (batch_size, ...) and "
                         "y_pred must have shape of (batch_size, "
                         "num_categories, ...) or (batch_size, ...), but "
                         "given {} vs {}.".format(y.shape, y_pred.shape))
    y_shape = y.shape
    y_pred_shape = y_pred.shape
    if y.ndimension() + 1 == y_pred.ndimension():
        y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]
    if not (y_shape == y_pred_shape):
        raise ValueError("y and y_pred must have compatible shapes.")
    return y_pred, y

def inference(model, batch, args):
    model.eval()
    with torch.no_grad():
        (input_ids, mc_token_ids, lm_labels,
                mc_labels, token_type_ids) = batch
        lm_logits, mc_logits, *_ = model(input_ids,
                                         token_type_ids=token_type_ids,
                                         mc_token_ids=mc_token_ids)
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(
                -1, lm_logits.size(-1)
            )
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        return ((lm_logits_flat_shifted, mc_logits),
                (lm_labels_flat_shifted, mc_labels))

def accuracy(y_pred, y):
    y_pred, y = _check_shape(y_pred, y)
    indices = torch.argmax(y_pred, dim=1)
    correct = torch.eq(indices, y).view(-1)
    return torch.sum(correct).float() / correct.shape[0]

nll_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
def compute_loss_val(model, batch, args):
    (input_ids, mc_token_ids, lm_labels,
            mc_labels, token_type_ids) = batch

    logits, labels = inference(model, batch, args)
    lm_logits, mc_logits = logits
    lm_labels, mc_labels = labels
    nll = nll_criterion(lm_logits, lm_labels)
    acc = accuracy(mc_logits, mc_labels)
    return nll, acc

import time
def compute_loss_train(model, batch, args):
    (input_ids, mc_token_ids, lm_labels,
            mc_labels, token_type_ids) = batch

    lm_loss, mc_loss, *_ = model(
        input_ids, token_type_ids=token_type_ids,
        mc_token_ids=mc_token_ids,
        mc_labels=mc_labels, lm_labels=lm_labels
    )

    loss = (lm_loss * args['lm_coef'] + mc_loss * args['mc_coef'] + 0.5 * args['l2_reg'] * torch.norm(parameters_to_vector(model.parameters())) ** 2)
    # there are no metrics, but still need to return a tuple
    return loss,


# pc: personachat
def pc_test(net_g, datatest, args, is_train_set):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    total = 0
    # data_loader = DataLoader(datatest, batch_size=args['bs'])
    # l = len(data_loader)
    # for idx, (data, target) in enumerate(data_loader):
    if not is_train_set:
        for idx, data in enumerate(datatest):
            # data = data.to(args['device'])
            data = [data_i.to(args['device']) for data_i in data]
            # log_probs = net_g(data)
            # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # y_pred = log_probs.data.max(1, keepdim=True)[1]
            # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            curr_loss, curr_acc = compute_loss_val(net_g, data, args)
            test_loss += curr_loss * len(data[0])
            total += len(data[0])
            correct += curr_acc * len(data[0])
    else:
        datatest = [data_i.to(args['device']) for data_i in datatest]
        curr_loss, curr_acc = compute_loss_val(net_g, datatest, args)
        test_loss += curr_loss * len(datatest[0])
        total += len(datatest[0])
        correct += curr_acc * len(datatest[0])

    test_loss /= total
    accuracy = 100.00 * correct / total
    return accuracy.item(), test_loss



class pc_LocalUpdate(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        # self.loss_func = nn.CrossEntropyLoss()
        self.dataset = [i.to(self.args['device']) for i in dataset]
        # self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        # self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum = self.args['local_momentum'], weight_decay=self.weight_decay)
        # optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        norms = []
        while(True):
            net.zero_grad()
            loss, = compute_loss_train(net, self.dataset, self.args)
            loss.backward()
                
            if(self.use_gradient_clipping ==True):
                total_norm = torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
                if total_norm >= self.max_norm:
                    norms.append(total_norm)
            
            optimizer.step()
            batch_loss.append(loss.item())
            step_count=step_count+1
            if(step_count >= self.args['cp']):
                break

        with torch.no_grad():
                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
        
        if self.use_gradient_clipping:
            return model_to_return, norms
        else:
            return model_to_return

class pc_LocalUpdate_scaffold(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
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

                if(self.use_gradient_clipping ==True):
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
            
        if self.use_gradient_clipping:
            return model_to_return, norms
        else:
            return model_to_return

class pc_LocalUpdate_fedprox(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = nn.MSELoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
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

                if(self.use_gradient_clipping==True):
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
            
        if self.use_gradient_clipping:
            return model_to_return, norms
        else:
            return model_to_return





def pc_get_grad(net_glob, args, args_hyperparameters,  dataset, alg, idx, mem_mat, c, t, grad_square_avg=None):

    if(alg=='fedadam' or alg == 'fedexp' or alg =='fedavg' or alg=='fedavgm' or alg=='fedavgm(exp)' or alg=='fedadam' or alg=='fedadagrad'):
        local = pc_LocalUpdate(args, args_hyperparameters, dataset=dataset)
        grad = local.train_and_sketch(copy.deepcopy(net_glob))
        return grad

    elif(alg=='scaffold' or alg=='scaffold(exp)'):
         local = pc_LocalUpdate_scaffold(args, args_hyperparameters, dataset=dataset)
         grad = local.train_and_sketch(copy.deepcopy(net_glob),idx,mem_mat,c)
         return grad
    
    elif(alg=='fedprox' or alg=='fedprox(exp)'):            
         local = pc_LocalUpdate_fedprox(args, args_hyperparameters, dataset=dataset)
         grad = local.train_and_sketch(copy.deepcopy(net_glob))
         return grad


def adjust_lr(optimizer, net, vec_prev, curr_lr, adjustLR_coef):
    local_grad_list = None
    for param in net.parameters():
        if not isinstance(local_grad_list, torch.Tensor):
            local_grad_list = param.grad.reshape(-1)
        else:
            local_grad_list = torch.cat((local_grad_list, param.grad.reshape(-1)), 0)
    similarity = torch.sum(local_grad_list * vec_prev) / (torch.linalg.norm(local_grad_list) * torch.linalg.norm(vec_prev) + 1e-6)
    ratio = np.exp(adjustLR_coef * similarity.item())
    # lr_new = curr_lr * max(ratio.item(), 1)
    for group in optimizer.param_groups:
        # group['lr'] = lr_new
        group['lr'] = group['lr'] * ratio
        ratio_accum = group['lr'] / curr_lr
    return ratio, ratio_accum



ATTR_TO_SPECIAL_TOKEN = {
                         'bos_token': '<bos>',
                         'eos_token': '<eos>',
                         'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>',
                                                       '<speaker2>')
                        }


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model

    if they have not already been added.
    """
    orig_num_tokens = len(tokenizer.encoder)
    # returns 0 and doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(
                new_num_tokens=orig_num_tokens + num_added_tokens
            )