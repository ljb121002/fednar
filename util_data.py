from util_libs import *


def getDirichletData(y, n, alpha, num_c):
        n_nets = n
        K = num_c

        labelList_true = y


        min_size = 0
        N = len(labelList_true)
        rnd = 0

        net_dataidx_map = {}

        p_client = np.zeros((n,K))

        for i in range(n):
          p_client[i] = np.random.dirichlet(np.repeat(alpha,K))

    
        
        idx_batch = [[] for _ in range(n)]
        
        m = int(N/n)
        
        for k in range(K):
            idx_k = np.where(labelList_true == k)[0]

            np.random.shuffle(idx_k)

            proportions = p_client[:,k]

            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList_true[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

        return idx_batch
    
def getDirichletData_equal(y, n, alpha, num_c):
        n_nets = n
        K = num_c

        labelList_true = y


        min_size = 0
        N = len(labelList_true)
        rnd = 0

        net_dataidx_map = {}

        p_client = np.zeros((n,K))

        for i in range(n):
          p_client[i] = np.random.dirichlet(np.repeat(alpha,K))
            
        p_client_cdf = np.cumsum(p_client, axis=1)
      
        
        idx_batch = [[] for _ in range(n)]
        
        m = int(N/n)
        
        
        idx_labels = [np.where(labelList_true==k)[0] for k in range(K)]

        
        idx_counter = [0 for k in range(K)]
        total_cnt = 0
        
        
        while(total_cnt<m*n):
                
            curr_clnt = np.random.randint(n)
            
            if (len(idx_batch[curr_clnt])>=m):
                continue

            
            total_cnt += 1
            curr_prior = p_client_cdf[curr_clnt]
                
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if (idx_counter[cls_label] >= len(idx_labels[cls_label])):
                    continue

                idx_batch[curr_clnt].append(idx_labels[cls_label][idx_counter[cls_label]])
                idx_counter[cls_label] += 1

                break

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList_true[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

        return idx_batch
    
    

def get_dataset(datatype, n_client, n_c, alpha, partition_equal=True):

    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trans_fashionmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


    if(datatype=='CIFAR10' or datatype=='CIFAR100' or datatype=='MNIST' or datatype =='FashionMNIST' or datatype == 'CINIC10'):
    
        if(datatype=='CIFAR10'):

            dataset_train_global = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)
            dataset_test_global = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)
                
        if(datatype=='CINIC10'):
            cinic_mean = [0.47889522, 0.47227842, 0.43047404]
            cinic_std = [0.24205776, 0.23828046, 0.25874835]
            dataset_train_global = datasets.ImageFolder('./data/cinic10/train',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
            dataset_test_global = datasets.ImageFolder('./data/cinic10/test',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))

        elif(datatype=='CIFAR100'):

            dataset_train_global = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar)
            dataset_test_global = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar)

        elif(datatype=='MNIST'):

            dataset_train_global = datasets.MNIST('./data/mnist', train=True, download=True, transform=trans_mnist)
            dataset_test_global = datasets.MNIST('./data/mnist', train=False, download=True, transform=trans_mnist)

        elif(datatype=='FashionMNIST'):


            dataset_train_global = datasets.FashionMNIST('./data/fashionmnist', train=True, download=True, transform=trans_fashionmnist)
            dataset_test_global = datasets.FashionMNIST('./data/fashionmnist', train=False, download=True, transform=trans_fashionmnist)


        

        train_loader = DataLoader(dataset_train_global, batch_size=len(dataset_train_global))
        test_loader  = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))

        X_train = next(iter(train_loader))[0].numpy()
        Y_train = next(iter(train_loader))[1].numpy()

        X_test = next(iter(test_loader))[0].numpy()
        Y_test = next(iter(test_loader))[1].numpy()


        if(partition_equal == True):
            inds = getDirichletData_equal(Y_train, n_client, alpha, n_c)
        else:
            inds = getDirichletData(Y_train, n_client, alpha, n_c)


        dataset_train=[]
        dataset_test = []

        len_test = int(len(X_test)/n_client)


        for (i,ind) in enumerate(inds):


            ind = inds[i]
            
            x = X_train[ind]
            y = Y_train[ind]
                

            x_test = X_test[i*len_test:(i+1)*len_test]
            y_test = Y_test[i*len_test:(i+1)*len_test]
            

            n_i = len(ind)

            x_train = torch.Tensor(x[0:n_i])
            y_train = torch.LongTensor(y[0:n_i])

            x_test = torch.Tensor(x_test)
            y_test = torch.LongTensor(y_test)


            print ("Client ", i, " Training examples-" , len(x_train), " Test examples-", len(x_test))

            dataset_train_torch = TensorDataset(x_train,y_train)
            dataset_test_torch = TensorDataset(x_test,y_test)

            dataset_train.append(dataset_train_torch)
            dataset_test.append(dataset_test_torch)
    

    if(datatype=='EMNIST'):

      
      with ZipFile('emnist_dataset_umifa.npy.zip', 'r') as f:
        f.extractall()


      emnist_data = np.load('emnist_dataset_umifa.npy', allow_pickle= True).item()
      dataset_train_emnist = emnist_data['dataset_train']
      dataset_test_emnist = emnist_data['dataset_test']
      dict_users_emnist = emnist_data['dict_users']
      emnist_clients = list(dict_users_emnist.keys())

      x_train = dataset_train_emnist[:][0]
      y_train = dataset_train_emnist[:][1]
      x_train = x_train[:,None]

      dataset_train_emnist_new = TensorDataset(x_train,y_train)

      x_test = dataset_test_emnist[:][0]
      y_test = dataset_test_emnist[:][1]
      x_test = x_test[:,None]

      dataset_test_emnist_new = TensorDataset(x_test,y_test)

      dataset_test_global = dataset_test_emnist_new

      dataset_train=[]
      dataset_test = []

      n = len(emnist_clients)

      len_test = int(len(dataset_test_emnist)/n)


      ctr = 0

      for i in range(n):
          
        ind = dict_users_emnist[i]

        x_train = dataset_train_emnist_new[ind][0]
        y_train = dataset_train_emnist_new[ind][1]

        x_test = dataset_test_emnist_new[i*len_test:(i+1)*len_test][0]
        y_test = dataset_test_emnist_new[i*len_test:(i+1)*len_test][1]

        

        n_i = len(ind)

        dataset_train_torch = TensorDataset(x_train,y_train)
        dataset_test_torch = TensorDataset(x_test,y_test)


        dataset_train.append(dataset_train_torch)
        dataset_test.append(dataset_test_torch)


    return dataset_train, dataset_test_global
    



from LEAF.utils_eval.model_utils import *
from LEAF.utils_eval.language_utils import *
        
class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=100, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.        
        # Change structure to DatasetObject structure
        
        self.users = users 

        tst_data_count_per_clnt = (crop_amount//tst_ratio)
        # Group clients that have at least crop_amount datapoints
        arr = []
        for clnt in range(len(users)):
            if (len(np.asarray(train_data[users[clnt]]['y'])) > crop_amount 
                and len(np.asarray(test_data[users[clnt]]['y'])) > tst_data_count_per_clnt):
                arr.append(clnt)

        # choose n_client clients randomly
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]
          
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        
        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[idx]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[idx]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[idx]]['y'])[start:start+crop_amount]
            
        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))
        
        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[idx]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['y'])[start:start+curr_amount]
            tst_data_count += curr_amount
            
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)
        
        # Convert characters to numbers
        
        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)
        
        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)
        
        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))
        

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            
            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)
                
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.clnt_data = [TensorDataset(torch.from_numpy(self.clnt_x[i]), torch.from_numpy(self.clnt_y[i]).view(-1)) for i in range(len(self.clnt_x))]
        
        
        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))
                
        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)
        
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

        self.tst_data = TensorDataset(torch.from_numpy(self.tst_x), torch.from_numpy(self.tst_y).view(-1))
    




from itertools import chain
from pytorch_transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

class Logger:
    def debug(self, msg, args=None):
        print(msg.format(args))
    def info(self, msg, args=None):
        print(msg.format(args))
    def warn(self, msg, args=None):
        print(msg.format(args))
    def error(self, msg, args=None):
        print(msg.format(args))
    def critical(self, msg, args=None):
        print(msg.format(args))

logger = Logger()

class FedPERSONA:
    def __init__(self, dataset_dir,  tokenizer, num_candidates, max_history, personality_permutations, \
                 do_iid=False, num_clients=None, train=True, download=False, use_debug_max=False, *args, **kwargs):
        self.dataset_dir = dataset_dir
        self.do_iid = do_iid
        self._num_clients = num_clients
        self.type = "train" if train else "val"
        if not do_iid and num_clients == 1:
            raise ValueError("can't have 1 client when non-iid")
        if not os.path.exists(self.stats_fn()):
            self.prepare_datasets(download=download)

        self._load_meta(train)

        if self.do_iid:
            self.iid_shuffle = np.random.permutation(len(self))

        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_history = max_history
        self.personality_permutations = personality_permutations

        # keep the entire val set in memory, since why not
        if self.type == "val":
            with open(self.validation_fn(), "r") as val_f:
                self.raw_val_set = json.load(val_f)
            if use_debug_max == True:
                self.raw_val_set = self.raw_val_set[:10]
        if self.type == 'train':
            self.raw_train_set = []
            for client_id in range(len(self.dialogs_per_client)):
                with open(self.client_fn(client_id), 'r') as train_f:
                    self.raw_train_set.append(json.load(train_f))

        self.dataset = self.get_datasets(train)

    def client_fn(self, client_id):
        fn = "client{}.json".format(client_id)
        return os.path.join(self.dataset_dir, fn)

    def validation_fn(self):
        return os.path.join(self.dataset_dir, "validation.json")
    
    def download_dataset(self, dataset_path):
        # download personachat to dataset_path
        msg = "Downloading personachat from S3 into {}"
        logger.info(msg.format(dataset_path))
        return cached_path(PERSONACHAT_URL, dataset_path)

    def prepare_datasets(self, download=True):
        # download the dataset
        os.makedirs(self.dataset_dir, exist_ok=True)
        dataset_path = self.download_dataset(self.dataset_dir)

        # split into client datasets and one validation set
        datasets, stats = self.split_dataset(dataset_path)
        client_datasets, validation_set = datasets
        dialogs_per_client, train_utterances_per_dialog, \
                val_utterances_per_dialog = stats

# =================== no check exist ===============================================
        # save client datasets to disk
        for client_id, personality in enumerate(client_datasets):
            fn = self.client_fn(client_id)
            if os.path.exists(fn):
                # raise RuntimeError("won't overwrite existing split")
                pass
            else:
                with open(fn, "w") as f:
                    json.dump(client_datasets[tuple(personality)], f)

        # save validation set to disk
        fn = self.validation_fn()
        if os.path.exists(fn):
            pass
            # raise RuntimeError("won't overwrite existing val set")
        else:
            with open(fn, "w") as f:
                json.dump(validation_set, f)
# =================== no check exist ===============================================

        # save stats to disk
        fn = self.stats_fn()
        if os.path.exists(fn):
            raise RuntimeError("won't overwrite existing stats file")
        stats = {"dialogs_per_client": dialogs_per_client,
                 "train_utterances_per_dialog":train_utterances_per_dialog,
                 "val_utterances_per_dialog": val_utterances_per_dialog}
        with open(fn, "w") as f:
            json.dump(stats, f)

    def split_dataset(self, dataset_path):
        """ Produces one file per client, and one with global stats

        Reads in a JSON file at `dataset_path`, partitions the data
        into clients based on the personality, and writes the data
        for each client to a separate JSON file. Also writes global
        stats needed for fast indexing to self.stats_fn()
        """
        raw_dataset = None
        with open(dataset_path, "r") as dataset_file:
            raw_dataset = json.load(dataset_file)

        val_set = raw_dataset["valid"]
        val_utterances_per_dialog = [len(dialog["utterances"])
                                     for dialog in val_set]

        client_datasets = defaultdict(list)
        for dialog in raw_dataset["train"]:
            personality = dialog["personality"]
            client_datasets[tuple(personality)].append(dialog)

        # so that we can quickly turn an utterance index into
        # a client idx, figure out how many dialogs per client and
        # how many utterances per dialog

        # fix the order of the clients
        client_personalities = list(client_datasets.keys())
        dialogs_per_client = []
        train_utterances_per_dialog = []
        for p in client_personalities:
            dialogs = client_datasets[p]
            dialogs_per_client.append(len(dialogs))
            train_utterances_per_dialog.extend([len(dialog["utterances"])
                                                for dialog in dialogs])

        datasets = (client_datasets, val_set)
        stats = (dialogs_per_client,
                 train_utterances_per_dialog,
                 val_utterances_per_dialog)
        return datasets, stats

    def _load_meta(self, train):
        with open(self.stats_fn(), "r") as f:
            stats = json.load(f)
            self.dialogs_per_client = stats["dialogs_per_client"]
            self.train_utterances_per_dialog = \
                    stats["train_utterances_per_dialog"]
            self.val_utterances_per_dialog = \
                    stats["val_utterances_per_dialog"]
        
    def stats_fn(self):
        return os.path.join(self.dataset_dir, "stats.json")

    def get_datasets(self, train):
        if train == False:
            # return a list, each item is all data on a client, and the data is padded and is ready to be sent to the model
            val_set = []
            for i in range(len(self.raw_val_set)):
                client_input = self.get_client_input(self.raw_val_set[i], train=False)
                val_set.append(personachat_collate_fn(client_input))
            return val_set
        elif train == True:
            # train_set = []
            # for i in range(len(self.raw_train_set)):
            #     client_input = self.get_client_input(self.raw_train_set[i], train=True)
            #     train_set.append(personachat_collate_fn(client_input))
            #     if i % 100 == 0:
            #         print('finish: ', i)
            # return train_set
            return None

    def __getitem__(self, idx):
        if self.type == 'val':
            return self.dataset[idx]
        else:
            client_input = self.get_client_input(self.raw_train_set[idx][0], train=True)
            return personachat_collate_fn(client_input)

    def __len__(self):
        if self.type == 'val':
            return len(self.dataset)
        else:
            return len(self.raw_train_set)

    
    def get_client_input(self, dialog, train):
        personality = dialog["personality"]
        utterance = dialog["utterances"]
        model_inputs = []
        if train == True:
            for i in range(len(utterance)):
                for _ in range(self.personality_permutations): 
                    random.shuffle(personality)
                    model_input = self.utterance_to_input(personality, utterance[i])
                    model_inputs.append(model_input)
        else:
            for i in range(len(utterance)): # 7 or 8
                model_input = self.utterance_to_input(personality, utterance[i])
                model_inputs.append(model_input)
        return model_inputs 

    
    def utterance_to_input(self, personality, utterance):
        history = utterance["history"]
        candidates = utterance["candidates"]

        num_candidates = len(candidates)
        # restrict to self.num_candidates if we're training
        # if self.num_candidates > 0 and self.type == "train":
        if self.num_candidates > 0:
            num_candidates = min(self.num_candidates, num_candidates)

        candidates = utterance["candidates"][-num_candidates:]
        history = utterance["history"][-(2 * self.max_history + 1):]

        return raw_to_input(self.tokenizer, personality,
                            history, candidates)



def tokenize(obj, tokenizer):
    """ Recursively tokenize all strings in obj """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(obj)
            )
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels",
                "mc_labels", "token_type_ids"]
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]

def build_input_from_segments(persona, history, reply, tokenizer,
                              lm_labels=False, with_eos=True):
    """ Build a sequence of input

    Builds from 3 segments: persona, history and last reply.
    """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS[:-1]
        )

    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history
    sequence += [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2
                                 if (len(sequence) - i) % 2 == 0
                                 else speaker1]
                                + s
                                for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1
                                  for i, s in enumerate(sequence)
                                  for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = [-1] * sum(len(s) for s in sequence[:-1])
        instance["lm_labels"] += [-1] + sequence[-1][1:]
    return instance




def raw_to_input(tokenizer, personality, history, candidates):
    """ Converts from dict of strings to (almost) valid input for the model

    "Almost" since we still need the collate_fn to pad & combine
    the tensors for each candidate
    """
    personality = tokenize(personality, tokenizer)
    history = tokenize(history, tokenizer)
    candidates = tokenize(candidates, tokenizer)

    model_input = defaultdict(list)
    num_candidates = len(candidates)
    # several of the model's inputs are num_candidates x sequence_len,
    # so process each candidate and append the result to model_input[...]
    for j, candidate in enumerate(candidates):
        lm_labels = bool(j == num_candidates - 1)
        instance = build_input_from_segments(personality, history,
                                             candidate, tokenizer,
                                             lm_labels)
        for input_name, input_array in instance.items():
            model_input[input_name].append(input_array)

    # the last candidate is always the correct choice
    model_input["mc_labels"] = num_candidates - 1
    # model_input["mc_labels"] = torch.zeros(num_candidates)
    # model_input["mc_labels"][num_candidates-1] = 1

    for input_name in MODEL_INPUTS:
        # tensorize
        if input_name != "mc_labels":
            tensors = [torch.tensor(l) for l in model_input[input_name]]
            model_input[input_name] = tensors

    # convert from dict to tuple in the correct order
    model_input = tuple(model_input[name] for name in MODEL_INPUTS)

    return model_input


from torch.nn.utils.rnn import pad_sequence
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
def personachat_collate_fn(records):
    # records is a list of tuples, where each tuple contains columns
    # (client_id,) + MODEL_INPUTS

    # need to return a batch, which is a tuple of tensors,
    # appropriately padded

    batch = []
    # input_ids has one sequence for each candidate, and all other
    # sequence model inputs have the same lengths, so we can just use
    # the max sequence length in input_ids
    # max_l = max(len(input_ids)
    #              for record in records
    #              for input_ids in record[1])
    for i, name in enumerate(MODEL_INPUTS):
        if name in PADDED_INPUTS:
            pad_val = 0 if name != "lm_labels" else -1
            sequences = [s for record in records for s in record[i]]
            padded = pad_sequence(sequences,
                                  batch_first=True,
                                  padding_value=pad_val)
            # padded has shape len(sequences) x max_l, where
            # len(sequences) = num_candidates * len(records)

            # we want batch_size x num_candidates x seq_len
            # where batch_size = len(records)
            reshaped = padded.view(len(records), len(records[0][1]), -1)
            batch.append(reshaped)
        else:
            batch.append(torch.stack([torch.tensor(record[i])
                                      for record in records]))

    return tuple(batch)
