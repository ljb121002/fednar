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
    
    

def get_dataset(n_client, n_c, alpha, partition_equal=True):

    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])


    dataset_train_global = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)
    dataset_test_global = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)

    

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
    
