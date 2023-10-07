import numpy as np
from fedlab.utils.dataset.partition import CIFAR10Partitioner

def dirichlet_train(dataset, num_users, alpha, seed):
    partition = CIFAR10Partitioner(targets=dataset.targets, 
                                   num_clients=num_users, 
                                   balance=None, 
                                   partition='dirichlet', 
                                   dir_alpha=alpha, 
                                   seed=seed)
    dict_users_train = partition.client_dict
    client_data_num = np.array(partition.client_sample_count).reshape(-1)

    ratio = np.zeros((num_users, 10))
    labels = np.array(dataset.targets)

    for i in range(num_users):
        count = np.zeros(10)
        for j in dict_users_train[i]:
            count[labels[j]] += 1
        ratio[i] = count / client_data_num[i]

    return dict_users_train, ratio, client_data_num

def noniid_test(dataset, num_users, ratio, test_data_size):
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}

    labels = np.array(dataset.targets)

    # {label -> [idx]}
    bucket = [[] for _ in range(10)]
    for i in range(len(labels)):
        bucket[labels[i]].append(i)

    for i in range(num_users):
        total = test_data_size
        for j in range(9):
            num = int(ratio[i][j] * test_data_size)
            rand_idxs = np.random.choice(bucket[j], num, replace=False)
            dict_users_test[i] = np.concatenate((dict_users_test[i], rand_idxs), axis=0)
            total -= num
        rand_idxs = np.random.choice(bucket[9], total, replace=False)
        dict_users_test[i] = np.concatenate((dict_users_test[i], rand_idxs), axis=0)

    return dict_users_test
