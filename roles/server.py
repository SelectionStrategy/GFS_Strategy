import numpy as np
import os
from models.nets import CNNMnist, CNNCifar
from torch.utils.data import DataLoader, Dataset
from utils.datasets import DatasetSplit

import copy
import torch

class Server:
    def __init__(self, cfg, client_data_num, test_set, dict_users_test):
        if cfg.dataset == 'cifar':
            self.net = CNNCifar().to(cfg.device)
        else:
            self.net = CNNMnist().to(cfg.device)
            
        self.cfg = cfg
        self.client_data_num = np.array(client_data_num)
        self.property = np.ones(cfg.num_clients)
        self.select_num = int(cfg.frac * cfg.num_clients)
        self.test_set = test_set
        self.dict_users_test = dict_users_test
        self.time_record = np.zeros(cfg.num_clients)
        self.gradients = [None for _ in range(cfg.num_clients)]
        self.weight = np.zeros((cfg.num_clients, cfg.num_clients))
        self.max_data_num = np.max(client_data_num)

        print("Server created...")
    
    def fed_avg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))

        self.net.load_state_dict(w_avg)

    def update_gradient(self, idx, w_client, t):
        self.time_record[idx] = t
        w_server = self.net.state_dict()
        grad = []
        for k in w_server.keys():
            grad.append((w_client[k] - w_server[k]).view(-1))
        self.gradients[idx] = torch.cat(grad)
    
    def update_weight(self):
        for i in range(self.cfg.num_clients):
            for j in range(self.cfg.num_clients):
                if self.gradients[i] == None or self.gradients[j] == None:
                    self.weight[i][j] = 0.05
                else:
                    delta = abs(self.time_record[i] - self.time_record[j])
                    self.weight[i][j] = torch.cosine_similarity(self.gradients[i], self.gradients[j], dim=0) * pow(self.cfg.decay, delta)
            self.weight[i] = self.weight[i] + 0.05

    def calc_income(self):
        return np.array([self.test_global_model(i) for i in range(self.cfg.num_clients)]) * self.cfg.income_coefficient
        
    def calc_cost(self, S):
        return self.client_data_num * np.array(S) * self.cfg.cost_coefficient

    def update_property(self, client_idxs):
        S = self.list_to_set(client_idxs)
        # Update property
        self.property = self.property + self.calc_income() - self.calc_cost(S)

    def sigmoid(self, x):
        coefficient = -3 / (self.select_num * self.max_data_num)
        return 1 / (1 + np.exp(coefficient * x))
    
    def estimate_income_by_gradients(self, S):
        income = np.zeros(self.cfg.num_clients)
        for i in range(self.cfg.num_clients):
            for j in range(self.cfg.num_clients):
                income[i] += self.weight[i][j] * S[j] * self.client_data_num[j]
        
        # income = (income - np.min(income)) / (np.max(income) - np.min(income) + 1e-5) * self.cfg.income_coefficient
        income = self.sigmoid(income) * self.cfg.income_coefficient
        return income

    # Maximize this function
    def social_welfare_with_fairness(self, S, t):
        income = self.estimate_income_by_gradients(S)
        cost = self.calc_cost(S)
        prop = self.property + income - cost
        lamda = self.cfg.lamda / self.cfg.iter * t # increasing lambda
        return np.sum(income - cost) / self.cfg.num_clients - lamda * self.gini_coefficient(prop)
    
    def gini_coefficient(self, prop):
        up = 0.0
        for i in range(self.cfg.num_clients):
            for j in range(self.cfg.num_clients):
                up += abs(prop[i] - prop[j])
        return up / (2 * self.cfg.num_clients * self.cfg.num_clients * np.mean(prop))
        
    def random_selection(self):
        return np.random.choice(range(self.cfg.num_clients), self.select_num, replace=False)
    
    def pow_d_selection(self):
        all_idxs = np.random.choice(range(self.cfg.num_clients), self.select_num * 2, replace=False)
        acc = [self.test_global_model(i) for i in all_idxs]
        order = np.argsort(acc)
        idxs = [all_idxs[i] for i in order[:self.select_num]]
        return idxs
    
    def poor_selection(self, num):
        all_idxs = np.argsort(self.property)
        idxs = [all_idxs[i] for i in range(num)]
        return idxs
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))
    
    def list_to_set(self, idxs):
        S = np.zeros(self.cfg.num_clients)
        for k in idxs:
            S[k] = 1
        return S
    
    def set_to_list(self, S):
        idxs = []
        for k in range(self.cfg.num_clients):
            if S[k] == 1:
                idxs.append(k)
        return idxs
    
    def greedy_selection(self, t):
        clients = np.array([i for i in range(self.cfg.num_clients)])
        idxs = np.random.choice(clients, size=self.cfg.random_select_num, replace=False)
        clients = np.setdiff1d(clients, idxs)
        for _ in range(self.select_num - self.cfg.random_select_num):
            select = np.random.choice(clients, size=self.cfg.greedy_select_num, replace=False)
            idx = -1
            cur_max = -1e5
            S = self.list_to_set(idxs)
            for k in select:
                S[k] = 1
                cur = self.social_welfare_with_fairness(S, t)
                S[k] = 0
                if cur > cur_max:
                    cur_max = cur
                    idx = k

            clients = np.delete(clients, np.where(clients == idx))
            idxs = np.append(idxs, idx)
        
        return idxs
                
    def test_global_model(self, idx):
        dataset = DataLoader(DatasetSplit(self.test_set, self.dict_users_test[idx]), batch_size=self.cfg.test_batch_size, shuffle=True)

        self.net.eval()

        correct = 0
        total = 0
        for i, (images, labels) in enumerate(dataset):
            images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
            outputs = self.net(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        acc = correct / total
        return acc

    # def save_model(self):
    #     torch.save(self.net.state_dict(), os.path.join(self.cfg.save_path, self.cfg.selection_mode+"_"+self.cfg.dataset+".pth"))
    #     print('Global model saved...')
