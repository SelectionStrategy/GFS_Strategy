import os

class FedConfig:
    def __init__(self, cfg=None):

        # path
        self.root_path = os.path.dirname(os.path.dirname(__file__))
        self.save_path = os.path.join(self.root_path, "save")

        # general settings
        self.device = "cuda:0"
        # self.dataset = "cifar"
        self.dataset = "fmnist"
        self.test_data_size = 200
        self.seed = 2023 # random seed

        # server configs
        self.iter = 400 # communication round
        self.num_clients = 100
        self.frac = 0.1 # proportion
        # self.selection_mode = 'random'
        # self.selection_mode = 'pow_d'
        # self.selection_mode = 'poor'
        self.selection_mode = 'gfs'

        # gfs settings
        self.income_coefficient = 1 # income weight
        self.cost_coefficient = 0.00001 # cost weight
        self.lamda = 30 # fairness coefficient
        # self.lamda = 60 # cifar 0.4 fairness coefficient
        self.random_select_num = 2 # fmnist
        self.greedy_select_num = 40 # fmnist
        self.cold_start = 30 # fmnist
        # self.random_select_num = 4 # cifar
        # self.greedy_select_num = 30 # cifar
        # self.cold_start = 50 # cifar
        self.decay = 0.9 # decay coefficient
        
        # data partition
        self.alpha = 0.2 # Dirichlet parameter

        # client configs
        self.local_batch_size = 50
        self.local_epoch = 5
        self.local_lr = 0.01
        self.local_momentum = 0.5
        self.weight_decay = 1e-3

        # general test
        self.test_batch_size = 32

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)

    def __str__(self):
        string = ""
        for (k, v) in self.__dict__.items():
            string += "{}:{}\n".format(k,v)

        return string