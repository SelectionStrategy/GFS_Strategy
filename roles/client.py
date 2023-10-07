import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.datasets import DatasetSplit

import torch.nn as nn
import torch

class Client:
    def __init__(self, cfg, dataset, idxs, net):
        self.cfg = cfg
        self.dataset = DataLoader(DatasetSplit(dataset, idxs), batch_size=cfg.local_batch_size, shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        self.net = copy.deepcopy(net).to(cfg.device)
        self.avg_loss = 0

    def local_train(self):
        self.net.train()

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.cfg.local_lr, momentum=self.cfg.local_momentum, weight_decay=self.cfg.weight_decay)

        train_loss = []

        for iter in range(self.cfg.local_epoch):
            for batch_idx, (images, labels) in enumerate(self.dataset):
                images, labels = images.to(self.cfg.device), labels.to(self.cfg.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(loss)
        
        self.avg_loss = sum(train_loss) / len(train_loss)
