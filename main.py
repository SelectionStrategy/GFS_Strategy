import copy
import numpy as np
import os

from utils.configs import FedConfig
from utils.sample import dirichlet_train, noniid_test
from roles.server import Server
from roles.client import Client

from torchvision import transforms, datasets

def main(cfg=None):
    
    if cfg is None:
        cfg = FedConfig()

    # Load dataset
    if cfg.dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.FashionMNIST(root='./data/fmnist', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data/fmnist', train=False, download=True, transform=transform)
    elif cfg.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
    else:
        exit('Error: unrecognized dataset...')

    print("Dataset loaded...")

    np.random.seed(cfg.seed)
    
    # Split dataset {user_id: [list of data index]}
    dict_users_train, ratio, client_data_num = dirichlet_train(train_set, cfg.num_clients, cfg.alpha, cfg.seed)
    dict_users_test = noniid_test(test_set, cfg.num_clients, ratio, cfg.test_data_size)

    print('Data finished...')

    # Server
    server = Server(cfg, client_data_num, test_set, dict_users_test)

    # Other parameters
    train_loss = [] # Per iteration
    gini = [] # Per iteration
    property_mean = [] # Per iteration
    acc_mean = [] # Per 10 iteration
    acc_variance = [] # Per 10 iteration
    frequency = np.zeros(cfg.num_clients)

    # Federated learning
    for t in range(cfg.iter):
        w_locals = [] # Selected clients' local models
        loss_locals = [] # Selected clients' local train loss

        # Client selection
        if cfg.selection_mode == 'random':
            client_idxs = server.random_selection()
        elif cfg.selection_mode == 'pow_d':
            client_idxs = server.pow_d_selection()
        elif cfg.selection_mode == 'gfs':
            if t <= cfg.cold_start:
                client_idxs = server.random_selection()
            else:
                client_idxs = server.greedy_selection(t + 1)
        elif cfg.selection_mode == 'poor':
            if t <= cfg.cold_start:
                client_idxs = server.random_selection()
            else:
                client_idxs = server.poor_selection(server.select_num)
        else:
            print('Selection mode Error...')
            exit(1)

        # Print selection
        print("Round {:3d}, Selected clients' id: ".format(t + 1) + str(client_idxs))

        # Local train
        for idx in client_idxs:
            client = Client(cfg, train_set, dict_users_train[idx], server.net)
            client.local_train()

            w_locals.append(client.net.state_dict())
            loss_locals.append(client.avg_loss.item())

            frequency[idx] += 1

            # Update client's gradient
            if cfg.selection_mode == 'gfs':
                server.update_gradient(idx, client.net.state_dict(), t + 1)
        
        # Aggregate
        server.fed_avg(w_locals)

        # Update property
        server.update_property(client_idxs)

        # Update weight
        if cfg.selection_mode == 'gfs':
            server.update_weight()
        
        # Calculate average local loss 
        loss_avg = np.mean(loss_locals)
        # loss_avg = sum(loss_locals) / len(loss_locals)

        # Record
        train_loss.append(loss_avg)
        property_mean.append(np.mean(server.property))
        gini.append(server.gini_coefficient(server.property))

        if (t + 1) % 5 == 0:
            acc = [server.test_global_model(i) for i in range(cfg.num_clients)]
            acc_mean.append(np.mean(acc))
            acc_variance.append(np.var(acc))
            print('Round {:3d}, Average loss {:.6f}'.format(t + 1, train_loss[-1]))
            print('Round {:3d}, Gini coefficient {:.6f}'.format(t + 1, gini[-1]))
            print('Round {:3d}, Property mean {:6f}'.format(t + 1, property_mean[-1]))
            print('Round {:3d}, Acc variance {:6f}'.format(t + 1, acc_variance[-1]))
            print('Round {:3d}, Acc mean {:6f}'.format(t + 1, acc_mean[-1]))

    print("Train finished...")

    # Record parameters
    para_dict = {}
    para_dict["train_loss"] = train_loss
    para_dict["acc_mean"] = acc_mean
    para_dict["acc_variance"] = acc_variance
    para_dict["gini"] = gini
    para_dict["property_mean"] = property_mean
    para_dict["property"] = server.property
    para_dict["frequency"] = frequency

    # Save model
    # server.save_model()

    save_str = str(cfg.alpha)+"_"+cfg.selection_mode+"_"+cfg.dataset+".npy"

    np.save(os.path.join(cfg.save_path, save_str), para_dict)

    print('Model saved...')
    
    return

if __name__ == '__main__':
    main()