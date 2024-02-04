import os
import numpy  as np
import argparse
from scipy.stats import pearsonr
import warnings
import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from config import config
from model_pytorch import dnn_4l, dnn_3l, dnn_2l
from model_pytorch import msenanloss, ukbb_multi_task_dataset
from mics import mics_z_norm, mics_infer_metric, mics_log, print_result
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

def main(args):
    '''main function for meta-matching (DNN)

    Args:
        args: args from command line

    Returns:
        None
    '''

    print('\nclassical DNN with argument: ' + str(args))

    batch_size = args.batch_size
    runs = args.runs
    dropout = args.dropout
    n_layer = args.n_layer
    n_l1 = args.n_l1
    n_l2 = args.n_l2
    n_l3 = args.n_l3
    n_l4 = args.n_l4
    epochs = args.epochs
    momentum = args.momentum
    weight_decay = args.weight_decay
    scheduler_decrease = args.scheduler_decrease
    lr = args.lr

    npz = os.path.join(args.in_dir, 'input.npz')
    data = np.load(npz)
    split_all = np.load("./split_100.npy")
    tes_phe = data['tes_phe']
    phenos = data['y_test_different_set_phe']
    edges = data['x_test']
    n_rng = 100
    ks = [10, 20, 50, 100, 200]
    meta_cor = np.zeros((n_rng, len(ks), len(tes_phe)))
    meta_p = np.zeros((n_rng, len(ks), len(tes_phe)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(n_rng):
        for ik, k in enumerate(ks):
            for ib, phe in enumerate(tes_phe):
                split_ind = split_all[i, ik, :]
                split = np.squeeze(split_ind)
                split_k = (split == 0)
                split_tes = (split == 1)
                x_train = edges[split_k, :]
                x_test = edges[split_tes, :]
                y_train = phenos[split_k,:]
                y_train = y_train[:,ib].reshape((-1,1))
                y_test = phenos[split_tes,:]
                y_test = y_test[:,ib].reshape((-1,1))

                # z norm based on y_train
                y_train, y_test, _, t_sigma = mics_z_norm(y_train, y_test)
                n_phe = y_train.shape[1]

                # load dataset for PyTorch
                dset_train = ukbb_multi_task_dataset(x_train, y_train)
                trainloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=1)
                dset_test = ukbb_multi_task_dataset(x_test, y_test)
                testLoader = DataLoader(dset_test, batch_size=batch_size, shuffle=True, num_workers=1)

                # Code running - with multiple ensemble runs
                for run in range(runs):
                    # initialization of network
                    if n_layer == 2:
                        net = dnn_2l(x_train.shape[1], n_l1, dropout, output_size=n_phe)
                    elif n_layer == 3:
                        net = dnn_3l(x_train.shape[1], n_l1, n_l2, dropout, output_size=n_phe)
                    elif n_layer == 4:
                        net = dnn_4l(x_train.shape[1], n_l1, n_l2, n_l3, dropout, output_size=n_phe)
                    else:
                        assert False, "Only support 2, 3, 4, 5 layers."
                    net.to(device)

                    # other components of network
                    criterion = msenanloss
                    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                    scheduler = MultiStepLR(
                        optimizer,
                        milestones=[scheduler_decrease, scheduler_decrease * 2],
                        gamma=0.1)

                    # start epoch training
                    for epoch in range(epochs):
                        # training
                        train_loss = 0.0
                        net.train(True)
                        for (x, y) in trainloader:
                            x, y = x.to(device), y.to(device)
                            optimizer.zero_grad()
                            outputs = net(x)
                            mask = torch.isnan(y)
                            y.masked_fill_(mask, 0)
                            loss = criterion(outputs, y, mask=mask)
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                            
                        scheduler.step() 

                net.train(False)
                y_real = np.zeros((0, n_phe))  # real value
                y_pred = np.zeros((0, n_phe))  # prediction value
                for (x, y) in testLoader:
                    x, y = x.to(device), y.to(device)
                    outputs = net(x)
                    y_real = np.concatenate((y_real, y.data.cpu().numpy()), axis=0)
                    y_pred = np.concatenate((y_pred, outputs.data.cpu().numpy()), axis=0)
                
                meta_cor[i, ik, ib], meta_p[i, ik, ib], = pearsonr(y_real.reshape(-1,),y_pred.reshape(-1,))
                
        print("rng %d: cor %.5f, pval %.5f" %(i, np.nanmean(meta_cor[:i + 1, :, :]),np.nanmean(meta_p[:i + 1, :, :])))

    meta_cor_npz = npz = os.path.join(args.in_dir, 'classical_dnn.npz')          
    np.savez(meta_cor_npz, meta_cor=meta_cor, meta_p=meta_p)

    return


def get_args():
    '''function to get args from command line and return the args

    Returns:
        argparse.ArgumentParser: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--in_dir', type=str, default=config.IN_DIR)
    parser.add_argument('--out_dir', '-o', type=str, default=config.OUT_DIR)
    parser.add_argument('--seed', type=int, default=config.RAMDOM_SEED)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--runs', type=int, default=config.RUNS)
    parser.add_argument('--metric', type=str, default='cod')
    parser.add_argument('--gpu', type=int, default=0)

    # hyperparameter
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--scheduler_decrease', type=int, default=75)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_l1', type=int, default=64)
    parser.add_argument('--n_l2', type=int, default=32)
    parser.add_argument('--n_l3', type=int, default=32)
    parser.add_argument('--n_l4', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
