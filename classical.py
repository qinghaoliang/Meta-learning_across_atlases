import os
import numpy  as np
from config import config
import argparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
import warnings
from sklearn.preprocessing import StandardScaler
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
    for i in range(n_rng):
        for ik, k in enumerate(ks):
            for ib, phe in enumerate(tes_phe):
                split_ind = split_all[i, ik, :]
                split = np.squeeze(split_ind)
                split_k = (split == 0)
                split_tes = (split == 1)
                phen_train = phenos[split_k,:]
                phen_test = phenos[split_tes,:]
                results_test =[]
                scaler = StandardScaler()
                behav_train = phen_train[:,ib]
                behav_test = phen_test[:,ib]
                behav_train = scaler.fit_transform(behav_train.reshape(-1,1)).reshape(-1,)
                behav_test = scaler.transform(behav_test.reshape(-1,1)).reshape(-1,)
                edges_train = edges[split_k, :]
                edges_test = edges[split_tes, :]
                alphas_ridge=10**np.linspace(3, -10, 20)
                ridge_grid = GridSearchCV(Ridge(), cv=5, param_grid={'alpha': alphas_ridge})
                ridge_grid.fit(edges_train, behav_train)
                y_pred = ridge_grid.predict(edges_test)
                meta_cor[i, ik, ib], meta_p[i, ik, ib], = pearsonr(behav_test, y_pred)

        print("rng %d: cor %.5f, pval %.5f" %(i, np.nanmean(meta_cor[:i + 1, :, :]),np.nanmean(meta_p[:i + 1, :, :])))

    meta_cor_npz = npz = os.path.join(args.in_dir, 'classical.npz')          
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

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())

