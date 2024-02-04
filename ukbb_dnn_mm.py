import os
import time
import random
import argparse
import numpy as np
from scipy.stats import pearsonr
from config import config
from mics import mics_z_norm, cod_znormed


def mm_dnn(y_test, y_pred, split_ind):
    '''meta-matching with DNN

    Args:
        y_test (ndarray): original test data
        y_pred (ndarray): predicted for test subjects with base model trained
            on training meta-set
        split_ind (ndarray): array that indicate K subjects

    Returns:
        Tuple: result (correlation and COD) of meta-matching with DNN, best
            matched phenotypes, prediction of meta-matching on k shot subjects
    '''
    # get split from krr pure baseline
    split = np.squeeze(split_ind)
    split_k = split == 0
    split_tes = split == 1

    y_pred_k = y_pred[split_k, :]
    y_pred_remain = y_pred[split_tes, :]

    # z norm based on y_train
    _, y_test, _, _ = mics_z_norm(y_test[split_k], y_test)
    y_test_k = y_test[split_k]
    y_test_remain = y_test[split_tes]

    # exclude nan value from y_test_remain
    real_index = ~np.isnan(y_test_remain)
    y_test_remain = y_test_remain[real_index]
    y_pred_remain = y_pred_remain[real_index, :]

    best_score = float("-inf")
    best_phe = -1
    best_sign = 1
    for i in range(y_pred.shape[1]):
        sign = 1
        tmp = cod_znormed(y_test_k, y_pred_k[:, i])
        y_pred_k_tmp = -y_pred_k[:, i]
        tmp1 = cod_znormed(y_test_k, y_pred_k_tmp)
        if tmp1 > tmp:
            tmp = tmp1
            sign = -1
        if tmp >= best_score:
            best_score = tmp
            best_phe = i
            best_sign = sign
    y_mm_pred = best_sign * y_pred_remain[:, best_phe]
    y_mm_pred_k = best_sign * y_pred_k[:, best_phe]
    res_cor = pearsonr(y_test_remain, y_mm_pred)[0]
    res_cod = cod_znormed(y_test_remain, y_mm_pred)

    return res_cor, res_cod, best_phe, y_mm_pred_k


def main(args):
    '''main function for meta-matching (DNN)

    Args:
        args: args from command line

    Returns:
        None
    '''

    print('\nmeta-matching (DNN) with argument: ' + str(args))

    # set all the seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    n_rng = args.rng
    tuned_by = 'cod'
    ks = [10, 20, 50, 100, 200]
    log_str = args.log_stem + '_result_' + args.split
    meta_cor_npz = os.path.join(args.out_dir, log_str + '.npz')
    meta_temp_npz = os.path.join(args.out_dir, 'tmp/' + log_str + '_rng.npz')
    os.makedirs(os.path.join(args.out_dir, 'tmp'), exist_ok=True)

    # load base prediction result
    npz = os.path.join(args.out_dir, 'dnn_base.npz')
    npz = np.load(npz)
    tes_res_record = npz['tes_res_record']
    val_record = npz['val_' + tuned_by + '_record']
    temp = np.mean(val_record[0, :, :], axis=1)
    temp = np.convolve(temp, np.ones(3, dtype=int), 'valid') / 3
    index = np.nanargmax(temp)
    index = index + 1
    print('\nBest validation at index: ', index)
    y_pred = tes_res_record[0, index, :, :]

    # load original data
    npz = os.path.join(args.in_dir, 'input.npz')
    npz = np.load(npz, allow_pickle=True)
    y_test = npz['y_test_different_set_phe']
    tes_phe = npz['tes_phe']
    tra_phe = npz['tra_phe']

    split = np.load("./split_100.npy")

    # perform meta matching with DNN and transfer learning
    start_time = time.time()
    meta_cor = np.zeros((n_rng, len(ks), len(tes_phe)))
    meta_cod = np.zeros((n_rng, len(ks), len(tes_phe)))
    meta_phe = np.zeros((n_rng, len(ks), len(tes_phe)))

    for i in range(n_rng):
        for ik, k in enumerate(ks):
            for ib, phe in enumerate(tes_phe):
                split_ind = split[i, ik, :]
                meta_cor[i, ik, ib], meta_cod[
                    i, ik, ib], meta_phe[i, ik, ib], tmp_pred = mm_dnn(
                        y_test[:, ib], y_pred, split_ind)
        print("rng %d at %ss: cor %.5f, cod %.5f" %
              (i, time.time() - start_time, np.nanmean(meta_cor[:i + 1, :, :]),
               np.nanmean(meta_cod[:i + 1, :, :])))
        mean_cor = np.squeeze(
            np.nanmean(np.nanmean(meta_cor[:i + 1, :, :], axis=2), axis=0))
        mean_cod = np.squeeze(
            np.nanmean(np.nanmean(meta_cod[:i + 1, :, :], axis=2), axis=0))
        print(' '.join('%.6f' % tmp for tmp in mean_cor), ' COD ',
              ' '.join('%.6f' % tmp for tmp in mean_cod))
        np.savez(
            meta_temp_npz,
            meta_cor=meta_cor,
            meta_cod=meta_cod,
            current_rng=i,
            meta_phe=meta_phe,
            tes_phe=tes_phe,
            tra_phe=tra_phe)

    np.savez(
        meta_cor_npz,
        meta_cor=meta_cor,
        meta_cod=meta_cod,
        meta_phe=meta_phe,
        tes_phe=tes_phe,
        tra_phe=tra_phe)

    return


def get_args():
    '''function to get args from command line and return the args

    Returns:
        argparse.ArgumentParser: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--out_dir', '-o', type=str, default=config.OUT_DIR)
    parser.add_argument('--in_dir', type=str, default=config.IN_DIR)
    parser.add_argument('--inter_dir', type=str, default=config.INTER_DIR)
    parser.add_argument('--log_stem', type=str, default='meta')
    parser.add_argument('--seed', type=int, default=config.RAMDOM_SEED)
    parser.add_argument('--rng', type=int, default=100)
    parser.add_argument('--split', type=str, default='test')

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
