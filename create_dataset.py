import numpy as np

x_train_raw = np.load("./data/meta_train.npz")
x_test = np.load("./data/meta_test.npz")
phes = np.load("phes.npz")
tra_phe, tes_phe = phes['tra_phe'], phes['tes_phe']
y_train_raw, y_test_different_set_phe = phes['y_train_raw'], phes['y_test_different_set_phe']
np.savez('./input/input.npz', tes_phe=tes_phe, tra_phe=tra_phe, x_test=x_test, x_train_raw=x_train_raw,\
        y_test_different_set_phe=y_test_different_set_phe, y_train_raw=y_train_raw)