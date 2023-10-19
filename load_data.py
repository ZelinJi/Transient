import pickle
import numpy as np
import torch

def load_data():
    with open(r"./dataset/radar_data.pkl", 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        # data.keys() = ['8psk', -10]
        snrs, mods, trans = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [2, 1, 0])
        X = []
        lbl = []

        for mod in mods:
            for snr in snrs:
                X.append(data[(mod, snr)])
                for i in range(data[(mod, snr)].shape[0]): lbl.append((mod, snr))
        X = np.vstack(X)

        np.random.seed(2023)
        n_examples = X.shape[0]
        n_train = n_examples * 0.8     # 划分数据集 n_train : n_test = 8:2
        train_idx = np.random.choice(range(0, int(n_examples)), size=int(n_train), replace=False)
        test_idx = list(set(range(0, n_examples)) - set(train_idx))
        X_train = X[train_idx]
        X_test = X[test_idx]
        def to_onehot(yy):
            yy1 = np.zeros([len(yy), max(yy)+1])
            yy1[np.arange(len(yy)), yy] = 1
            return yy1
        Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
        Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

        X_train = torch.from_numpy(X_train)
        X_train = X_train.unsqueeze(1)      # [176000,1,2,128]
        Y_train = torch.from_numpy(Y_train)
        X_test = torch.from_numpy(X_test)
        X_test = X_test.unsqueeze(1)        # [44000,1,2,128]
        Y_test = torch.from_numpy(Y_test)

        return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()
