import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename=r"./dataset/RML2016.10a_dict.pkl"):
#    Xd1 = pickle.load(open(filename1,'rb'),encoding='iso-8859-1')#Xd1.keys() mod中没有AM-SSB Xd1(120W,2,128)
    Xd =pickle.load(open(filename,'rb'),encoding='iso-8859-1')#Xd2(22W,2,128)
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ]
    X = []
    # X2=[]
    lbl = []
    # lbl2=[]
    train_idx=[]
    val_idx=[]
    np.random.seed(2016)
    a=0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     #ndarray(1000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
            train_idx+=list(np.random.choice(range(a*1000,(a+1)*1000), size=600, replace=False))
            val_idx+=list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=200, replace=False))
            a+=1
    X = np.vstack(X) #(220000,2,128)  mods * snr * 1000,total 220000 samples
    n_examples=X.shape[0]
    # n_test=X2.shape[0]
    test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    # test_idx=np.random.choice(range(0,n_test),size=n_test,replace=False)
    X_train = X[train_idx]
    X_val=X[val_idx]
    X_test =  X[test_idx]
    print(len(train_idx))
    print(len(val_idx))
    print(len(test_idx))


    def to_onehot(yy):
        # yy1 = np.zeros([len(yy), max(yy)+1])
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    # yy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val=to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    X_train=X_train.swapaxes(2,1)
    X_val=X_val.swapaxes(2,1)
    X_test=X_test.swapaxes(2,1)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_val.shape)
    print(Y_test.shape)
    return (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx)



def plot_constellation(X, mod, snr, idx):
    data = X[idx]
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], marker='.', color='red')
    plt.title(f"constellation - modulation: {mod} - SNR: {snr} dB")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.show()

def plot_all_snrs_for_mod(mod, X, lbl):
    snrs_for_mod = sorted(set([l[1] for l in lbl if l[0] == mod]))

    for snr in snrs_for_mod:
        # 获取这个调制格式和SNR的所有样本的索引
        idxs_for_snr = [i for i, l in enumerate(lbl) if l == (mod, snr)]

        # 从这些样本中随机选择一个来绘制。如果要绘制所有样本，请使用循环。
        sample_idx = np.random.choice(idxs_for_snr)
        plot_constellation(X, mod, snr, sample_idx)

# def plot_all_snrs_for_mod(mod, X, lbl):
#     plt.figure(figsize=(20, 20))
#     for snr in snrs:
#         # 此处确定样本的索引
#         idx = [i for i, label in enumerate(lbl) if label == (mod, snr)]
#         if len(idx) == 0:
#             continue
#
#         plt.subplot(len(snrs), 1, snrs.index(snr) + 1)
#         plot_constellation(X, mod, snr, np.random.choice(idx, 1)[0])
#
#
# def plot_constellation(X, mod, snr, sample_idx):
#     sample_signal = X[sample_idx]
#     I = sample_signal[0, :]
#     Q = sample_signal[1, :]
#     plt.scatter(I, Q, c='r', marker='o')
#     plt.title(f"星座图 for {mod} at {snr} dB")
#     plt.xlabel("I")
#     plt.ylabel("Q")
#     plt.grid(True)
#     plt.xlim([-2, 2])
#     plt.ylim([-2, 2])

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train, Y_train),(X_val,Y_val), (X_test, Y_test), (train_idx,val_idx,test_idx) = load_data()
    # chosen from ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    chosen_mod = "CPFSK"
    chosen_snr = 18
    lbl_test = [lbl[i] for i in test_idx]
    sample_idx = np.random.choice(np.where(np.array(lbl_test) == (chosen_mod, chosen_snr))[0], 1)[0]
    print(sample_idx)
    plot_constellation(X_test, chosen_mod, chosen_snr, 36976)

    # plot_all_snrs_for_mod(chosen_mod, X_test, lbl)
