# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T12:37:18.001152Z","iopub.execute_input":"2023-02-02T12:37:18.00268Z","iopub.status.idle":"2023-02-02T12:37:18.806222Z","shell.execute_reply.started":"2023-02-02T12:37:18.002621Z","shell.execute_reply":"2023-02-02T12:37:18.804889Z"}}
import numpy as np  # linear algebra
import h5py
import scipy.io as io
import sklearn

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T12:37:20.569702Z","iopub.execute_input":"2023-02-02T12:37:20.570179Z","iopub.status.idle":"2023-02-02T12:37:20.576533Z","shell.execute_reply.started":"2023-02-02T12:37:20.570134Z","shell.execute_reply":"2023-02-02T12:37:20.575228Z"}}
your_path = '/kaggle/input/deepradar/'
classes = ['LFM', '2FSK', '4FSK', '8FSK', 'Costas', '2PSK', '4PSK', '8PSK', 'Barker', 'Huffman', 'Frank', 'P1', 'P2',
           'P3', 'P4', 'Px', 'Zadoff-Chu', 'T1', 'T2', 'T3', 'T4', 'NM', 'Noise']

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T12:37:22.470506Z","iopub.execute_input":"2023-02-02T12:37:22.471459Z","iopub.status.idle":"2023-02-02T12:40:26.957559Z","shell.execute_reply.started":"2023-02-02T12:37:22.471404Z","shell.execute_reply":"2023-02-02T12:40:26.95455Z"}}
with h5py.File(your_path + 'X_train.mat', 'r') as f:
    X_train = np.array(f['X_train'], dtype='float32').T
with h5py.File(your_path + 'X_val.mat', 'r') as f:
    X_val = np.array(f['X_val'], dtype='float32').T
with h5py.File(your_path + 'X_test.mat', 'r') as f:
    X_test = np.array(f['X_test'], dtype='float32').T

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T13:02:52.519114Z","iopub.execute_input":"2023-02-02T13:02:52.52065Z","iopub.status.idle":"2023-02-02T13:02:52.850289Z","shell.execute_reply.started":"2023-02-02T13:02:52.520555Z","shell.execute_reply":"2023-02-02T13:02:52.848946Z"}}
Y_train = io.loadmat(your_path + 'Y_train.mat')['Y_train']
Y_val = io.loadmat(your_path + 'Y_val.mat')['Y_val']
Y_test = io.loadmat(your_path + 'Y_test.mat')['Y_test']
lbl_train = io.loadmat(your_path + 'lbl_train.mat')['lbl_train']
lbl_test = io.loadmat(your_path + 'lbl_test.mat')['lbl_test']
lbl_val = io.loadmat(your_path + 'lbl_val.mat')['lbl_val']

# %% [markdown]
# ### If we want to use these data with an (Amplitude,Phase) instead of (In-Phase,Quadratore):

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T10:40:20.244529Z","iopub.execute_input":"2023-02-02T10:40:20.245532Z","iopub.status.idle":"2023-02-02T10:40:20.250891Z","shell.execute_reply.started":"2023-02-02T10:40:20.245492Z","shell.execute_reply":"2023-02-02T10:40:20.249517Z"}}
AP = False

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T10:40:22.099427Z","iopub.execute_input":"2023-02-02T10:40:22.09987Z","iopub.status.idle":"2023-02-02T10:40:22.107593Z","shell.execute_reply.started":"2023-02-02T10:40:22.099834Z","shell.execute_reply":"2023-02-02T10:40:22.106493Z"}}
if AP:
    I_tr = X_train[:, :, 0]
    Q_tr = X_train[:, :, 1]
    X_tr = I_tr + 1j * Q_tr

    X_train[:, :, 1] = np.arctan2(Q_tr, I_tr) / np.pi
    X_train[:, :, 0] = np.abs(X_tr)

    I_te = X_test[:, :, 0]
    Q_te = X_test[:, :, 1]
    X_te = I_te + 1j * Q_te

    X_test[:, :, 1] = np.arctan2(Q_te, I_te) / np.pi
    X_test[:, :, 0] = np.abs(X_te)

    del I_tr
    del Q_tr
    del X_tr
    del I_te
    del Q_te
    del X_te

# %% [markdown]
# ### Randomly shuffle the data

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T13:02:57.075385Z","iopub.execute_input":"2023-02-02T13:02:57.076141Z","iopub.status.idle":"2023-02-02T13:03:29.957474Z","shell.execute_reply.started":"2023-02-02T13:02:57.076085Z","shell.execute_reply":"2023-02-02T13:03:29.955976Z"}}
np.random.seed(2022)
X_train, Y_train, lbl_train = sklearn.utils.shuffle(X_train[:], Y_train[:], lbl_train[:], random_state=2022)
X_val, Y_val, lbl_val = sklearn.utils.shuffle(X_val[:], Y_val[:], lbl_val[:], random_state=2022)
X_test, Y_test, lbl_test = sklearn.utils.shuffle(X_test[:], Y_test[:], lbl_test[:], random_state=2022)

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T13:03:54.639303Z","iopub.execute_input":"2023-02-02T13:03:54.63984Z","iopub.status.idle":"2023-02-02T13:03:54.650246Z","shell.execute_reply.started":"2023-02-02T13:03:54.639791Z","shell.execute_reply":"2023-02-02T13:03:54.648705Z"}}
print("X train shape: ", X_train.shape)
print("X vak shape: ", X_val.shape)
print("X test shape: ", X_test.shape)
print("Y train shape: ", Y_train.shape)
print("Y val shape: ", Y_val.shape)
print("Y test shape: ", Y_test.shape)
print("Label train shape: ", lbl_train.shape)
print("Label val shape: ", lbl_val.shape)
print("Label test shape: ", lbl_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T13:10:27.323342Z","iopub.execute_input":"2023-02-02T13:10:27.323799Z","iopub.status.idle":"2023-02-02T13:10:27.332394Z","shell.execute_reply.started":"2023-02-02T13:10:27.323762Z","shell.execute_reply":"2023-02-02T13:10:27.331255Z"}}
[classes[i] for i in range(len(Y_train[0])) if Y_train[0][i] == 1]

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T13:08:07.529772Z","iopub.execute_input":"2023-02-02T13:08:07.530758Z","iopub.status.idle":"2023-02-02T13:08:07.538771Z","shell.execute_reply.started":"2023-02-02T13:08:07.530705Z","shell.execute_reply":"2023-02-02T13:08:07.537395Z"}}
print([i for i in Y_train[1] if i == 1])
print(Y_train[2])
print(Y_train[3])
print(Y_train[4])

# %% [code] {"execution":{"iopub.status.busy":"2023-02-02T13:11:51.617341Z","iopub.execute_input":"2023-02-02T13:11:51.617869Z","iopub.status.idle":"2023-02-02T13:11:51.628239Z","shell.execute_reply.started":"2023-02-02T13:11:51.617825Z","shell.execute_reply":"2023-02-02T13:11:51.626832Z"}}
print("Signal 0 of the training set corresponds to a " +
      [classes[i] for i in range(len(Y_train[0])) if Y_train[0][i] == 1][0] + " modulated signal with SNR " + str(
    lbl_train[0][1]))
print("Signal 1 of the training set corresponds to a " +
      [classes[i] for i in range(len(Y_train[1])) if Y_train[1][i] == 1][0] + " modulated signal with SNR " + str(
    lbl_train[1][1]))
print("Signal 3 of the training set corresponds to a " +
      [classes[i] for i in range(len(Y_train[2])) if Y_train[2][i] == 1][0] + " modulated signal with SNR " + str(
    lbl_train[2][1]))
    lbl_train[2][1]))