# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_list) > 0:
  for gpu in gpu_list:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
       print(e)
else:
   print("Got no GPUs")

from LSTM2_args import get_args
args = get_args()

import keras
import numpy as np
import matplotlib.pyplot as plt
import mltools,rmldataset2016

class LSTMModel(tf.keras.Model):
    def __init__(self, input_size=2, hidden_size=128, output_size=11):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.w_ih1 = self.add_weight(shape=(input_size, 4 * hidden_size))
        self.w_hh1 = self.add_weight(shape=(hidden_size, 4 * hidden_size))
        self.b_ih1 = self.add_weight(shape=(4 * hidden_size,))
        self.b_hh1 = self.add_weight(shape=(4 * hidden_size,))
        self.w_ih2 = self.add_weight(shape=(hidden_size, 4 * hidden_size))
        self.w_hh2 = self.add_weight(shape=(hidden_size, 4 * hidden_size))
        self.b_ih2 = self.add_weight(shape=(4 * hidden_size,))
        self.b_hh2 = self.add_weight(shape=(4 * hidden_size,))
        self.fc_w = self.add_weight(shape=(hidden_size, output_size))
        self.fc_b = self.add_weight(shape=(output_size,))

    def call(self, inputs):
        # print ('1:', tf.shape(inputs), '2:',print (inputs.shape))
        batch_size = tf.shape(inputs)[0]
        print('batch_size: ', batch_size)
        h1 = c1 = tf.zeros((batch_size, self.hidden_size))
        h2 = c2 = tf.zeros((batch_size, self.hidden_size))
        for i in range(inputs.shape[1]):
            x = inputs[:, i, :]
            gates1 = tf.matmul(x, self.w_ih1) + self.b_ih1 + tf.matmul(h1, self.w_hh1) + self.b_hh1
            i1, f1, g1, o1 = tf.split(gates1, num_or_size_splits=4, axis=1)
            i1 = tf.sigmoid(i1)
            f1 = tf.sigmoid(f1)
            g1 = tf.tanh(g1)
            o1 = tf.sigmoid(o1)
            c1 = f1 * c1 + i1 * g1
            h1 = o1 * tf.tanh(c1)

            gates2 = tf.matmul(h1, self.w_ih2) + self.b_ih2 + tf.matmul(h2, self.w_hh2) + self.b_hh2
            i2, f2, g2, o2 = tf.split(gates2, num_or_size_splits=4, axis=1)
            i2 = tf.sigmoid(i2)
            f2 = tf.sigmoid(f2)
            g2 = tf.tanh(g2)
            o2 = tf.sigmoid(o2)
            c2 = f2 * c2 + i2 * g2
            h2 = o2 * tf.tanh(c2)

        out_logits = tf.matmul(h2, self.fc_w) + self.fc_b
        out_softmax = tf.nn.softmax(out_logits)
        return out_softmax

# 创建模型实例
model = LSTMModel(input_size=2, hidden_size=128, output_size=11)

# ###############################################################################
# # 随机生成训练数据
# num_samples = 1000
# sequence_length = 128
# input_dimenstionality = 2
# num_classes = 11
# X_train = np.random.rand(num_samples, sequence_length, input_dimenstionality)
# y_train = np.random.randint(num_classes, size=(num_samples))
# ################################################################################

##################################################################################
print(f'RML2016.a')

(mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
    rmldataset2016.load_data()

in_shp = list(X_train.shape[1:])
print('X_train.shape: ', X_train.shape, 'input shape: ', in_shp)
y_train = Y_train

###############################################################################
adam = keras.optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','mse','mae','mape'])

# 训练模型并保存训练历史
# history = model.fit(X_train, y_train, epochs=100)
history = model.fit(X_train, y_train, batch_size=args.batch_size, validation_data=(X_val, Y_val), epochs=100)
# history = model.fit(X_train, y_train,steps_per_epoch = 1, batch_size=32, shuffle = True, validation_data=(X_val, Y_val), epochs=10000)

# 绘制训练历史图像
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(history.history['loss'], label='loss')
plt.title('Loss history')
plt.subplot(122)
plt.plot(history.history['accuracy'], label='accuracy')
plt.title('Accuracy history')
plt.show()

# 打印误差相关参数
print("Final training loss: ", history.history['loss'][-1])
print("Final training accuracy: ", history.history['accuracy'][-1])

test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("\nTest loss: ", test_loss)
print("Test accuracy: ", test_accuracy)

# 打印所有权重参数到txt文件中
for i in range(len(model.weights)):
    np.savetxt(f'weight_{i}.txt', model.weights[i].numpy())