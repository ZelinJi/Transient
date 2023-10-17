import numpy as np
from scipy import signal
from scipy import interpolate
import sys
import math
import matplotlib.pyplot as plt


# from chart.test import min_indices, max_indices

# Find peaks
# order:两侧使用多少点进行比较
def find_peaks(data):
    global peak, peak_indexes, Max_index, Max, peak_data_left, peak_data_right
    cross_point = find_zero_crossings(data_y)  # 此处的crossPoint是寻找输入信号的过零点来初步判断输入信号的周期长度
    if len(cross_point) >= 3:  # 如果过零点数量大于3则基本可以断定为一个完整周期，在周期内寻找极大值，否则不满一个周期在全局搜索极大值
        period_length = (cross_point[2] - cross_point[0])
        peak_indexes = signal.argrelextrema(data, np.greater, order=period_length // 2)  # 此处的peak_indexes是最大值对应的序列下标
        # order: How many points on each side to use for the comparison to consider comparator(n, n+x) to be True.
        # print(peak_indexes)
    else:
        peak_indexes = signal.argrelextrema(data, np.greater, order=len(data_y) // 2)  # 此处的peak_indexes是最大值对应的序列下标

    peak_indexes = peak_indexes[0]

    if len(peak_indexes) == 0:
        Max_index = np.argmax(data)
        Max.append(data_y[Max_index])
        peak_data_left = data_y[0:Max_index + 1]
        peak_data_right = data_y[Max_index + 1:-1]
        return Max

    # 在此处需要设置中断，有可能报错index 0 is out of bounds for axis 0 with size 0，此时输出没有找到最大值

    peak_indexe1 = int(peak_indexes[0])
    # peak = data_y[peak_indexe1]
    peak.append(data_y[peak_indexe1])
    peak_data_left = data_y[0:peak_indexe1 + 1]  # 从极大值开始向前或者向后截取序列
    peak_data_right = data_y[peak_indexe1 + 1:-1]
    # print(peak_data_left)
    # print(peak_data_right)
    # print(peak)


# Find valleys
# order:两侧使用多少点进行比较
def find_valleys(data):
    global valley_data_left, valley_data_right, valley_indexes, valley, Min_index, Min

    cross_point = find_zero_crossings(data_y)  # 此处的crossPoint是寻找输入信号的过零点来初步判断输入信号的周期长度
    if len(cross_point) >= 3:  # 如果过零点数量大于3则基本可以断定为一个完整周期，在周期内寻找极大值，否则不满一个周期在全局搜索极大值
        period_length = (cross_point[2] - cross_point[0]) // 2
        valley_indexes = signal.argrelextrema(data, np.less, order=period_length // 2)  # 此处的peak_indexes是最大值对应的序列下标
        # print(valley_indexes)
    else:
        valley_indexes = signal.argrelextrema(data, np.less, order=len(data_y) // 2)  # 此处的peak_indexes是最大值对应的序列下标

    # valley_indexes = signal.argrelextrema(data, np.less, order=10)  # 此处的valley_indexes是最大值对应的序列下标
    valley_indexes = valley_indexes[0]

    # 在此处需要设置中断，有可能报错index 0 is out of bounds for axis 0 with size 0，此时输出没有找到最小值
    if len(valley_indexes) == 0:
        Min_index = np.argmin(data)
        Min.append(data_y[Min_index])
        valley_data_left = data_o[0:Min_index + 1]
        valley_data_right = data_o[Min_index + 1:-1]
        return Min

    valley_indexe1 = int(valley_indexes[0])
    # valley = data_y[valley_indexe1]
    valley.append(data_y[valley_indexe1])
    valley_data_left = data_o[0:valley_indexe1 + 1]  # 从极小值开始向前或者向后截取序列
    valley_data_right = data_o[valley_indexe1 + 1:-1]
    # print(valley_data_left)
    # print(valley_data_right)
    # print(valley)


# # 判断非完整周期序列含有的是极大值还是极小值
# diff = np.diff(np.sign(np.diff(data_y)))
#
# # 使用 np.diff 和 np.sign 函数计算极值点
# diff = np.diff(np.sign(np.diff(data_y)))
# max_indices = np.where(diff == -2)[0] + 1
# min_indices = np.where(diff == 2)[0] + 1

count = 0
period = None


def period_judge(data):
    global count, period
    for i in data:
        if data[0] < data[1] < data[2]:
            if (i > data[0]) & (i < data[2]):
                count += 1
        elif data[0] > data[1] > data[2]:
            if (i < data[0]) & (i > data[2]):
                count += 1
        elif (data[1] > data[0]) & (data[1] > data[2]):
            if (i > data[0]) & (i > data[2]):
                count += 1
        elif (data[1] < data[0]) & (data[1] < data[2]):
            if (i < data[0]) & (i < data[2]):
                count += 1
        else:
            print(f"当前波形不符合识别标准")
    if count >= 3:
        period = True
    return period


# 寻找离散周期序列中的过零点
# positive_mask[:-1] 表示 positive_mask 数组的切片，去除最后一个元素。这是因为异或操作需要两个数组具有相同的长度。
# positive_mask[1:] 表示 positive_mask 数组的切片，去除第一个元素。这是因为异或操作需要两个数组具有相同的长度。
# np.logical_xor(positive_mask[:-1], positive_mask[1:]) 执行逻辑异或操作，返回一个布尔数组，表示满足逻辑异或条件的位置。
# np.where(...)[0] 使用 np.where 函数找到布尔数组中为 True 的位置，并返回一个包含索引的数组。由于 np.where 的返回结果是一个元组，其中包含满足条件的位置数组，因此我们使用 [0] 来获取第一个元素，即位置数组。
# + 1 对位置数组进行修正，将索引从零开始转换为从一开始。这是因为 Python 中的索引是从零开始的，但在某些应用场景中，习惯从一开始计数。
def find_zero_crossings(x):
    positive_mask = x > 0
    #  np.where 函数找到布尔数组中为 True 的位置,括号内判断了从第一位开始相邻两数据的正负关系是否相同。
    zero_crossings = np.where(np.logical_xor(positive_mask[:-1], positive_mask[1:]))[0] + 1
    cross_data_left = x[zero_crossings - 1]  # 过零点左测的数据
    # return zero_crossings, cross_data_left
    return zero_crossings


# 生成测试集  right_left的左右二分之一训练集，从-1到1或者从1到-1
def srlgenerate(x1, x2, size):
    x1 = math.asin(x1)
    x2 = math.asin(x2)
    x = np.linspace(x1, x2, size)
    y = np.sin(x)

    return y


def trlgenerate(x1, x2, size):
    y = np.linspace(x1, x2, size)

    return y


# 生成测试集  up_down的上下二分之一训练集，从0到1到0或者从0到-1到0
def sudgenerate(x1, x2, size):
    x1 = math.asin(x1)
    x2 = math.pi - math.asin(x2)
    x = np.linspace(x1, x2, size)
    y = np.sin(x)

    return y


def tudgenerate(x1, x2, size):
    x2 = 2 - x2
    x = np.linspace(x1, x2, size)
    y = x
    for i in range(len(x)):
        if x[i] <= 1:
            y[i] = x[i]
        else:
            y[i] = -x[i] + 2

    return y


def knn_loop(test_data):
    answer = [0, 0, 0]
    s1 = sudgenerate(0, 0, 50)
    s2 = sudgenerate(0.3, 0.3, 50)
    s3 = sudgenerate(0.7, 0.7, 50)
    s4 = sudgenerate(0.3, 0, 50)
    s5 = sudgenerate(0.7, 0.3, 50)
    s6 = sudgenerate(0.9, 0, 50)
    s7 = sudgenerate(0, 0.9, 50)
    s8 = sudgenerate(0.3, 0.7, 50)
    s9 = sudgenerate(0, 0.3, 50)
    s10 = sudgenerate(-0.1, -0.1, 50)
    t1 = tudgenerate(0, 0, 50)
    t2 = tudgenerate(0.3, 0.3, 50)
    t3 = tudgenerate(0.7, 0.7, 50)
    t4 = tudgenerate(0.3, 0, 50)
    t5 = tudgenerate(0.7, 0, 50)
    t6 = tudgenerate(0.9, 0, 50)
    t7 = tudgenerate(0, 0.9, 50)
    t8 = tudgenerate(0, 0.7, 50)
    t9 = tudgenerate(0, 0.3, 50)
    t10 = tudgenerate(-0.1, 0.1, 50)
    # 计算测试集的平均值
    avg = sum(test_data) / len(test_data)
    # 根据平均值判断实在x轴上半部分还是下半部分来处理测试集
    if avg < 0:
        s1 = -s1
        s2 = -s2
        s3 = -s3
        s4 = -s4
        s5 = -s5
        s6 = -s6
        s7 = -s7
        s8 = -s8
        s9 = -s9
        s10 = -s10
        t1 = -t1
        t2 = -t2
        t3 = -t3
        t4 = -t4
        t5 = -t5
        t6 = -t6
        t7 = -t7
        t8 = -t8
        t9 = -t9
        t10 = -t10

    # 初始化k1，k2，分别为正弦波训练集和三角波训练集的记录数组
    k1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    k2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 使用knn算法计算欧式距离
    for i in range(len(test_data)):
        k1[0] = k1[0] + abs(test_data[i] - s1[i])
        k1[1] = k1[1] + abs(test_data[i] - s2[i])
        k1[2] = k1[2] + abs(test_data[i] - s3[i])
        k1[3] = k1[3] + abs(test_data[i] - s4[i])
        k1[4] = k1[4] + abs(test_data[i] - s5[i])
        k1[5] = k1[5] + abs(test_data[i] - s6[i])
        k1[6] = k1[6] + abs(test_data[i] - s7[i])
        k1[7] = k1[7] + abs(test_data[i] - s8[i])
        k1[8] = k1[8] + abs(test_data[i] - s9[i])
        k1[9] = k1[9] + abs(test_data[i] - s10[i])

        k2[0] = k2[0] + abs(test_data[i] - t1[i])
        k2[1] = k2[1] + abs(test_data[i] - t2[i])
        k2[2] = k2[2] + abs(test_data[i] - t3[i])
        k2[3] = k2[3] + abs(test_data[i] - t4[i])
        k2[4] = k2[4] + abs(test_data[i] - t5[i])
        k2[5] = k2[5] + abs(test_data[i] - t6[i])
        k2[6] = k2[6] + abs(test_data[i] - t7[i])
        k2[7] = k2[7] + abs(test_data[i] - t8[i])
        k2[8] = k2[8] + abs(test_data[i] - t9[i])
        k2[9] = k2[9] + abs(test_data[i] - t10[i])

    # s = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # # 临时去零
    # for i in s:
    #     if k1[i] == 0:
    #         del k1[i]
    #     if k2[i] == 0:
    #         del k2[i]

    # 对k1和k2进行排序
    for i in range(len(k1) - 1):
        for j in range(len(k1) - i - 1):
            if k1[j] > k1[j + 1]:
                temp = k1[j]
                k1[j] = k1[j + 1]
                k1[j + 1] = temp

    for i in range(len(k2) - 1):
        for j in range(len(k2) - i - 1):
            if k2[j] > k2[j + 1]:
                temp = k2[j]
                k2[j] = k2[j + 1]
                k2[j + 1] = temp

    def gaussian(dist, sigma=2):
        # 采用Gaussian函数进行不同距离的样本的权重优化，当训练样本与测试样本距离↑，该距离值权重↓。
        # 给更近的邻居分配更大的权重(你离我更近，那我就认为你跟我更相似，就给你分配更大的权重)，而较远的邻居的权重相应地减少，取其加权平均。
        """ Input a distance and return it`s weight"""
        weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
        return weight

    # print(k1)
    # print(k2)

    k = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    s_count = 0
    t_count = 0

    # 使用加权knn算法，记录距离两个集合的总权重
    for i in range(len(k)):
        if k1[0] < k2[0]:
            k[i] = k1[0]
            del k1[0]
            weight = gaussian(k[i])
            s_count = s_count + weight
        else:
            k[i] = k2[0]
            del k2[0]
            weight = gaussian(k[i])
            t_count = t_count + weight

    # 设置异常半径
    strange_flag = 1
    for i in range(len(k)):
        if k[i] < 5:
            strange_flag = 0
    # print(k)
    answer[0] = s_count
    answer[1] = t_count
    answer[2] = strange_flag

    return answer


'''
计算绝对值abs，计算abs后的序列的平均值，将所有数据除以平均值
首先判断是不是方波（遍历abs和均值后的所有数据，如果超过95%的数据均大于0.9小于1.1，则判定为方波）
'''

# 数据输入

pulse_data = np.loadtxt('./dataset/pulse_data')
triangle251_data = np.loadtxt('./dataset/triangle251_data')
triangle51_data = np.loadtxt('./dataset/triangle51_data')
triangle1001_data = np.loadtxt('./dataset/triangle1001_data')
sin251_data = np.loadtxt('./dataset/sin251_data', )
sin51_data = np.loadtxt('./dataset/sin51_data')
sin1001_data = np.loadtxt('./dataset/sin1001_data')
sin_sequence_20dB = np.loadtxt('./dataset/sin_sequence_20dB')
triangle_sequence_10dB = np.loadtxt('./dataset/triangle_sequence_10dB')
mix_data = np.loadtxt('./dataset/mix_data')

# 将序列转换为NumPy数组
# data_y = sin_sequence_20dB
data_y = triangle251_data
data_o = data_y

# data_y = tudgenerate(0, 0, 100)  #整周期为200个点
data_x = np.arange(start=0, stop=len(data_y), step=1, dtype='int')

# 平滑处理
# data_y = signal.savgol_filter(data_y, 3, 1)

# 序列求绝对值
abs_data_y = list(abs(i) for i in data_y)
# print(abs_data_y)

# 计算abs后的序列的平均值，将所有数据除以平均值
mean_abs_data_y = []
for i in abs_data_y:
    i /= np.mean(abs_data_y)
    mean_abs_data_y.append(i)
# print(mean_abs_data_y)

# scipy.signal.argrelextrema是Scipy库中的一个函数,而numpy的操作通常比scipy的操作更快.
peak = []
valley = []
peak_data_left = []
peak_data_right = []
valley_data_left = []
valley_data_right = []
peak_indexes = []
valley_indexes = []

Max = []
Min = []
Max_index = 0
Min_index = 0

knn_flag = 0
knnDirect_flag = 0
loop_flag = 0

percent_list = []

test_data = []

num = 0
data_length = len(data_y)
for i in mean_abs_data_y:
    if (i > 0.9) & (i < 1.1):
        num += 1
per = num / data_length
if per > 0.95:
    print(f"该序列为方波信号")
    # 画出方波信号
    x_values = np.linspace(0, len(pulse_data) - 1, len(pulse_data))
    plt.plot(x_values, pulse_data)
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.grid(True)
    plt.show()

    sys.exit(0)
else:
    print("初步判断该序列为正弦波或三角波")

    # 找到极大值后，对所有数据进行归一化
    # f_abs_data_y = []
    # for i in abs_data_y:
    #     i /= peak  # 对数据逐个处理
    #     f_abs_data_y.append(i)
    # print(f_abs_data_y)  # f_data_y是归一化之后的序列

# if len(peak) > 0:
#     print(f"序列中含有极大值{peak}.")
# else:
#     print("序列中不含有极大值。")
#
# if len(valley) > 0:
#     print(f"序列中含有极小值{valley}.")
# else:
#     print("序列中不含有极小值。")

# 平滑处理
data_y = signal.savgol_filter(data_y, 3, 1)

# 将测试集归一化
M1 = 0
M2 = 0
m1 = 0
m2 = 0
if len(valley) > 0 or len(peak) > 0:
    if len(valley) > 0:
        M1 = abs(np.max(valley))
        m1 = abs(np.min(valley))
    if len(peak) > 0:
        M2 = abs(np.max(peak))
        m2 = abs(np.min(peak))
    if M1 > M2:
        M = M1
    else:
        M = M2
    if m1 > m2:
        m = m1
        M = m1
    else:
        m = m2
        M = m2
else:
    # 求出测试集的最大值
    M = abs(np.max(data_y))
    m = abs(np.min(data_y))

if M > m:
    data_o = data_o / M
else:
    data_o = data_o / m
    M = m

# 寻找极值点
find_peaks(data_y)
find_valleys(data_y)

start = data_y[1] - data_y[0]
middle = data_y[data_length // 2 + 1] - data_y[data_length // 2]
end = data_y[-1] - data_y[-2]
period_judge(data_y)  # 调用后返还的是True或False
if not period:  # 如果序列没有一个完整的周期
    f = np.sum(np.signbit(data_y) > 0)  # 统计布尔数据中true的个数,即data_y中小于0的数据个数
    # print(f)
    n = len(data_y)
    per = f / n
    if (per > 0.1) & (per < 0.9):  # 判断异号的数据个数，确定是否有过零点
        # print(f"当前的数据含有过零点")
        derivative_s_m_e = start * middle * end
        knn_flag = 1
        if derivative_s_m_e >= 0:
            test_data = data_o
            # print(f"序列单调（左右二分之一周期波形作为测试集）")
        else:
            # 序列含有极大值或者极小值
            if (len(valley) > 0) & (len(peak) == 0):
                # print(f"序列中含有极小值{valley}，不含有极大值")
                if len(valley_data_left) > data_length / 2:
                    test_data = valley_data_left
                    # print(f"将极小值左侧数据作为测试集（左右二分之一周期波形作为测试集）")
                else:
                    test_data = valley_data_right
                    # print(f"将极小值右侧数据作为测试集（左右二分之一周期波形作为测试集）")
            elif (len(valley) == 0) & (len(peak) > 0):
                # print(f"序列中含有极大值{peak}，不含有极小值")
                if len(peak_data_left) > data_length / 2:
                    test_data = peak_data_left
                    # print(f"将极大值左侧数据作为测试集（左右二分之一周期波形作为测试集）")
                else:
                    test_data = peak_data_right
                    # print(f"将极大值右侧数据作为测试集（左右二分之一周期波形作为测试集）")
            elif (len(valley) > 0) & (len(peak) > 0):
                # print(f"序列中含有极大值{peak}，也含有极小值{valley}")
                # 现在需要从极大极小值处进行剪切。
                if valley_indexes[0] > peak_indexes[0]:
                    test_data = data_y[peak_indexes[0]:valley_indexes[0] + 1]
                else:
                    test_data = data_y[valley_indexes[0]:peak_indexes[0] + 1]
                # print(f"从极大极小值处截取波形作为测试集（左右二分之一周期波形作为测试集）")
            elif (len(valley) == 0) & (len(peak) == 0):
                # print(f"序列中不含有极大值和极小值")
                if Min_index > Max_index:
                    test_data = data_o[Max_index:Min_index + 1]
                else:
                    test_data = data_o[Min_index:Max_index + 1]
                # print(f"从最大最小值处截取波形作为测试集（左右二分之一周期波形作为测试集）")
    else:
        # print(f"当前数据不含有过零点")
        derivative_s_e = start * end
        test_data = data_o
        if derivative_s_e >= 0:
            knn_flag = 2
            # print(f"则整个序列不包括极值，或者起点和终点中有一个是极值点")
            if (start >= 0) & (end >= 0) & (start > end):
                knnDirect_flag = 1
                # print(f"进入训练集a")
            elif (start <= 0) & (end <= 0) & (start > end):
                knnDirect_flag = 2
                # print(f"进入训练集b")
            elif (start <= 0) & (end <= 0) & (start < end):
                knnDirect_flag = 3
                # print(f"进入训练集c")
            elif (start >= 0) & (end >= 0) & (start < end):
                knnDirect_flag = 4
                # print(f"进入训练集d")
        else:
            knn_flag = 3
            # print(f"则整个序列包括极值，上下二分之一周期波形作为测试集")

if period:  # 如果序列有一个完整的周期
    # 寻找过零点
    knn_flag = 3
    crossings = find_zero_crossings(data_y)
    if len(crossings) < 1:
        print("周期判断错误")
    elif 1 <= len(crossings) < 2:
        test_data = data_o[0:crossings[0]]  # 一个从0开始到0结束的标准波形，因为开头和结尾可能都是无限趋于零但是不越过零的正数或者负数，所以导致只有正中间的点为过零点
        loop_flag = 1  # 仅有一个周期，且波形应该关于过零点对称
    elif 2 <= len(crossings) < 3:
        test_data = data_o[crossings[0]:crossings[1]]
        loop_flag = 2  # 序列长度介于一个周期和一个半周期之间
    elif len(crossings) >= 3:
        test_data = data_o[crossings[0]:crossings[1]]
        loop_flag = 3  # 序列长度一定大于一个完整周期
    # # 输出新的过零点测试集
    # print(f"完整周期的上下二分之一波形测试集：", data_y)

    # 输出新的极大极小值测试集
    # if valley_indexes[0] > peak_indexes[0]:
    #     # data_y = data_y[peak_indexes[0]:valley_indexes[0] + 1]
    #     test_data = data_y[peak_indexes[0]:valley_indexes[0] + 1]
    # else:
    #     # data_y = data_y[valley_indexes[0]:peak_indexes[0] + 1]
    #     test_data = data_y[valley_indexes[0]:peak_indexes[0] + 1]
    # print(f"从极大极小值处截取数据量较大的一半波形作为测试集（左右二分之一周期波形作为测试集）:", test_data)

# knn算法
# 数据归一化
test_data = np.array(test_data)


# 根据输入数据的大小对数据进行抽取或者插值，将目标数据变为50个
def resample_array(arr, target_size=50):
    # 数组原始大小
    original_size = arr.shape[0]

    if original_size == target_size:
        # 如果数组的大小已经是目标大小，则直接返回
        return arr
    elif original_size < target_size:
        # 如果数组的大小小于目标大小，进行插值
        x = np.linspace(0, original_size - 1, num=original_size, dtype=float)
        y = arr
        f = interpolate.interp1d(x, y)

        x_new = np.linspace(0, original_size - 1, num=target_size, dtype=float)
        y_new = f(x_new)

        return y_new
    else:
        # 如果数组的大小大于目标大小，进行抽取
        gap = original_size / target_size
        gap_List = np.linspace(0, 50 * gap - 1, 50)
        gap_List = np.around(gap_List)
        y_new = np.zeros(len(gap_List))
        for i in range(50):
            t = int(gap_List[i])
            y_new[i] = arr[int(gap_List[i])]

        return y_new


# 对测试集合进行处理
test_data = resample_array(test_data, 50)

if knn_flag == 1:
    # print(f"生成左右二分之一训练集")
    # 根据起点终点，生成同样数量的正弦波和三角波训练集
    s1 = srlgenerate(-1, 1, 50)
    s2 = srlgenerate(-0.1, 1, 50)
    s3 = srlgenerate(-1, 0.1, 50)
    s4 = srlgenerate(-0.3, 1, 50)
    s5 = srlgenerate(-0.7, 1, 50)
    s6 = srlgenerate(-1, 0.3, 50)
    s7 = srlgenerate(-1, 0.7, 50)
    s8 = srlgenerate(-0.7, 0.7, 50)
    s9 = srlgenerate(-0.3, 0.3, 50)

    t1 = trlgenerate(-1, 1, 50)
    t2 = trlgenerate(-0.1, 1, 50)
    t3 = trlgenerate(-1, 0.1, 50)
    t4 = trlgenerate(-0.3, 1, 50)
    t5 = trlgenerate(-0.7, 1, 50)
    t6 = trlgenerate(-1, 0.3, 50)
    t7 = trlgenerate(-1, 0.7, 50)
    t8 = trlgenerate(-0.7, 0.7, 50)
    t9 = trlgenerate(-0.3, 0.3, 50)

    # 计算起始点导数，若小于零则认为是从1到-1，将测试集取负
    test_data_s = signal.savgol_filter(test_data, 3, 1)
    derivative_s = test_data_s[1] - test_data_s[0]

    if derivative_s < 0:
        s1 = -s1
        s2 = -s2
        s3 = -s3
        s4 = -s4
        s5 = -s5
        s6 = -s6
        s7 = -s7
        s8 = -s8
        s9 = -s9

        t1 = -t1
        t2 = -t2
        t3 = -t3
        t4 = -t4
        t5 = -t5
        t6 = -t6
        t7 = -t7
        t8 = -t8
        t9 = -t9

elif knn_flag == 2:
    # print(f"生成四分之一训练集")
    # 根据起点终点，生成同样数量的正弦波和三角波训练集
    if knnDirect_flag == 1:
        s1 = srlgenerate(-0.1, 1, 50)
        s2 = srlgenerate(0, 1, 50)
        s3 = srlgenerate(0.1, 1, 50)
        s4 = srlgenerate(0.2, 1, 50)
        s5 = srlgenerate(0.3, 1, 50)
        s6 = srlgenerate(0.4, 1, 50)
        s7 = srlgenerate(0.5, 1, 50)
        s8 = srlgenerate(0.6, 1, 50)
        s9 = srlgenerate(0.7, 1, 50)

        t1 = trlgenerate(-0.1, 1, 50)
        t2 = trlgenerate(0, 1, 50)
        t3 = trlgenerate(0.1, 1, 50)
        t4 = trlgenerate(0.2, 1, 50)
        t5 = trlgenerate(0.3, 1, 50)
        t6 = trlgenerate(0.4, 1, 50)
        t7 = trlgenerate(0.5, 1, 50)
        t8 = trlgenerate(0.6, 1, 50)
        t9 = trlgenerate(0.7, 1, 50)

    # 四分之一测试集存在四种状态，根据预先设定的a，b，c，d四个条件将生成的训练集变为适应的类型
    elif knnDirect_flag == 2:
        s1 = s1[::-1]
        s2 = s2[::-1]
        s3 = s3[::-1]
        s4 = s4[::-1]
        s5 = s5[::-1]
        s6 = s6[::-1]
        s7 = s7[::-1]
        s8 = s8[::-1]
        s9 = s9[::-1]

        t1 = t1[::-1]
        t2 = t2[::-1]
        t3 = t3[::-1]
        t4 = t4[::-1]
        t5 = t5[::-1]
        t6 = t6[::-1]
        t7 = t7[::-1]
        t8 = t8[::-1]
        t9 = t9[::-1]

    elif knnDirect_flag == 3:
        s1 = -s1
        s2 = -s2
        s3 = -s3
        s4 = -s4
        s5 = -s5
        s6 = -s6
        s7 = -s7
        s8 = -s8
        s9 = -s9

        t1 = -t1
        t2 = -t2
        t3 = -t3
        t4 = -t4
        t5 = -t5
        t6 = -t6
        t7 = -t7
        t8 = -t8
        t9 = -t9

    elif knnDirect_flag == 4:
        s1 = -s1[::-1]
        s2 = -s2[::-1]
        s3 = -s3[::-1]
        s4 = -s4[::-1]
        s5 = -s5[::-1]
        s6 = -s6[::-1]
        s7 = -s7[::-1]
        s8 = -s8[::-1]
        s9 = -s9[::-1]

        t1 = -t1[::-1]
        t2 = -t2[::-1]
        t3 = -t3[::-1]
        t4 = -t4[::-1]
        t5 = -t5[::-1]
        t6 = -t6[::-1]
        t7 = -t7[::-1]
        t8 = -t8[::-1]
        t9 = -t9[::-1]

elif knn_flag == 3:
    # print(f"生成上下二分之一训练集")
    # 根据起点终点，生成同样数量的正弦波和三角波训练集
    s1 = sudgenerate(0, 0, 50)
    s2 = sudgenerate(0.3, 0.3, 50)
    s3 = sudgenerate(0.7, 0.7, 50)
    s4 = sudgenerate(0.3, 0, 50)
    s5 = sudgenerate(0.7, 0.3, 50)
    s6 = sudgenerate(0.9, 0, 50)
    s7 = sudgenerate(0, 0.9, 50)
    s8 = sudgenerate(0.3, 0.7, 50)
    s9 = sudgenerate(0, 0.3, 50)
    s10 = sudgenerate(-0.1, -0.1, 50)

    t1 = tudgenerate(0, 0, 50)
    t2 = tudgenerate(0.3, 0.3, 50)
    t3 = tudgenerate(0.7, 0.7, 50)
    t4 = tudgenerate(0.3, 0, 50)
    t5 = tudgenerate(0.7, 0, 50)
    t6 = tudgenerate(0.9, 0, 50)
    t7 = tudgenerate(0, 0.9, 50)
    t8 = tudgenerate(0, 0.7, 50)
    t9 = tudgenerate(0, 0.3, 50)
    t10 = tudgenerate(-0.1, 0.1, 50)

    # 计算测试集的平均值
    avg = sum(test_data) / len(test_data)

    # 根据平均值判断实在x轴上半部分还是下半部分来处理测试集
    if avg < 0:
        s1 = -s1
        s2 = -s2
        s3 = -s3
        s4 = -s4
        s5 = -s5
        s6 = -s6
        s7 = -s7
        s8 = -s8
        s9 = -s9
        s10 = -s10

        t1 = -t1
        t2 = -t2
        t3 = -t3
        t4 = -t4
        t5 = -t5
        t6 = -t6
        t7 = -t7
        t8 = -t8
        t9 = -t9
        t10 = -t10

else:
    print(f"error")

# 计算距离
# print(test_data)
if knn_flag == 1:

    # 初始化k1，k2，分别为正弦波训练集和三角波训练集的记录数组
    k1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    k2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 使用knn算法计算欧式距离
    for i in range(len(test_data)):
        k1[0] = k1[0] + abs(test_data[i] - s1[i])
        k1[1] = k1[1] + abs(test_data[i] - s2[i])
        k1[2] = k1[2] + abs(test_data[i] - s3[i])
        k1[3] = k1[3] + abs(test_data[i] - s4[i])
        k1[4] = k1[4] + abs(test_data[i] - s5[i])
        k1[5] = k1[5] + abs(test_data[i] - s6[i])
        k1[6] = k1[6] + abs(test_data[i] - s7[i])
        k1[7] = k1[7] + abs(test_data[i] - s8[i])
        k1[8] = k1[8] + abs(test_data[i] - s9[i])

        k2[0] = k2[0] + abs(test_data[i] - t1[i])
        k2[1] = k2[1] + abs(test_data[i] - t2[i])
        k2[2] = k2[2] + abs(test_data[i] - t3[i])
        k2[3] = k2[3] + abs(test_data[i] - t4[i])
        k2[4] = k2[4] + abs(test_data[i] - t5[i])
        k2[5] = k2[5] + abs(test_data[i] - t6[i])
        k2[6] = k2[6] + abs(test_data[i] - t7[i])
        k2[7] = k2[7] + abs(test_data[i] - t8[i])
        k2[8] = k2[8] + abs(test_data[i] - t9[i])


elif knn_flag == 2:

    # 初始化k1，k2，分别为正弦波训练集和三角波训练集的记录数组
    k1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    k2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 使用knn算法计算欧式距离
    for i in range(len(test_data)):
        k1[0] = k1[0] + abs(test_data[i] - s1[i])
        k1[1] = k1[1] + abs(test_data[i] - s2[i])
        k1[2] = k1[2] + abs(test_data[i] - s3[i])
        k1[3] = k1[3] + abs(test_data[i] - s4[i])
        k1[4] = k1[4] + abs(test_data[i] - s5[i])
        k1[5] = k1[5] + abs(test_data[i] - s6[i])
        k1[6] = k1[6] + abs(test_data[i] - s7[i])
        k1[7] = k1[7] + abs(test_data[i] - s8[i])
        k1[8] = k1[8] + abs(test_data[i] - s9[i])

        k2[0] = k2[0] + abs(test_data[i] - t1[i])
        k2[1] = k2[1] + abs(test_data[i] - t2[i])
        k2[2] = k2[2] + abs(test_data[i] - t3[i])
        k2[3] = k2[3] + abs(test_data[i] - t4[i])
        k2[4] = k2[4] + abs(test_data[i] - t5[i])
        k2[5] = k2[5] + abs(test_data[i] - t6[i])
        k2[6] = k2[6] + abs(test_data[i] - t7[i])
        k2[7] = k2[7] + abs(test_data[i] - t8[i])
        k2[8] = k2[8] + abs(test_data[i] - t9[i])

elif knn_flag == 3:

    # 初始化k1，k2，分别为正弦波训练集和三角波训练集的记录数组
    k1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    k2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 使用knn算法计算欧式距离
    for i in range(len(test_data)):
        k1[0] = k1[0] + abs(test_data[i] - s1[i])
        k1[1] = k1[1] + abs(test_data[i] - s2[i])
        k1[2] = k1[2] + abs(test_data[i] - s3[i])
        k1[3] = k1[3] + abs(test_data[i] - s4[i])
        k1[4] = k1[4] + abs(test_data[i] - s5[i])
        k1[5] = k1[5] + abs(test_data[i] - s6[i])
        k1[6] = k1[6] + abs(test_data[i] - s7[i])
        k1[7] = k1[7] + abs(test_data[i] - s8[i])
        k1[8] = k1[8] + abs(test_data[i] - s9[i])
        k1[9] = k1[9] + abs(test_data[i] - s10[i])

        k2[0] = k2[0] + abs(test_data[i] - t1[i])
        k2[1] = k2[1] + abs(test_data[i] - t2[i])
        k2[2] = k2[2] + abs(test_data[i] - t3[i])
        k2[3] = k2[3] + abs(test_data[i] - t4[i])
        k2[4] = k2[4] + abs(test_data[i] - t5[i])
        k2[5] = k2[5] + abs(test_data[i] - t6[i])
        k2[6] = k2[6] + abs(test_data[i] - t7[i])
        k2[7] = k2[7] + abs(test_data[i] - t8[i])
        k2[8] = k2[8] + abs(test_data[i] - t9[i])
        k2[9] = k2[9] + abs(test_data[i] - t10[i])

else:
    print(f"error")
# print(k1)
# print(k2)

# s = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# 临时去零
# for i in s:
#     if k1[i] == 0:
#         del k1[i]
#     if k2[i] == 0:
#         del k2[i]

# 对k1和k2进行排序
for i in range(len(k1) - 1):
    for j in range(len(k1) - i - 1):
        if k1[j] > k1[j + 1]:
            temp = k1[j]
            k1[j] = k1[j + 1]
            k1[j + 1] = temp

for i in range(len(k2) - 1):
    for j in range(len(k2) - i - 1):
        if k2[j] > k2[j + 1]:
            temp = k2[j]
            k2[j] = k2[j + 1]
            k2[j + 1] = temp


def gaussian(dist, sigma=2):
    # 采用Gaussian函数进行不同距离的样本的权重优化，当训练样本与测试样本距离↑，该距离值权重↓。
    # 给更近的邻居分配更大的权重(你离我更近，那我就认为你跟我更相似，就给你分配更大的权重)，而较远的邻居的权重相应地减少，取其加权平均。
    """ Input a distance and return it`s weight"""
    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return weight


# print(k1)
# print(k2)

# 初始化选择矩阵，选择10个最近的训练集作为knn算法的判断阈值
k = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s_count = 0
t_count = 0

# 使用加权knn算法，记录距离两个集合的总权重
for i in range(len(k)):
    if k1[0] < k2[0]:
        k[i] = k1[0]
        del k1[0]
        weight = gaussian(k[i])
        s_count = s_count + weight
    else:
        k[i] = k2[0]
        del k2[0]
        weight = gaussian(k[i])
        t_count = t_count + weight

# 设置异常半径
strange_flag = 1
for i in range(len(k)):
    if k[i] < 5:
        strange_flag = 0

# 根据获得的条件判断信号类型
if 0.4 < abs(s_count / (s_count + t_count) * 100) < 0.6:
    print(f"第1段为异常信号")
elif strange_flag == 1:
    print(f"第1段为异常信号")
elif s_count > t_count:
    percent = s_count / (s_count + t_count) * 100
    if percent >= 70:
        print(f"第1段为正弦波信号")
        print(f"有{percent}%的可能性为正弦波信号")
        percent_list.append(percent + 40)
    else:
        print(f"第1段为异常信号")
elif t_count > s_count:
    percent = t_count / (s_count + t_count) * 100
    if percent >= 70:
        print(f"第1段为三角波信号")
        print(f"有{percent}%的可能性为三角波信号")
        percent_list.append(percent)
    else:
        print(f"第1段为异常信号")
else:
    print(f"error")
# print(k)

if loop_flag == 1:
    test_data = data_y[crossings[0]:-1]

    test_data = np.array(test_data)

    # # 求出测试集的最大值
    # M = abs(np.max(test_data))
    # m = abs(np.min(test_data))

    # 抽取插值
    test_data = resample_array(test_data, 50)

    answer = knn_loop(test_data)
    s_count = answer[0]
    t_count = answer[1]
    strange_flag = answer[2]

    # 根据获得的条件判断信号类型
    if 0.4 < abs(s_count / (s_count + t_count) * 100) < 0.6:
        print(f"第2段为异常信号")
    elif strange_flag == 1:
        print(f"第2段为异常信号")
    elif s_count > t_count:
        percent = s_count / (s_count + t_count) * 100
        if percent >= 0.7:
            print(f"第2段为正弦波信号")
            print(f"有{percent}%的可能性为正弦波信号")
            percent_list.append(percent + 40)
        else:
            print(f"第2段为异常信号")
    elif t_count > s_count:
        percent = t_count / (s_count + t_count) * 100
        if percent >= 0.7:
            print(f"第2段为三角波信号")
            print(f"有{percent}%的可能性为三角波信号")
            percent_list.append(percent)
        else:
            print(f"第2段为异常信号")
    else:
        print(f"error")
    # print(k)

if loop_flag == 2:
    test_data_left = data_y[0:crossings[0]]
    test_data_right = data_y[crossings[1]:-1]
    if len(test_data_left) > len(test_data_right):
        test_data = test_data_left
    else:
        test_data = test_data_right

    test_data = np.array(test_data)

    # # 求出测试集的最大值
    # M = abs(np.max(test_data))
    # m = abs(np.min(test_data))

    # 抽取插值
    test_data = resample_array(test_data, 50)

    answer = knn_loop(test_data)
    s_count = answer[0]
    t_count = answer[1]
    strange_flag = answer[2]

    # 根据获得的条件判断信号类型
    if 0.4 < abs(s_count / (s_count + t_count) * 100) < 0.6:
        print(f"第2段为异常信号")
    elif strange_flag == 1:
        print(f"第2段为异常信号")
    elif s_count > t_count:
        percent = s_count / (s_count + t_count) * 100
        if percent >= 0.7:
            print(f"第2段为正弦波信号")
            print(f"有{percent}%的可能性为正弦波信号")
            percent_list.append(percent + 40)
        else:
            print(f"第2段为异常信号")
    elif t_count > s_count:
        percent = t_count / (s_count + t_count) * 100
        if percent >= 0.7:
            print(f"第2段为三角波信号")
            print(f"有{percent}%的可能性为三角波信号")
            percent_list.append(percent)
        else:
            print(f"第2段为异常信号")
    else:
        print(f"error")
    # print(k)

elif loop_flag == 3:
    for t in range(len(crossings) - 2):

        test_data = data_y[crossings[t + 1]:crossings[t + 2]]

        test_data = np.array(test_data)

        # # 求出测试集的最大值
        # M = abs(np.max(test_data))
        # m = abs(np.min(test_data))

        # 抽取插值
        test_data = resample_array(test_data, 50)

        answer = knn_loop(test_data)
        s_count = answer[0]
        t_count = answer[1]
        strange_flag = answer[2]

        # 根据获得的条件判断信号类型
        if 0.4 < abs(s_count / (s_count + t_count) * 100) < 0.6:
            print(f"第{t + 2}段为异常信号")
        elif strange_flag == 1:
            print(f"第{t + 2}段为异常信号")
        elif s_count > t_count:
            percent = s_count / (s_count + t_count) * 100
            print(f"第{t + 2}段为正弦波信号")
            print(f"有{percent}%的可能性为正弦波信号")
            percent_list.append(percent + 40)
        elif t_count > s_count:
            percent = t_count / (s_count + t_count) * 100
            print(f"第{t + 2}段为三角波信号")
            print(f"有{percent}%的可能性为三角波信号")
            percent_list.append(percent)
        else:
            print(f"error")
        # print(k)

# print(percent_list)
for i in range(len(percent_list) - 1):
    if percent_list[i + 1] - percent_list[0] > 30:
        print("相邻两段信号的识别种类不同，该信号为异常信号")
    elif percent_list[i + 1] - percent_list[0] > 5:
        print("相邻两段信号的识别概率不同，该信号为异常信号")

# # 你的数据
# s = [s1, s2, s3, s4, s5, s6, s7, s8]
#
# # 创建子图
# fig, axs = plt.subplots(2, 4, figsize=(20, 8))  # 改变子图数量和大小以适应您的需求
#
# for i, ax in enumerate(axs.flatten()):
#     ax.plot(s[i])
#     ax.set_title(f'Curve {i + 1}')  # 你可以设置合适的标题
#     ax.set_ylim([-1.1, 1.1])  # 设置y轴范围
# plt.tight_layout()
# plt.show()

x_values = np.linspace(0, len(mix_data)-1, len(mix_data))

# Plot the data
plt.plot(x_values, mix_data)

# Labeling and display

plt.xlabel("X values")
plt.ylabel("Y values")
plt.grid(True)
plt.show()


