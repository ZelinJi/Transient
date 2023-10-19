import pandas as pd
import os
import pickle

def read_label (hea_file):
    hea_path = os.path.join(hea_file)
    with open(hea_path, 'r') as hea_file:
        # | Signal class | SNR | Signal length | Carrier frequency | LFM Bandwidth OR symbol rate | Frequency excursion |
        transient_type = hea_file.readline().strip()
        id = hea_file.readline().strip()
        modulation = hea_file.readline().strip()
        SNR = hea_file.readline().strip()
        number_of_items = hea_file.readline().strip()
        carrier_freq = hea_file.readline().strip()
        symbol_rate = hea_file.readline().strip()
        xFSK = hea_file.readline().strip()
        data4 = hea_file.readline().strip()
        if transient_type == 'frequency converte':
            return (transient_type, xFSK, SNR)
        return (transient_type, modulation, SNR)

Folder_Path = r'./selfmake_dataset/'  # 要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Name = r'radar_data.pkl'  # 合并后要保存的文件名
SaveFile_Path = r'../'  # 拼接后要保存的文件路径

# 修改当前工作目录
os.chdir(Folder_Path)
# 将该文件夹下的所有文件名存入一个列表
file_list = [f for f in os.listdir() if f.endswith('.csv')]

sort_num_list = []
for file in file_list:
    sort_num_list.append(int(file.split('S')[1].split('.csv')[0])) #去掉前面的字符串和下划线以及后缀，只留下数字并转换为整数方便后面排序
    sort_num_list.sort() #然后再重新排序
print("finishing sorting nums...")

data = {}
# 读取第一个CSV文件并包含表头

for i in range (459):
    df = []
    label = read_label('S%d.hea' % sort_num_list[i*100])
    print('Processing %s %s %s' % label)
    for j in range (100):
        df.append(pd.read_csv('S%d.csv' %sort_num_list[0]).values)
    data[label] = df

with open(os.path.join(SaveFile_Path, SaveFile_Name),'wb') as f:
    pickle.dump(data, f)


# 将读取的第一个CSV文件写入合并后的文件保存
# df.to_csv(SaveFile_Path + '/' + SaveFile_Name, encoding="latin-1", index=False)

# 循环遍历列表中各个CSV文件名，并追加到合并后的文件
# for i in range(1, len(sort_num_list)):
#     df = pd.read_csv('S%d.csv' %sort_num_list[i])
#     if i % 100 == 0:
#         print('processing file %d' %i)
#     df.to_csv(SaveFile_Path + '/' + SaveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+')


