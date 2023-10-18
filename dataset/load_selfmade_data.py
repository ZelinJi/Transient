import pandas as pd
import os

Folder_Path = r'./selfmake_dataset'  # 要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path = r'..'  # 拼接后要保存的文件路径
SaveFile_Name = r'radar_data.csv'  # 合并后要保存的文件名

# 修改当前工作目录
os.chdir(Folder_Path)
# 将该文件夹下的所有文件名存入一个列表
file_list = os.listdir()

# 读取第一个CSV文件并包含表头
df = pd.read_csv(file_list[0])  # 编码默认UTF-8，若乱码自行更改

# 将读取的第一个CSV文件写入合并后的文件保存
df.to_csv(SaveFile_Path + '/' + SaveFile_Name, encoding="utf_8_sig", index=False)

# 循环遍历列表中各个CSV文件名，并追加到合并后的文件
for i in range(1, len(file_list)):
    if i % 100 == 0:
        print('processing file %d' %i)
    df = pd.read_csv(file_list[i])
    df.to_csv(SaveFile_Path + '/' + SaveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+')



#     transient_type = hea_file.readline().strip()
#     id = hea_file.readline().strip()
#     modulation = hea_file.readline().strip()
#     SNR = hea_file.readline().strip()
#     number_of_items = hea_file.readline().strip()
#     data1 = hea_file.readline().strip()
#     data2 = hea_file.readline().strip()
#     data3 = hea_file.readline().strip()
#     data4 = hea_file.readline().strip()


