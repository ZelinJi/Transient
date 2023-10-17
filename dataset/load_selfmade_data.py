import pandas as pd
import os
import csv

# 创建一个空的DataFrame来存储所有数据
# combined_data = pd.DataFrame()

# 设置CSV文件所在的目录
csv_directory = './selfmake_dataset'

i = 0
# 遍历目录中的CSV文件
with open('existing_file.csv', mode='a', newline='') as file:
    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            i += 1
            if i % 1000 == 0:
                print('Processing file %d...' % i)
            csv_path = os.path.join(csv_directory, filename)

        # 从CSV文件中读取数据
        df = pd.read_csv(csv_path)

        # 从对应的HEA文件中获取标签
        # hea_filename = os.path.splitext(filename)[0] + '.hea'
        # hea_path = os.path.join(csv_directory, hea_filename)

        # with open(hea_path, 'r') as hea_file:
        #     # 从HEA文件中提取标签信息，这取决于您的数据格式
        #     # 这里只是一个示例，您需要根据您的数据格式进行修改
        #     transient_type = hea_file.readline().strip()
        #     id = hea_file.readline().strip()
        #     modulation = hea_file.readline().strip()
        #     SNR = hea_file.readline().strip()
        #     number_of_items = hea_file.readline().strip()
        #     data1 = hea_file.readline().strip()
        #     data2 = hea_file.readline().strip()
        #     data3 = hea_file.readline().strip()
        #     data4 = hea_file.readline().strip()

        # 添加标签列
        # df['Label'] = [transient_type, id, modulation, SNR, number_of_items, data1, data2, data3, data4]


