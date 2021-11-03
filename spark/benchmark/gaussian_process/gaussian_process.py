import pickle
import uuid

import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
import random
import argparse
import os
import numpy as np
import time

#from nfs.colosseum.report import *

datadir = os.path.dirname((os.path.abspath(__file__)))
print("gaussian_process_datadir:" + datadir)
print()
time_dict = {}


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='gaussian_process')
    parser.add_argument('--lasso_select_file', type=str, required=True, help='lasso_select_file')
    parser.add_argument('--gaussian_process_component_params', type=str, required=True, nargs='+',
                        help='gaussian_process_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    args = parser.parse_args()
    return args


def data_in(data, selected_params, columns, performance):
    data = data.drop(performance, 1)

    ##首先处理没有优先关系的enum参数，即string参数
    char = []  # enum的列名称
    enum_index = []  # enum的原始列索引
    for name in columns[:-1]:
        if selected_params[name][0] == 'string' or selected_params[name][0] == 'enum':
            char.append(name)
            enum_index.append(columns.index(name))

    enum_number = []  # 每个enum参数对应的独热编码的长度
    enum_book = {}  # 每个enum参数的内容，字典形式存储
    m = 0
    for c in char:
        i = enum_index[m]

        new_data = pd.DataFrame({c: selected_params[c][1]})  # 添加几行，为了更好全面编码
        data = data.append(new_data, ignore_index=True)

        enum_book[c] = list(pd.get_dummies(data[c]).columns)
        enum_data = pd.get_dummies(data[c], prefix=c)  # 独热编码后的变量

        data = data.drop(c, 1)

        enum_list = list(enum_data.columns)
        enum_number.append(len(enum_list))

        for k in range(len(enum_list)):
            data.insert(i + k, enum_list[k], enum_data[enum_list[k]])  # 将向量移动到原来枚举值的位置
        m = m + 1
        enum_index = [j + len(enum_data.columns) - 1 for j in enum_index]  # 更新enum_index

        data.drop(data.index[-len(selected_params[c][1]):], inplace=True)  # 删除前3行
        # print(enum_index)
    # print(enum_number)
    # print(data)
    # print(enum_book)

    ##接着处理有优先关系的参数
    # char2 = []
    # enum_index2 = {}
    # for name in columns[:-1]:
    #     if selected_params[name][0] == 'enum':
    #         char2.append(name)
    #         enum_index2[name] = selected_params[name][1]
    # for enum_name in char2:
    #     size_range = list(range(len(enum_index2[enum_name])))
    #     mapping = dict(zip(enum_index2[enum_name],size_range))
    #     data[enum_name] = data[enum_name].map(mapping)
    return data, enum_number, enum_book


def data_out(data, selected_params, columns, enum_number, enum_book):
    if data.empty:
        return pd.DataFrame(columns=columns[:-1])
    data.columns = [i for i in range(len(data.columns))]
    hang = data.iloc[:, 0].size

    # 将data的数据变为符合的类型格式
    m = 0  # 更改columns之后的columns的索引
    index_enum_number = 0  # enum_number的序号索引
    enum_list = []  # enum的列名称
    for name in columns[:-1]:
        if selected_params[name][0] == 'int':
            data[m] = data[m].astype(np.int64)
            m = m + 1
        elif selected_params[name][0] == 'float' or selected_params[name][0] == 'double':
            m = m + 1
            continue
        # elif selected_params[name][0] =='enum':
        #     string_map = dict(zip(list(range(len(selected_params[name][1]))), selected_params[name][1]))
        #     data[[m]] = data[[m]].astype(np.int64)
        #     data[m] = data[m].map(string_map)
        #     m = m + 1
        else:
            enum_list.append(name)
            for i in range(enum_number[index_enum_number]):
                data[m + i] = data[m + i].round().astype(np.int64)
            m = m + enum_number[index_enum_number]
            index_enum_number = index_enum_number + 1
    # print(data)

    # 在数据尾部加入enum的原始列名称
    for i in enum_list:
        data_temp = pd.DataFrame(columns=[i])
        data = pd.concat([data, data_temp], 1)

    for index in range(hang):
        # 将向量转换成枚举值
        # print(data.iloc[[index]].values[0])
        for k in range(len(enum_list)):

            name_index = columns.index(enum_list[k])  # enum变量的原始列索引
            if k != 0:
                for a in range(k):
                    name_index = name_index + enum_number[a] - 1  # 变为enum改变后的索引
            # print(name_index)

            flag = True
            true_index = False
            first_index = -1
            for i in range(enum_number[k]):

                for j in range(i, enum_number[k]):
                    if i == j:

                        if round(data.loc[[index]].values[0][name_index + j]) == 1:
                            first_index = j
                            continue
                        else:
                            if j == enum_number[k] - 1:
                                flag = False
                            break
                    else:
                        if round(data.loc[[index]].values[0][name_index + j]) == 0:

                            if j == enum_number[k] - 1:
                                true_index = True
                                break
                            else:
                                continue
                        else:
                            flag = False
                            break

                if flag == False:
                    break
                if true_index == True:
                    break
            if flag == False:
                data.drop([index], inplace=True)
                break
            if (first_index + 1) != 0 and flag == True:
                data.loc[index, enum_list[k]] = enum_book[enum_list[k]][first_index]
    # print(data)

    # 删去枚举值所在的列
    number = 0
    for i in range(len(enum_list)):
        name_index = number + columns.index(enum_list[i])

        for j in range(enum_number[i]):
            data = data.drop([name_index + j], 1)

        number = number + enum_number[i] - 1
    # print(data)

    # 移动最后含枚举值的几列,对列进行重命名
    for i in range(len(enum_list)):
        orgin_index = columns.index(enum_list[i])
        data.insert(orgin_index, enum_list[i], data.pop(enum_list[i]))
    # print(data)

    col = {}
    for (key, value) in zip(list(data.columns), columns[:-1]):
        col[key] = value
    # print(col)
    data.rename(columns=col, inplace=True)
    # print(data)
    return data


# 此处我默认params可以含有参数的类型、范围等信息，如果不包含，或许可以添加另一个参数，读取随机采样模块中的配置文件
def gaussian_process(params, config_params, gaussian_process_component_params):
    temp = params.copy()
    performance = temp.pop('performance')
    columns = temp.pop('features')
    csv_path = temp.pop('lasso_select_result_path')
    min_or_max = temp.pop('min_or_max')
    selected_params = temp

    n_restarts_optimizer = gaussian_process_component_params['n_restarts_optimizer']
    alpha = gaussian_process_component_params['alpha']
    random_state = gaussian_process_component_params['random_state']

    # 计算模型训练时间
    start = time.time()
    df_samples = pd.read_csv(csv_path)
    df_samples = df_samples.replace(True, "true")
    df_samples = df_samples.replace(False, "false")

    x_data, _, _ = data_in(df_samples, selected_params, columns, performance)
    # print(x_train)

    # N(0, 1) 标准化
    scaler = preprocessing.StandardScaler()
    point = 80
    x_train, y_train = x_data.iloc[:point, :].values, df_samples.iloc[:point, -1].values
    x_test, y_test = x_data.iloc[point:, :].values, df_samples.iloc[point:, -1].values
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # 高斯过程回归模型
    model = GaussianProcessRegressor(n_restarts_optimizer=n_restarts_optimizer, alpha=alpha, random_state=random_state)
    # 用随机采样的所有数据进行训练
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print("预测值:", y_pre)
    print("真实值:", y_test)
    MRE = np.average(np.abs(y_test - y_pre) / y_test, axis=0)
    print("MRE:", MRE)
    # end = time.time()
    # print('gpr模型开始训练时间：', time.ctime(start))
    # print('gpr模型结束训练时间：', time.ctime(end))
    # print('gpr模型训练时间(s)：', end - start)
    # time_dict['gpr模型训练时间'] = end - start
    # print()
    #
    # time_txt = os.path.join(datadir, 'time.txt')
    # file_out = open(time_txt, 'w+', encoding='utf-8')
    # file_out.write(str(time_dict))
    # file_out.close()
    #
    # name1 = os.path.join(datadir, 'train_model.m')
    # joblib.dump(model, name1)
    #
    # name2 = os.path.join(datadir, "scaler.pkl")
    # pickle.dump(scaler, open(name2, 'wb'))
    #
    # params['model_path'] = name1
    # params['std_path'] = name2
    #
    # name3 = os.path.join(datadir, "result.txt")
    # file_out = open(name3, 'w+', encoding='utf-8')
    # file_out.write(str(params))
    # file_out.close()
    #
    # name4 = os.path.join(datadir, 'gaussian_process_file.txt')
    # file_out = open(name4, 'w+', encoding='utf-8')
    # file_out.write(str(name3))
    # file_out.close()


def run():
    args = parse_arguments()
    path = args.lasso_select_file
    file_read = open(path, 'r', encoding='utf-8')
    params = file_read.read()
    print("gaussian_process_shu_ru_nei_rong:" + params)
    params = dict(eval(params))
    # print(params)

    gaussian_process_component_params = eval("".join(args.gaussian_process_component_params))
    config_params = eval("".join(args.config_params))

    print("gaussian_process_component_params:", gaussian_process_component_params)
    print("config_params:", config_params)
    print()

    gaussian_process(params, config_params, gaussian_process_component_params)
    file_read.close()


if __name__ == '__main__':
    run()
