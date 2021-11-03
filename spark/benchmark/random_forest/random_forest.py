# -*- encoding: utf-8 -*-
"""
 @Time : 2021/3/9 10:38
 @Author : zspp
 @File : random_forest
 @Software: PyCharm 
"""
import uuid
import pandas as pd
import argparse
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

datadir = os.path.dirname((os.path.abspath(__file__)))


# print("random_forest:" + datadir)


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='random_forest')
    parser.add_argument('--random_sample_file', type=str, required=True, help='random_sample_file')
    parser.add_argument('--random_forest_component_params', type=str, required=True, nargs='+',
                        help='random_forest_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')

    args = parser.parse_args()
    return args


def data_in(data, selected_params, columns, performance, min_or_max):  # 每次只传入一组参数，所以与acgan、bogan的数据预处理不一样
    # print(data)
    # if min_or_max == 'max':
    #     data = data.sort_values(by=performance, ascending=False)
    # else:
    #     data = data.sort_values(by=performance, ascending=True)
    data = data.drop(performance, 1)
    # print(data)
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
        # print(selected_params)
        new_data = pd.DataFrame({c: selected_params[c][1]})  # 添加几行，为了更好全面编码
        data = data.append(new_data, ignore_index=True)
        # print(data)
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
    #     mapping = dict(zip(enum_index2[enum_name], size_range))
    #     data[enum_name] = data[enum_name].map(mapping)
    return data


def random_forest(train_data, selected_params, columns, performance, min_or_max, config_params, params):
    # 准备训练数据

    X = data_in(train_data, selected_params, columns, performance, min_or_max)
    print(X)
    print(X.columns)
    point = 80
    X_train = X.iloc[:point, :].values
    X_test = X.iloc[point:, :]
    y_train = train_data.iloc[:point, -1].values
    y_test = train_data.iloc[point:, -1].values

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train.astype(float))
    X_test = sc.transform(X_test.astype(float))
    # 训练随机森林解决回归问题
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pre = regressor.predict(X_test)
    # print(regressor.criterion)
    print("预测值:", y_pre)
    print("真实值:", y_test)
    MRE = np.average(np.abs(y_test - y_pre) / y_test, axis=0)
    print("MRE:", MRE)
    # output_dir = config_params['output_dir']
    # name1 = os.path.join(datadir, 'train_model.pkl')
    # # print(name1)
    # joblib.dump(regressor, name1)
    #
    # name2 = os.path.join(datadir, "scaler.pkl")
    # pickle.dump(sc, open(name2, 'wb'))
    #
    # params['model_path'] = name1
    # params['std_path'] = name2
    #
    # name3 = os.path.join(datadir, "result.txt")
    # file_out = open(name3, 'w+', encoding='utf-8')
    # file_out.write(str(params))
    # file_out.close()
    #
    # name4 = os.path.join(datadir, 'random_forest_file.txt')
    # file_out = open(name4, 'w+', encoding='utf-8')
    # file_out.write(str(name3))
    # file_out.close()


def run():
    args = parse_arguments()
    path = args.random_sample_file
    file_read = open(path, 'r', encoding='utf-8')
    params = file_read.read()
    # print("random_forest_shu_ru_nei_rong:" + params)
    params = dict(eval(params))
    # print(params)

    temp = params.copy()
    performance = temp.pop('performance')
    columns = temp.pop('columns')
    path = temp.pop('random_sample_result_path')
    min_or_max = temp.pop('min_or_max')
    sample_num = temp.pop('sample_num')
    selected_params = temp

    random_forest_component_params = eval("".join(args.random_forest_component_params))
    config_params = eval("".join(args.config_params))

    # print("random_forest_component_params:", random_forest_component_params)
    # print("config_params:", config_params)

    train_data = pd.read_csv(path)

    train_data = train_data.replace(True, "true")
    train_data = train_data.replace(False, "false")

    random_forest(train_data, selected_params, columns, performance, min_or_max, config_params, params)

    file_read.close()


if __name__ == '__main__':
    run()
