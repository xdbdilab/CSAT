import random
import time

import numpy as np
import math
import pandas as pd
import argparse
import os
import copy
import pickle
import joblib
from getPerformance import get_performance

# from nfs.colosseum.report import *
# from nfs.colosseum.getPerformance import get_performance

datadir = os.path.dirname((os.path.abspath(__file__)))
print("RS_datadir:" + datadir)
print()
time_dict = {}


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='RS')
    parser.add_argument('--gaussian_process_file', type=str, required=True, help='gaussian_process_file')
    parser.add_argument('--random_search_component_params', type=str, required=True, nargs='+',
                        help='random_search_component_params')
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


def random_search(columns, selected_params, performance, min_or_max, config_params, model, std,
                  random_search_component_params):
    # 最优配置搜索时间
    # start = time.time()
    model = model
    scaler = std
    # 随机采样3000组样本（只有参数，无性能值），用模型预测性能值，最优的一组配置为最终结果
    sample_num = random_search_component_params['sample_num']
    samples, columns = [], []
    for _ in range(sample_num):
        samples.append([])
    for key in selected_params.keys():
        columns.append(key)
        if selected_params[key][0] == 'int':
            for i in range(sample_num):
                samples[i].append(random.randint(selected_params[key][1][0]+1, selected_params[key][1][1]))
        elif selected_params[key][0] == 'float':
            for i in range(sample_num):
                samples[i].append(random.uniform(selected_params[key][1][0]+0.1, selected_params[key][1][1]))
        elif selected_params[key][0] == 'enum':
            for i in range(sample_num):
                samples[i].append(random.choice(selected_params[key][1]))
        else:
            continue
    columns.append(performance)


    # 随机采样结果加performance一列
    for i in range(len(samples)):
        samples[i].append(0)
    samples = pd.DataFrame(samples, columns=columns)

    # print(samples)


    # 预测性能值
    test_samples, enum_number, enum_book = data_in(samples, selected_params, columns, performance)
    test_samples = scaler.transform(test_samples.values)
    y_samples = model.predict(test_samples)

    samples[performance] = y_samples


    if min_or_max == 'max':
        res = samples.sort_values(by=performance, ascending=False)
    else:
        res = samples.sort_values(by=performance, ascending=True)


    res = res.iloc[:1, :]

    res.reset_index()
    res = res.to_dict('records')[0]
    res.pop(performance)

    res[performance] = get_performance(res, config_params, performance)

    print('最优配置：', res)

    res = pd.DataFrame(res, index=[0], columns=columns)
    fileName = 'best.csv'
    name = os.path.join(datadir, fileName)
    res.to_csv(name, index=False)

    # 最优配置输出到文件result.txt
    name1 = os.path.join(datadir, 'result.txt')
    file_out = open(name1, 'w+',encoding='utf-8')
    file_out.write(str(name))
    file_out.close()

    name2 = os.path.join(datadir, 'random_search_file.txt')
    file_out = open(name2, 'w+',encoding='utf-8')
    file_out.write(name1)
    file_out.close()

    # end = time.time()
    # print('最优配置搜索开始时间：', time.ctime(start))
    # print('最优配置搜索结束时间：', time.ctime(end))
    # print('最优配置搜索时间(s): ', end - start)
    # print()
    #
    # time_dict['最优配置搜索时间'] = end - start
    # time_txt = os.path.join(datadir, 'time.txt')
    # file_out = open(time_txt, 'w+', encoding='utf-8')
    # file_out.write(str(time_dict))
    # file_out.close()


def run():
    args = parse_arguments()
    path = args.gaussian_process_file
    file_read = open(path, 'r', encoding='utf-8')
    params = file_read.read()
    print("RS_shu_ru_nei_rong:" + params)
    params = dict(eval(params))
    # print(params)

    temp = params.copy()
    performance = temp.pop('performance')
    columns = temp.pop('features')
    path = temp.pop('lasso_select_result_path')
    min_or_max = temp.pop('min_or_max')
    std_path = temp.pop('std_path')
    model_path = temp.pop('model_path')

    selected_params = temp

    random_search_component_params = eval("".join(args.random_search_component_params))
    config_params = eval("".join(args.config_params))

    print("random_search_component_params:", random_search_component_params)
    print("config_params:", config_params)
    print()
    start = time.time()

    model = joblib.load(model_path)

    std = pickle.load(open(std_path, 'rb'))

    random_search(columns, selected_params, performance, min_or_max,
                  config_params, model, std, random_search_component_params)

    file_read.close()

    end = time.time()
    print('最优配置搜索开始时间：', time.ctime(start))
    print('最优配置搜索结束时间：', time.ctime(end))
    print('最优配置搜索时间(s): ', end - start)
    print()

    time_dict['最优配置搜索时间'] = end - start
    time_txt = os.path.join(datadir, 'time.txt')
    file_out = open(time_txt, 'w+', encoding='utf-8')
    file_out.write(str(time_dict))
    file_out.close()


if __name__ == '__main__':
    run()
