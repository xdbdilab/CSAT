# -*- encoding: utf-8 -*-
"""
 @Time : 2021/5/22 15:44
 @Author : zspp
 @File : sensitivity_analysis.py
 @Software: PyCharm 
"""
import numpy as np
import pandas as  pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sobol_seq import i4_sobol_generate
# from nfs.colosseum.getPerformance import get_performance
import argparse
import os
import uuid
import copy
from getPerformance import get_performance
datadir = os.path.dirname((os.path.abspath(__file__)))
# print('datadir:', datadir)
np.seterr(divide='ignore', invalid='ignore')


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='SA')
    parser.add_argument('--sensitivity_analysis_component_params', type=str, required=True, nargs='+',
                        help='sensitivity_analysis_component_params,dict_style')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    parser.add_argument('--common_params', type=str, required=True, nargs='+', help='common_params')
    args = parser.parse_args()
    return args


def sobel_sample(params, sobel_sample_number, config_params, performance):
    """
    进行sobel_sample
    :param config_params: 参数以及范围
    :param performance: 性能名称
    :param sobel_sample_number:100
    :param params: 包含参数取值,
    :return: 采样数据+列名
    """
    selected_params = params

    sample_num = sobel_sample_number

    sample_index = i4_sobol_generate(len(selected_params), sample_num)

    samples = []
    columns = []
    for _ in range(sample_num):
        samples.append([])

    for key, j in zip(selected_params.keys(), range(len(selected_params))):
        columns.append(key)
        if selected_params[key][0] == 'int':
            for i in range(sample_num):
                value = int(sample_index[i][j] * (selected_params[key][1][1] - selected_params[key][1][0]) + \
                        selected_params[key][1][0])
                samples[i].append(value)
        elif selected_params[key][0] == 'float':
            for i in range(sample_num):
                value = sample_index[i][j] * (selected_params[key][1][1] - selected_params[key][1][0]) + \
                        selected_params[key][1][0]
                samples[i].append(value)
        elif selected_params[key][0] == 'enum':
            for i in range(sample_num):
                value = int(sample_index[i][j] * len(selected_params[key][1]))
                samples[i].append(selected_params[key][1][value])
        else:
            continue
    for i in range(sample_num):
        x = dict(zip(columns, samples[i]))
        y = get_performance(x, config_params, performance)
        # y = i  # 测试用
        samples[i].append(y)
    columns.append(performance)
    print("{}次采样结束！！！".format(sobel_sample_number))
    # print(samples)
    # print(columns)
    df_samples = pd.DataFrame(samples, columns=columns)
    # print(df_samples)

    return df_samples, columns


def data_in(data, selected_params, columns, performance):
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
    #     mapping = dict(zip(enum_index2[enum_name], size_range))
    #     data[enum_name] = data[enum_name].map(mapping)
    return data, data.columns


def scale_data(data):
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(data)

    print("scaled DF >> ", scaled_df)
    return scaled_df


# 一轮SA
def get_sig_params(header, features, target, alpha, model_n_iterations):
    # print("features>>>> ", features)
    # print("target>>>", target)

    sig_conf_all_iterations = []
    # fit an Extra Trees model to the data
    model = ExtraTreesRegressor()
    n_params = int(float(alpha) * len(header))
    n_all_params = len(header)

    ####### build the model 100 time to overcome model randomness ###########
    for x in range(model_n_iterations):
        all_params = [0] * n_all_params
        model.fit(features, target)
        normalized_importance = 100 * (model.feature_importances_ / max(model.feature_importances_))
        # print('normalized_importance:', normalized_importance)

        indices = normalized_importance.argsort()[-n_params:][::-1]
        # print(indices)
        indices = np.array(indices)
        all_params = np.array(all_params)
        all_params[indices] = 1  # set the indices of the seleced params to 1
        # print("all_params >>>> ", all_params)
        sig_conf_all_iterations = np.append(sig_conf_all_iterations, all_params)
    sig_conf_all_iterations = np.reshape(sig_conf_all_iterations, (model_n_iterations, n_all_params))
    sig_conf_all_iterations = np.count_nonzero(sig_conf_all_iterations,
                                               axis=0)  # count the occurances of each param in the sig params over all the interations
    header = np.array(header)
    indices = sig_conf_all_iterations.argsort()[-n_params:][
              ::-1]  # select the params that have the most occurances in the sig params over all the interations
    indices = np.array(indices)
    h_inf_params = header[indices]
    # print(">>>>>>>>>", h_inf_params)
    return h_inf_params


def get_sig_conf_over_iterations(number_of_SA_iter, alpha, params, sobel_sample_number, config_params, performance,
                                 model_n_iterations):
    next_params = {}
    temp_csv = pd.DataFrame()
    for i in range(number_of_SA_iter):
        sample_data, columns = sobel_sample(params, sobel_sample_number, config_params, performance)
        temp_csv =temp_csv.append(sample_data)
        pre_data, columns = data_in(sample_data, params, columns, performance)
        features = pre_data.values
        target = sample_data.iloc[:, -1].values

        sig_conf = get_sig_params(columns, features, target, alpha, model_n_iterations)

        for conf in sig_conf:
            if conf not in params.keys():
                conf = conf.split('_')
                # print(conf)
                conf = "_".join(conf[:-1])
                # print(conf)
            if conf not in params.keys():
                continue
            next_params[conf] = params[conf]
        params = copy.deepcopy(next_params)

    namecsv = '20sample_data.csv'
    name = os.path.join(datadir, namecsv)
    temp_csv.to_csv(name, index=False)
    return next_params


def sensitivity_analysis(all_params, black_list, white_list, performance, min_or_max, config_params, sensitivity_analysis_component_params):
    params = {}
    for each in white_list:
        params[each] = all_params[each]

    for key in all_params.keys():
        if key in black_list or key in white_list:
            continue
        params[key] = all_params[key]

    alpha = sensitivity_analysis_component_params['alpha']
    number_of_SA_iter = sensitivity_analysis_component_params['number_of_SA_iter']
    sobel_sample_number = sensitivity_analysis_component_params['sobel_sample_number']
    model_n_iterations = sensitivity_analysis_component_params['model_n_iterations']
    choosed_config_andrange = get_sig_conf_over_iterations(number_of_SA_iter, alpha, params, sobel_sample_number,
                                                           config_params, performance, model_n_iterations)  # 返回选择的参数

    choosed_config_andrange['performance'] = performance
    choosed_config_andrange['min_or_max'] = min_or_max
    # output_dir = config_params['output_dir']

    name = os.path.join(datadir, 'result.txt')
    with open(name, 'w+') as f:
        f.write(str(choosed_config_andrange))

    name1 = os.path.join(datadir, 'sensitivity_analysis_file.txt')
    with open(name1, 'w+') as f:
        f.write(str(name))


def run():
    args = parse_arguments()

    args.sensitivity_analysis_component_params = eval("".join(args.sensitivity_analysis_component_params))
    args.config_params = eval("".join(args.config_params))
    args.common_params = eval("".join(args.common_params))

    print("sensitivity_analysis_component_params:", args.sensitivity_analysis_component_params)
    # print("config_params:", args.config_params)
    # print("common_params:", args.common_params)

    all_params = (args.common_params['all_params'])
    black_list = (args.common_params['black_list'])
    white_list = (args.common_params['white_list'])
    performance = args.common_params['performance']

    min_or_max = performance[-3:]
    performance = performance[:-4]

    # print("min_or_max:", min_or_max)
    # print("performance:", performance)
    # print("all_params:", all_params)
    # print("black_list:", black_list)
    # print("white_list:", white_list)

    sensitivity_analysis(all_params, black_list, white_list,
       performance, min_or_max, args.config_params, args.sensitivity_analysis_component_params)


if __name__ == '__main__':
    run()

# import os
#
# ###### main #######
# n_iterations = 2
# samples_file = os.path.join(os.getcwd(), 'Iris.csv')
# n_params = 4
# fraction = 0.5
# result_file = os.path.join(os.getcwd(), 'SA_1.txt')
# SA(samples_file, result_file, n_params, fraction)
