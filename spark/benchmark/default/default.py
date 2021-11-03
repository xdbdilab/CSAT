# -*- encoding: utf-8 -*-
"""
@File    : random_sample.py
@Time    : 2021/3/8 9:13
@Author  : zspp
@Software: PyCharm
"""
import random
import uuid
import pandas as pd
import argparse
import os
import time
# from nfs.colosseum.getPerformance import get_performance
from getPerformance import get_performance

datadir = os.path.dirname((os.path.abspath(__file__)))


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='random_sample')
    parser.add_argument('--select_all_file', type=str, required=True, help='select_all\'result_filename')
    parser.add_argument('--default_component_params', type=str, required=True, nargs='+',
                        help='default_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    args = parser.parse_args()
    return args


def default(params, default_component_params, config_params):
    """
    对选择后的参数进行随机采样
    :param config_params: config_params字典形式
    :param default_component_params:
    :param params: 包含参数取值,performance,min_or_max以字典存储
    :return: 无返回值,结果保存在random_sample_file.txt中
    """
    temp = params.copy()
    performance = temp.pop('performance')
    min_or_max = temp.pop('min_or_max')
    selected_params = temp


    samples = []
    columns = []

    for key in selected_params.keys():
        columns.append(key)
        samples.append(selected_params[key][2])

    x = dict(zip(columns, samples))
    print(x)
    y = get_performance(x, config_params, performance)
    samples.append(y)
    columns.append(performance)


    df_samples = pd.DataFrame([samples], columns=columns)

    namecsv = 'default.csv'
    name = os.path.join(datadir, namecsv)
    df_samples.to_csv(name, index=False)

    name2 = os.path.join(datadir, "result.txt")


    name3 = os.path.join(datadir, 'default_file.txt')
    file_out = open(name3, 'w+', encoding='utf-8')
    file_out.write(str(name2))
    file_out.close()


def run():
    args = parse_arguments()
    path = args.select_all_file
    file_read = open(path, 'r')
    params = file_read.read()
    params = dict(eval(params))

    default_component_params = eval("".join(args.default_component_params))
    config_params = eval("".join(args.config_params))


    default(params, default_component_params, config_params)
    file_read.close()


if __name__ == '__main__':
    run()


