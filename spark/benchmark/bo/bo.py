# -*- encoding: utf-8 -*-
"""
 @Time : 2021/5/22 18:47
 @Author : zspp
 @File : BO
 @Software: PyCharm
"""
from bayes_opt import BayesianOptimization
import functools
import argparse
import os
import math
import getPerformance
import pandas as  pd

# import nfs.colosseum.getPerformance

datadir = os.path.dirname((os.path.abspath(__file__)))


# print("GP_datadir:" + datadir)


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='BO')
    parser.add_argument('--sensitivity_analysis_file', type=str, required=True,
                        help='sensitivity_analysis\'result_filename')
    parser.add_argument('--bo_component_params', type=str, required=True, nargs='+',
                        help='bo_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    args = parser.parse_args()
    return args


def bo(params, bo_component_params, config_params):
    """
    对选择后的参数进行贝叶斯采样

    :param config_params: config_params字典形式
    :param bo_component_params: bo_component_params{"sample_num":5}
    :param params: 包含参数取值,performance,min_or_max以字典存储
    :return: 无返回值,结果保存在hyperopt_sample_file.txt中
    """
    temp = params.copy()
    performance = temp.pop('performance')
    min_or_max = temp.pop('min_or_max')
    selected_params = temp

    sample_num = bo_component_params['sample_num']

    columns = []

    space = dict()
    for key in selected_params.keys():
        columns.append(key)
        if selected_params[key][0] == 'int':
            space[key] = (selected_params[key][1][0]+1, selected_params[key][1][1]-1)
        elif selected_params[key][0] == 'float':
            space[key] = (selected_params[key][1][0]+0.1, selected_params[key][1][1]-0.1)
        elif selected_params[key][0] == 'enum':
            space[key] = (0, 1)

    def getmaxperf(**params):
        get = functools.partial(getPerformance.get_performance, config_params=config_params, performance=performance)
        for key in params:
            if selected_params[key][0] == 'enum' or selected_params[key][0] == 'string':
                params[key] = selected_params[key][1][math.floor(params[key] * (len(selected_params[key][1])-1))]
            elif selected_params[key][0] == 'int':
                params[key] = int(params[key])

        return get(params)

    def getminperf(**params):
        get = functools.partial(getPerformance.get_performance, config_params=config_params, performance=performance)
        for key in params:
            if selected_params[key][0] == 'enum' or selected_params[key][0] == 'string':
                params[key] = selected_params[key][1][math.floor(params[key] * len(selected_params[key][1]))]
            elif selected_params[key][0] == 'int':
                params[key] = int(params[key])
        return -get(params)

    if min_or_max == 'max':
        optimizer = BayesianOptimization(f=getmaxperf, pbounds=space, random_state=1)
        optimizer.maximize(
            init_points=0,
            n_iter=sample_num,
        )

    else:
        optimizer = BayesianOptimization(f=getminperf, pbounds=space, random_state=1)
        optimizer.maximize(
            init_points=0,
            n_iter=sample_num,
        )
    print("BO结束！！！")

    all_data = pd.DataFrame()
    for i, res in enumerate(optimizer.res):
        data = {}
        if min_or_max == 'max':
            data[performance] = res['target']
        else:
            data[performance] = -res['target']

        for conf in res['params']:
            if params[conf][0] == 'enum' or params[conf][0] == 'string':
                data[conf] = params[conf][1][math.floor(res['params'][conf] * (len(params[conf][1])-1))]
            else:
                data[conf] = res['params'][conf]
        all_data = all_data.append(data, ignore_index=True)

    all_data.to_csv(os.path.join(datadir, '{}bo.csv'.format(sample_num)), index=False)

    result = optimizer.max

    best_data = {}
    if min_or_max == 'max':
        best_data[performance] = result['target']
    else:
        best_data[performance] = -result['target']

    for conf in result['params']:
        if params[conf][0] == 'enum' or params[conf][0] == 'string':
            result['params'][conf] = params[conf][1][math.floor(result['params'][conf] * (len(params[conf][1])-1))]

    best_data.update(result['params'])

    data = pd.DataFrame(dict(best_data), index=[0])
    namecsv = 'best.csv'
    name = os.path.join(datadir, namecsv)
    data.to_csv(name, index=False)

    # print(best_data)
    # data(best_data, config_params)

    # name3 = os.path.join(datadir, 'BO_file.txt')
    # file_out = open(name3, 'w+')
    # file_out.write(str(best_data))
    # file_out.close()


def run():
    args = parse_arguments()
    path = args.sensitivity_analysis_file
    file_read = open(path, 'r')
    params = file_read.read()
    print("BO_shu_ru_nei_rong:" + params)
    params = dict(eval(params))

    bo_component_params = eval("".join(args.bo_component_params))
    config_params = eval("".join(args.config_params))

    print("bo_component_params:", bo_component_params)
    print("config_params:", config_params)

    bo(params, bo_component_params, config_params)
    file_read.close()


if __name__ == '__main__':
    run()
