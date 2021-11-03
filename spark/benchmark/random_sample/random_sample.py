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
# print("随机采样文件夹:" + datadir)
time_dict = {}


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='random_sample')
    parser.add_argument('--select_all_file', type=str, required=True, help='select_all\'result_filename')
    parser.add_argument('--random_sample_component_params', type=str, required=True, nargs='+',
                        help='random_sample_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    args = parser.parse_args()
    return args


def random_sample(params, random_sample_component_params, config_params):
    """
    对选择后的参数进行随机采样
    :param config_params: config_params字典形式
    :param random_sample_component_params: random_sample_component_params字典形式{"sample_num":5}
    :param params: 包含参数取值,performance,min_or_max以字典存储
    :return: 无返回值,结果保存在random_sample_file.txt中
    """
    temp = params.copy()
    performance = temp.pop('performance')
    min_or_max = temp.pop('min_or_max')
    selected_params = temp
    #
    sample_num = random_sample_component_params['sample_num']
    #
    # # 随机采样开始
    # random_sample_300_config_start_time = time.time()
    # print("开始产生{}组参数样本的时间：".format(sample_num), time.ctime(random_sample_300_config_start_time))
    # samples = []
    columns = []
    # for _ in range(sample_num):
    #     samples.append([])
    for key in selected_params.keys():
        columns.append(key)
    #     if selected_params[key][0] == 'int':
    #         for i in range(sample_num):
    #             samples[i].append(random.randint(selected_params[key][1][0] + 1, selected_params[key][1][1]))
    #     elif selected_params[key][0] == 'float':
    #         for i in range(sample_num):
    #             samples[i].append(random.uniform(selected_params[key][1][0] + 0.1, selected_params[key][1][1]))
    #     elif selected_params[key][0] == 'enum':
    #         for i in range(sample_num):
    #             samples[i].append(random.choice(selected_params[key][1]))
    #     else:
    #         continue
    #
    # # 随机采样300组结束
    # random_sample_300_config_end_time = time.time()
    # print("结束产生{}组参数样本的时间：".format(sample_num), time.ctime(random_sample_300_config_start_time))
    # print("随机采样{}组参数样本所耗费的时间：{}s".format(sample_num,
    #                                      random_sample_300_config_end_time - random_sample_300_config_start_time))  # 以秒为单位
    # time_dict['随机采样{}组参数样本所耗费的时间'.format(
    #     sample_num)] = random_sample_300_config_end_time - random_sample_300_config_start_time
    #
    # # 300组参数样本获取性能值的开始时间
    # performance_get_300_config_start_time = time.time()
    # print("{}组参数样本获取性能值的开始时间：".format(sample_num), time.ctime(performance_get_300_config_start_time))
    # for i in range(sample_num):
    #     x = dict(zip(columns, samples[i]))
    #     y = get_performance(x, config_params, performance)
    #     # y = random.random() #测试用
    #     samples[i].append(y)
    #     print("第{}条数据！！！\n".format(i))
    columns.append(performance)
    # # print("采样结束！！！")
    # performance_get_300_config_end_time = time.time()
    # print("{}组参数样本获取性能值的结束时间：".format(sample_num), time.ctime(performance_get_300_config_end_time))
    # print("{}组参数样本获取性能值的耗费时间：{}s".format(sample_num,
    #                                      performance_get_300_config_end_time - performance_get_300_config_start_time))
    # time_dict['{}组参数样本获取性能值的耗费时间'.format(
    #     sample_num)] = performance_get_300_config_end_time - performance_get_300_config_start_time
    # # print(time_dict)
    # time_txt = os.path.join(datadir, 'time.txt')
    # file_out = open(time_txt, 'w+', encoding='utf-8')
    # file_out.write(str(time_dict))
    # file_out.close()

    # df_samples = pd.DataFrame(samples, columns=columns)
    #
    # # output_dir = config_params['output_dir']
    namecsv = 'random{}.csv'.format(sample_num)
    name = os.path.join(datadir, namecsv)
    # df_samples.to_csv(name, index=False)

    params['random_sample_result_path'] = name
    params['columns'] = columns
    params['sample_num'] = sample_num

    name2 = os.path.join(datadir, "result.txt")
    file_out = open(name2, 'w+', encoding='utf-8')
    file_out.write(str(params))
    file_out.close()
    # print("随机采样模块传入下个模块的内容:" + str(params))

    name3 = os.path.join(datadir, 'random_sample_file.txt')
    file_out = open(name3, 'w+', encoding='utf-8')
    file_out.write(str(name2))
    file_out.close()


def run():
    args = parse_arguments()
    path = args.select_all_file
    file_read = open(path, 'r')
    params = file_read.read()
    # print("随机采样来自上一模块的输入内容:" + params)
    params = dict(eval(params))

    random_sample_component_params = eval("".join(args.random_sample_component_params))
    config_params = eval("".join(args.config_params))

    # print("随机采样模块参数:", random_sample_component_params)
    # print("config_params:", config_params)

    random_sample(params, random_sample_component_params, config_params)
    file_read.close()


if __name__ == '__main__':
    run()

    # 测试
    # random_sample({'sort_buffer_size': ['int', [131072, 16777216], 262144], 'join_buffer_size': ['int', [131072, 16777216], 262144], 'innodb_buffer_pool_size': ['int', [268435456, 3221225472], 268435456], 'innodb_thread_concurrency': ['int', [0, 50], 0], 'innodb_adaptive_hash_index': ['enum', ['OFF', 'ON'], 'ON'], 'innodb_flush_log_at_trx_commit': ['enum', [0, 1, 2], 1], 'tmp_table_size': ['int', [131072, 67108864], 16777216], 'thread_cache_size': ['int', [0, 50], 0], 'read_rnd_buffer_size': ['int', [0, 134217728], 262144], 'max_heap_table_size': ['int', [262144, 67108864], 16777216], 'sutName': 'mysql', 'envId': 13, 'sample_num': 5, 'performance': 'throughput', 'min_or_max': 'max'},1,1)
