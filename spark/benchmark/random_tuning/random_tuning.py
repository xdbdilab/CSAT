# -*- encoding: utf-8 -*-
"""
 @Time : 2021/3/8 11:27
 @Author : zspp
 @File : random_tuning.py
 @Software: PyCharm 
"""
import pandas as pd
import argparse
import os
import time


datadir = os.path.dirname((os.path.abspath(__file__)))
# print("random_tuning_datadir:" + datadir)
time_dict = {}


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='random_tuning')
    parser.add_argument('--random_sample_file', type=str, required=True, help='random_sample_file')
    parser.add_argument('--random_tuning_component_params', type=str, required=True, nargs='+',
                        help='random_tuning_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    args = parser.parse_args()
    return args


def random_tuning(params, config_params):
    min_or_max = params['min_or_max']
    performance = params['performance']
    sample_num = params['sample_num']
    csv_path = params['random_sample_result_path']
    df_samples = pd.read_csv(csv_path)
    # print(df_samples)

    random_tuning_300_config_start_time = time.time()
    print("开始选择最优参数的时间：", time.ctime(random_tuning_300_config_start_time))
    if min_or_max == 'max':
        df_samples = df_samples.sort_values(by=performance, ascending=False)
    else:
        df_samples = df_samples.sort_values(by=performance, ascending=True)

    # df_samples = df_samples.drop(performance, axis=1)
    data = pd.DataFrame(dict(df_samples.iloc[0, :]), index=[0])

    # best_data = {}
    # best_data[performance] = df_samples.pop(performance)
    # best_data['paramList'] = df_samples

    random_tuning_300_config_end_time = time.time()
    print("结束选择最优参数的时间：", time.ctime(random_tuning_300_config_end_time))
    print("{}组参数选择最优参数所耗费的时间：{}s".format(sample_num,
        random_tuning_300_config_end_time - random_tuning_300_config_start_time))  # 以秒为单位
    time_dict['{}组参数选择最优参数所耗费的时间'.format(sample_num)] = random_tuning_300_config_end_time - random_tuning_300_config_start_time

    time_txt = os.path.join(datadir, 'time.txt')
    file_out = open(time_txt, 'w+', encoding='utf-8')
    file_out.write(str(time_dict))
    file_out.close()

    # print(best_data)
    # data(best_data, config_params)

    namecsv = 'best.csv'
    name = os.path.join(datadir, namecsv)
    data.to_csv(name, index=False)

    # name2 = os.path.join(datadir, 'random_tuning_file.txt')
    # file_out = open(name2, 'w+')
    # file_out.write(str(best_data))
    # file_out.close()


def run():
    args = parse_arguments()
    path = args.random_sample_file
    file_read = open(path, 'r',encoding='utf-8')
    params = file_read.read()
    print("random_tuning_shu_ru_nei_rong:" + params)
    params = dict(eval(params))
    # print(params)

    random_tuning_component_params = eval("".join(args.random_tuning_component_params))
    config_params = eval("".join(args.config_params))

    # print("random_tuning_component_params:", random_tuning_component_params)
    # print("config_params:", config_params)

    random_tuning(params, config_params)
    file_read.close()


if __name__ == '__main__':
    run()
