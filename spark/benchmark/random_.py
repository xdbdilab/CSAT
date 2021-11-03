# -*- encoding: utf-8 -*-
"""
 @Time : 2021/6/10 15:43
 @Author : zspp
 @File : random
 @Software: PyCharm 
"""
import os
import json
import time

if __name__ == "__main__":
    c1 = time.time()
    a = json.load(open('paras.json'), strict=False)
    select_all_component_params = a['select_all_component_params']
    random_sample_component_params = a['random_sample_component_params']
    random_tuning_component_params = a['random_tuning_component_params']
    config_params = a['config_params']
    common_params = a['common_params']

    select_all_py = os.getcwd() + '\select_all\select_all.py'
    random_sample_py = os.getcwd() + '\\random_sample\\random_sample.py'
    random_tuning_py = os.getcwd() + '\\random_tuning\\random_tuning.py'

    select_all_file = os.getcwd() + '\select_all\\result.txt'
    random_sample_file = os.getcwd() + '\\random_sample\\result.txt'

    # select_all
    # command1 = 'python %s --select_all_component_params %s --config_params %s --common_params %s' % (
    #     select_all_py, select_all_component_params, config_params, common_params)
    # command1 = command1.replace('\"', '\\"')
    # os.system(command1)
    #
    # # random_sample
    # command2 = "python {} --select_all_file {} --random_sample_component_params {} --config_params {}".format(
    #     random_sample_py, select_all_file, random_sample_component_params, config_params)
    # command2 = command2.replace('\"', '\\"')
    # os.system(command2)

    # random_tuning
    command3 = 'python {} --random_sample_file {} --random_tuning_component_params {} --config_params {}'.format(
        random_tuning_py, random_sample_file, random_tuning_component_params, config_params)
    command3 = command3.replace('\"', '\\"')
    print(command3)

    os.system(command3)
    print(command3)

    c2 = time.time()
    print('random算法选择最优参数的运行时间(random_tuning):{}s'.format(c2 - c1))
