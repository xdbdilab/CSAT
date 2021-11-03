# -*- encoding: utf-8 -*-
"""
 @Time : 2021/6/10 17:14
 @Author : zspp
 @File : actgan
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
    actgan_component_params = a['actgan_component_params']
    config_params = a['config_params']
    common_params = a['common_params']

    select_all_py = os.getcwd() + '\select_all\select_all.py'
    random_sample_py = os.getcwd() + '\\random_sample\\random_sample.py'
    actgan_py = os.getcwd() + '\\actgan\\actgan.py'

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

    # actgan
    command3 = "python {} --random_sample_file {} --actgan_component_params {} --config_params {}".format(actgan_py,
        random_sample_file, actgan_component_params, config_params)
    command3 = command3.replace('\"', '\\"')
    os.system(command3)

    c2 = time.time()
    print('actgan模块运行时间:{}s'.format(c2 - c1))
