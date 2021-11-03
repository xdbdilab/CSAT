# -*- encoding: utf-8 -*-
"""
 @Time : 2021/6/10 20:27
 @Author : zspp
 @File : rfhoc
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
    random_forest_component_params = a['random_forest_component_params']
    genetic_algorithm_component_params = a['genetic_algorithm_component_params']
    config_params = a['config_params']
    common_params = a['common_params']

    common_params = json.dumps(common_params)
    select_all_py = os.getcwd() + '\\select_all\\select_all.py'
    random_sample_py = os.getcwd() + '\\random_sample\\random_sample.py'
    random_forest_py = os.getcwd() + '\\random_forest\\random_forest.py'
    genetic_algorithm_py = os.getcwd() + '\\genetic_algorithm\\genetic_algorithm.py'

    select_all_file = os.getcwd() + '\\select_all\\result.txt'
    random_sample_file = os.getcwd() + '\\random_sample\\result.txt'
    random_forest_file = os.getcwd() + '\\random_forest\\result.txt'

    # select_all
    command1 = 'python %s --select_all_component_params %s --config_params %s --common_params %r' % (
        select_all_py, select_all_component_params, config_params, common_params)

    command1 = command1.replace('\"', '\\"')
    command1 = command1.replace('\'', '\"')
    os.system(command1)


    c2 = time.time()

    # random_sample
    command2 = "python {} --select_all_file {} --random_sample_component_params {} --config_params {}".format(
        random_sample_py, select_all_file, random_sample_component_params, config_params)
    command2 = command2.replace('\"', '\\"')
    os.system(command2)

    c3 = time.time()

    # random_forest
    command3 = "python {} --random_sample_file {} --random_forest_component_params {} --config_params {}".format(
        random_forest_py,
        random_sample_file,
        random_forest_component_params,
        config_params)
    command3 = command3.replace('\"', '\\"')
    os.system(command3)
    #
    # c4 = time.time()
    # time_dict = {}
    # time_dict['构建随机森林模型需要的时间'] = c4 - c3
    # time_txt = os.path.join('./random_forest/', 'time.txt')
    # file_out = open(time_txt, 'w+', encoding='utf-8')
    # file_out.write(str(time_dict))
    # file_out.close()

    # ga
    command4 = "python {} --random_forest_file {} --genetic_algorithm_component_params {} --config_params {}".format(
        genetic_algorithm_py,
        random_forest_file,
        genetic_algorithm_component_params,
        config_params)

    command4 = command4.replace('\"', '\\"')
    #os.system(command4)

    # c5 = time.time()
    # print('rfhoc算法运行时间:{}s'.format(c5 - c1))
    #
    # time_dict2 = {}
    # time_dict2['GA搜索所需时间'] = c5 - c4
    #
    #
    # time_txt = os.path.join('./genetic_algorithm/', 'time.txt')
    # file_out = open(time_txt, 'w+', encoding='utf-8')
    # file_out.write(str(time_dict2))
    # file_out.close()

