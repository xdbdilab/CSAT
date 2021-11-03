# -*- encoding: utf-8 -*-
"""
 @Time : 2021/6/10 20:27
 @Author : zspp
 @File : bestconfig
 @Software: PyCharm 
"""
import os
import json
import time

if __name__ == "__main__":
    c1 = time.time()
    a = json.load(open('paras.json'), strict=False)
    select_all_component_params = a['select_all_component_params']
    space_search_component_params = a['space_search_component_params']
    config_params = a['config_params']
    common_params = a['common_params']

    select_all_py = os.getcwd() + '\select_all\select_all.py'
    space_search_py = os.getcwd() + '\\space_search\\space_search.py'

    select_all_file = os.getcwd() + '\\select_all\\result.txt'

    # select_all
    # command1 = 'python %s --select_all_component_params %s --config_params %s --common_params %s' % (
    #     select_all_py, select_all_component_params, config_params, common_params)
    # command1 = command1.replace('\"', '\\"')
    # os.system(command1)

    # space_search
    command2 = "python {} --select_all_file {} --space_search_component_params {} --config_params {}".format(
        space_search_py, select_all_file, space_search_component_params, config_params)
    command2 = command2.replace('\"', '\\"')
    os.system(command2)


    c2 = time.time()
    print('bestconfig算法运行时间:{}s'.format(c2 - c1))

