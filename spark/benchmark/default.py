# -*- encoding: utf-8 -*-
"""
 @Time : 2021/6/15 12:11
 @Author : zspp
 @File : default
 @Software: PyCharm 
"""
import os
import json
import time

if __name__ == "__main__":
    c1 = time.time()
    a = json.load(open('paras.json'), strict=False)
    select_all_component_params = a['select_all_component_params']
    config_params = a['config_params']
    common_params = a['common_params']
    default_component_params = a['default_component_params']

    select_all_py = os.getcwd() + '\select_all\select_all.py'
    default_py = os.getcwd() + '\\default\\default.py'

    select_all_file = os.getcwd() + '\select_all\\result.txt'

    # select_all
    # command1 = 'python %s --select_all_component_params %s --config_params %s --common_params %s' % (
    #     select_all_py, select_all_component_params, config_params, common_params)
    # command1 = command1.replace('\"', '\\"')
    # os.system(command1)

    #default
    command2 = "python {} --select_all_file {} --default_component_params {} --config_params {}".format(
        default_py, select_all_file, default_component_params, config_params)
    command2 = command2.replace('\"', '\\"')
    os.system(command2)


    c2 = time.time()
    print('default算法运行时间:{}s'.format(c2 - c1))
