# -*- encoding: utf-8 -*-
"""
 @Time : 2021/6/10 18:57
 @Author : zspp
 @File : SA_bo
 @Software: PyCharm 
"""
import os
import json
import time

if __name__ == "__main__":
    c1 = time.time()
    a = json.load(open('paras.json'), strict=False)
    sensitivity_analysis_component_params = a['sensitivity_analysis_component_params']
    bo_component_params = a['bo_component_params']
    config_params = a['config_params']
    common_params = a['common_params']
    common_params = json.dumps(common_params)
    sensitivity_analysis_component_params = json.dumps(sensitivity_analysis_component_params)

    bo_py = os.getcwd() + '\\bo\\bo.py'
    sensitivity_analysis_py = os.getcwd() + '\\sensitivity_analysis\\sensitivity_analysis.py'

    sensitivity_analysis_file = os.getcwd() + '\\sensitivity_analysis\\result.txt'

    # sa
    command1 = "python %s --common_params %r --sensitivity_analysis_component_params %s --config_params %s" % (
        sensitivity_analysis_py,
        common_params,
        sensitivity_analysis_component_params,
        config_params
    )
    command1 = command1.replace('\"', '\\"')
    command1 = command1.replace('\'', '\"')
    os.system(command1)

    c2 = time.time()
    # bo
    command2 = "python {} --sensitivity_analysis_file {} --bo_component_params {} --config_params {}".format(
        bo_py,
        sensitivity_analysis_file,
        bo_component_params,
        config_params
    )

    command2 = command2.replace('\"', '\\"')
    os.system(command2)

    c3 = time.time()
    time_dict = {}
    time_dict['SA搜索所需时间'] = c2 - c1
    time_dict2 = {}
    time_dict2['bo搜索所需时间'] = c3 - c2

    time_txt = os.path.join('./sensitivity_analysis/', 'time.txt')
    file_out = open(time_txt, 'w+', encoding='utf-8')
    file_out.write(str(time_dict))
    file_out.close()

    time_txt = os.path.join('./bo/', 'time.txt')
    file_out = open(time_txt, 'w+', encoding='utf-8')
    file_out.write(str(time_dict2))
    file_out.close()

    print('SA_bo算法运行时间:{}s'.format(c3 - c1))
