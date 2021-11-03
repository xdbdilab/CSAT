# -*- encoding: utf-8 -*-
import os
import json
import time

if __name__ == "__main__":
    c1 = time.time()
    a = json.load(open('paras.json'), strict=False)
    select_all_component_params = a['select_all_component_params']
    random_sample_component_params = a['random_sample_component_params']
    lasso_select_component_params = a['lasso_select_component_params']
    gaussian_process_component_params = a['gaussian_process_component_params']
    random_search_component_params = a['random_search_component_params']
    config_params = a['config_params']
    common_params = a['common_params']
    common_params = json.dumps(common_params)
    lasso_select_component_params = json.dumps(lasso_select_component_params)

    select_all_py = os.getcwd() + '\\select_all\\select_all.py'
    random_sample_py = os.getcwd() + '\\random_sample\\random_sample.py'
    lasso_select_py = os.getcwd() + '\\lasso_select\\lasso_select.py'
    gaussian_process_py = os.getcwd() + '\\gaussian_process\\gaussian_process.py'
    random_search_py = os.getcwd() + '\\random_search\\random_search.py'

    select_all_file = os.getcwd() + '\\select_all\\result.txt'
    random_sample_file = os.getcwd() + '\\random_sample\\result.txt'
    lasso_select_file = os.getcwd() + '\\lasso_select\\result.txt'
    gaussian_process_file = os.getcwd() + '\\gaussian_process\\result.txt'

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

    # lasso_select
    command3 = "python %s --random_sample_file %s --lasso_select_component_params %r --config_params %s --common_params %r"%(lasso_select_py, random_sample_file, lasso_select_component_params, config_params, common_params)
    command3 = command3.replace('\"', '\\"')
    command3 = command3.replace('\'', '\"')
    os.system(command3)

    # gaussian process model
    command4 = "python {} --lasso_select_file {} --gaussian_process_component_params {} --config_params {}".format(
        gaussian_process_py, lasso_select_file, gaussian_process_component_params, config_params)
    command4 = command4.replace('\"', '\\"')
    os.system(command4)

    # random search
    command5 = "python {} --gaussian_process_file {} --random_search_component_params {} --config_params {}".format(
        random_search_py, gaussian_process_file, random_search_component_params, config_params)
    command5 = command5.replace('\"', '\\"')
    #os.system(command5)

    # c2 = time.time()
    # print('ottertune后三块模块运行时间（没有包括随机采样模块的时间）:{}s'.format(c2 - c1))
