# -*- encoding: utf-8 -*-
"""
 @Time : 2021/3/9 10:39
 @Author : zspp
 @File : main
 @Software: PyCharm 
"""
import os
import json

if __name__ == "__main__":
    # all_params = {
    #                 'a':  ['float', [0, 50], 0],
    #                 'b':  ['int', [0, 50], 0],
    #                 'ce': ['int', [0, 50], 0],
    #                 'd': ['int', [0, 50], 0],
    #                 'for': ['int', [0, 50], 0],
    #                 'gh': ['int', [0, 50], 0],
    #                 'haha': ['float', [0, 50], 0],
    #                 'hehe': ['float', [0, 50], 0],
    #                 'sort_buffer_size': ['int', [131072, 16777216], 262144],
    #                 'join_buffer_size': ['int', [131072, 16777216], 262144],
    #                 'innodb_buffer_pool_size': ['int', [268435456, 3221225472], 268435456],
    #                 'innodb_thread_concurrency': ['int', [0, 50], 0],
    #                 'innodb_adaptive_hash_index': ['string', ['OFF', 'ON'], 'ON'],
    #                 'innodb_flush_log_at_trx_commit': ['enum', [0, 1, 2], 1],
    #                 'tmp_table_size': ['int', [131072, 67108864], 16777216],
    #                 'thread_cache_size': ['int', [0, 50], 0],
    #                 'read_rnd_buffer_size': ['int', [0, 134217728], 262144],
    #                 'max_heap_table_size': ['int', [262144, 67108864], 16777216],
    #             }
    # black_list = ['a','b']
    # white_list = ['sort_buffer_size','join_buffer_size','innodb_buffer_pool_size',
    #               'innodb_thread_concurrency','innodb_adaptive_hash_index',
    #               'innodb_flush_log_at_trx_commit','tmp_table_size','thread_cache_size',
    #               'read_rnd_buffer_size','max_heap_table_size']

    common_params = {
        "white_list": "['audit_log_buffer_size','audit_log_format']",
        "performance": "queries_performed_read_max",
        "black_list": "['connect_timeout','ddl-rewriter']",
        "all_params": "{'audit_log_buffer_size':['int',[4096,1888888],1048576],'audit_log_format':['enum',['OLD','NEW','jSON'],'NEW'],'connect_timeout':['int',[2,31536000],10],'ddl-rewriter':['enum',['ON','OFF','FORCE','FORCE_PLUS_PERMANENT'],'ON']} "
    }
    config_params = {
        "ip_port": "120.27.69.55:8020",
        "conversationId": "7e5ae356-db1c-4288-85dd-529e8dd3d856",
        "output_dir": "/nfs/colosseum",
        "performance_retry_times": 2,
        "report_retry_times": 3,
        "report_retry_interval": 5
    }
    select_all_component_params = {}
    random_sample_component_params = {'sample_num': 300}
    sensitivity_analysis_component_params = {'alpha': 0.5, 'sobel_sample_number': 10, 'number_of_SA_iter': 2,
                                             'model_n_iterations': 1}
    bo_component_params = {'sample_num': 100}
    binary_numeric_sample_component_params = {'measuments': 8, 'level': 3}
    stepwise_linear_regression_component_params = {}
    random_tuning_component_params = {}
    rfhoc_component_params = {}
    actgan_component_params = {
        'Epoch': 100,
        'LR_G': 0.0001,
        'LR_D': 0.0001,
        'N_IDEAS': 5,
        'NumOfLine': 16,
        'BATCH_SIZE': 32
    }
    space_search_component_params = {
        "InitialSampleSetSize": 50,
        "RRSMaxRounds": 5
    }
    random_forest_component_params = {}

    genetic_algorithm_component_params = {'uniformRate': 0.5,  # 交叉概率
                                          'mutationRate': 0.015,  # 突变概率
                                          'tournamentSize': 5,  # 淘汰数组的大小
                                          'elitism': 'True',  # 精英主义
                                          'Population_SIZE': 100,
                                          'EPOCH': 10
                                          }
    hyperopt_sample_component_params = {"sample_num": 20}
    hyperopt_tuning_component_params = {}
    bogan_component_params = {
        'Epoch': 100,
        'LR_G': 0.0001,
        'LR_D': 0.0001,
        'N_IDEAS': 5,
        'NumOfLine': 5,
        'BATCH_SIZE': 10,
        'n1': 3,
        'n2': 2,
        'lam': 0.06
    }
    ottertune_component_params = {}

    select_all_component_params = json.dumps(select_all_component_params)
    random_sample_component_params = json.dumps(random_sample_component_params)
    sensitivity_analysis_component_params = json.dumps(sensitivity_analysis_component_params)
    rfhoc_component_params = json.dumps(rfhoc_component_params)
    random_tuning_component_params = json.dumps(random_tuning_component_params)
    actgan_component_params = json.dumps(actgan_component_params)
    space_search_component_params = json.dumps(space_search_component_params)
    random_forest_component_params = json.dumps(random_forest_component_params)
    genetic_algorithm_component_params = json.dumps(genetic_algorithm_component_params)
    hyperopt_sample_component_params = json.dumps(hyperopt_sample_component_params)
    hyperopt_tuning_component_params = json.dumps(hyperopt_tuning_component_params)
    bogan_component_params = json.dumps(bogan_component_params)
    bo_component_params = json.dumps(bo_component_params)
    binary_numeric_sample_component_params = json.dumps(binary_numeric_sample_component_params)
    stepwise_linear_regression_component_params = json.dumps(stepwise_linear_regression_component_params)
    ottertune_component_params = json.dumps(ottertune_component_params)

    config_params = json.dumps(config_params)
    common_params = json.dumps(common_params)

    select_all_py = os.getcwd() + '\select_all\select_all.py'
    random_sample_py = os.getcwd() + '\\random_sample\\random_sample.py'
    sensitivity_analysis_py = os.getcwd() + '\\sensitivity_analysis\\sensitivity_analysis.py'
    random_tuning_py = os.getcwd() + '\\random_tuning\\random_tuning.py'
    rfhoc_py = os.getcwd() + '\\rfhoc\\rfhoc.py'
    actgan_py = os.getcwd() + '\\actgan\\actgan.py'
    space_search_py = os.getcwd() + '\\space_search\\space_search.py'
    random_forest_py = os.getcwd() + '\\random_forest\\random_forest.py'
    genetic_algorithm_py = os.getcwd() + '\\genetic_algorithm\\genetic_algorithm.py'
    hyperopt_sample_py = os.getcwd() + '\\hyperopt_sample\\hyperopt_sample.py'
    hyperopt_tuning_py = os.getcwd() + '\\hyperopt_tuning\\hyperopt_tuning.py'
    bogan_py = os.getcwd() + '\\bogan\\bogan.py'
    bo_py = os.getcwd() + '\\bo\\bo.py'
    binary_numeric_sample_py = os.getcwd() + '\\binary_numeric_sample\\binary_numeric_sample.py'
    stepwise_linear_regression_py = os.getcwd() + '\\stepwise_linear_regression\\stepwise_linear_regression.py'
    ottertune_py = os.getcwd() + '\\ottertune\\ottertune.py'

    command1 = 'python %s --select_all_component_params %s --config_params %s --common_params %s' % (
        select_all_py, select_all_component_params, config_params, common_params)
    command1 = command1.replace('\"', '\\"')
    # print(command1)
    # os.system(command1)

    select_all_file = os.getcwd() + '\select_all\\83bdb364-b311-11eb-a118-74e5f9ebfdbf.txt'
    command2 = "python {} --select_all_file {} --random_sample_component_params {} --config_params {}".format(
        random_sample_py, select_all_file, random_sample_component_params, config_params)
    command2 = command2.replace('\"', '\\"')
    # print(command2)
    # os.system(command2)

    random_sample_file = os.getcwd() + '\\random_sample\\3f37a55e-c9bf-11eb-ac0f-005056c00008.txt'
    command3 = "python {} --random_sample_file {} --rfhoc_component_params {} --config_params {}".format(rfhoc_py,
                                                                                                         random_sample_file,
                                                                                                         rfhoc_component_params,
                                                                                                         config_params)
    command3 = command3.replace('\"', '\\"')
    # os.system(command3)

    command4 = "python {} --random_sample_file {} --random_tuning_component_params {} --config_params {}".format(
        random_tuning_py, random_sample_file, random_tuning_component_params, config_params)
    random_sample_file = os.getcwd() + '\\random_sample\\4c9d2322-b316-11eb-86c1-74e5f9ebfdbf.txt'
    command4 = command4.replace('\"', '\\"')
    os.system(command4)

    command5 = "python {} --random_sample_file {} --actgan_component_params {} --config_params {}".format(actgan_py,
                                                                                                          random_sample_file,
                                                                                                          actgan_component_params,
                                                                                                          config_params)
    random_sample_file = os.getcwd() + '\\random_sample\\4c9d2322-b316-11eb-86c1-74e5f9ebfdbf.txt'
    command5 = command5.replace('\"', '\\"')
    # os.system(command5)

    select_all_file = os.getcwd() + '\select_all\\83bdb364-b311-11eb-a118-74e5f9ebfdbf.txt'
    command6 = "python {} --select_all_file {} --space_search_component_params {} --config_params {}".format(
        space_search_py, select_all_file, space_search_component_params, config_params)
    command6 = command6.replace('\"', '\\"')
    # print(command6)
    # os.system(command6)

    command7 = "python {} --random_sample_file {} --random_forest_component_params {} --config_params {}".format(
        random_forest_py,
        random_sample_file,
        random_forest_component_params,
        config_params)
    command7 = command7.replace('\"', '\\"')
    # os.system(command7)

    random_forest_file = os.getcwd() + '\\random_forest\\e9a375b5-b324-11eb-aef3-74e5f9ebfdbf.txt'
    command8 = "python {} --random_forest_file {} --genetic_algorithm_component_params {} --config_params {}".format(
        genetic_algorithm_py,
        random_forest_file,
        genetic_algorithm_component_params,
        config_params)

    command8 = command8.replace('\"', '\\"')
    # os.system(command8)

    command9 = "python {} --select_all_file {} --hyperopt_sample_component_params {} --config_params {}".format(
        hyperopt_sample_py,
        select_all_file,
        hyperopt_sample_component_params,
        config_params)

    command9 = command9.replace('\"', '\\"')
    # os.system(command9)

    hyperopt_sample_file = os.getcwd() + '\\hyperopt_sample\\3538bf8a-b3fe-11eb-8cc1-74e5f9ebfdbf.txt'
    command10 = "python {} --hyperopt_sample_file {} --hyperopt_tuning_component_params {} --config_params {}".format(
        hyperopt_tuning_py,
        hyperopt_sample_file,
        hyperopt_tuning_component_params,
        config_params)

    command10 = command10.replace('\"', '\\"')
    # os.system(command10)

    command11 = "python {} --hyperopt_sample_file {} --bogan_component_params {} --config_params {}".format(
        bogan_py,
        hyperopt_sample_file,
        bogan_component_params,
        config_params)

    command11 = command11.replace('\"', '\\"')
    # os.system(command11)

    command12 = "python {} --common_params {} --sa_component_params {} --config_params {}".format(
        sensitivity_analysis_py,
        common_params,
        sensitivity_analysis_component_params,
        config_params
    )

    command12 = command12.replace('\"', '\\"')
    # os.system(command12)

    sensitivity_analysis_file = os.getcwd() + '\\SA\\917296ca-bae6-11eb-8f16-74e5f9ebfdbf.txt'
    command13 = "python {} --sensitivity_analysis_file {} --bo_component_params {} --config_params {}".format(
        bo_py,
        sensitivity_analysis_file,
        bo_component_params,
        config_params
    )

    command13 = command13.replace('\"', '\\"')
    # os.system(command13)

    command14 = "python {} --select_all_file {} --binary_numeric_sample_component_params {} --config_params {}".format(
        binary_numeric_sample_py,
        select_all_file,
        binary_numeric_sample_component_params,
        config_params
    )

    command14 = command14.replace('\"', '\\"')
    # os.system(command14)

    binary_numeric_sample_file = os.getcwd() + '\\binary_numeric_sample\\8f57e174-bc72-11eb-8a9c-74e5f9ebfdbf.txt'
    command15 = "python {} --binary_numeric_sample_file {} --stepwise_linear_regression_component_params {} --config_params {}".format(
        stepwise_linear_regression_py,
        binary_numeric_sample_file,
        stepwise_linear_regression_component_params,
        config_params
    )

    command15 = command15.replace('\"', '\\"')
    print(command15)
    # os.system(command15)

    command16 = "python {} --ottertune_component_params {} --random_sample_file {} --config_params {}".format(
        ottertune_py,
        ottertune_component_params,
        random_sample_file,
        config_params
    )

    command16 = command16.replace('\"', '\\"')
    print(command16)
    # os.system(command16)
