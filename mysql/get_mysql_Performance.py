# -*- encoding: utf-8 -*-
"""
 @Time : 2021/4/7 15:46
 @Author : zspp
 @File : getPerformance
 @Software: PyCharm
"""
import urllib
from urllib import request, parse
import sys
import socket
import random
import time


# 性能获取
# def get_performance(params, config_params, performance):
def get_performance(params):
    """

    获取性能值
    :param performance: 性能指标名称
    :param config_params:
    :param params: {'param1': 10, 'param2': 's1', ...}
    :return: 性能值
    """
    headers = {"user-agent": "Mizilla/5.0"}
    value = parse.urlencode(params)

    # 请求url
    req = request.Request('http://47.104.81.179:8080/mysql80/exec?%s' % value, headers=headers)  # 这样就能把参数带过去了

    # 下面是获得响应
    i = 0
    while True:
        try:
            f = request.urlopen(req, timeout=300)
            # print(f)
            Data = f.read()
            data = Data.decode('utf-8')
            if data == "error":
                print("查不到，报errror，重新尝试\n")
                f.close()
                if i < 2:
                    i += 1
                    time.sleep(5)
                else:
                    return -1.0
            else:
                # print("get_performance成功\n")
                f.close()
                break

        except Exception  as e:
            print(e)
            i = i + 1
            if i < 2:
                time.sleep(5)
            else:
                return -1.0
    return (float)(data)
if __name__ == "__main__":
    # Test
    params = {'innodb_buffer_pool_size': int(300000),
              'innodb_thread_concurrency': int(25),
              'innodb_adaptive_hash_index': int(1),
              'innodb_flush_log_at_trx_commit': int(1),
              'join_buffer_size': int(150000),
              'sort_buffer_size': int(150000),
              'tmp_table_size': int(150000),
              'thread_cache_size': int(25),
              'read_rnd_buffer_size': int(20480),
              'max_heap_table_size': int(1048576)}
    print(get_performance(params))

    # http://47.104.81.179:8080/mysql80/exec?
    # innodb_buffer_pool_size=300000&
    # innodb_thread_concurrency=25&
    # innodb_adaptive_hash_index=1&
    # innodb_flush_log_at_trx_commit=1&
    # join_buffer_size=150000&
    # sort_buffer_size=150000&
    # tmp_table_size=150000&
    # thread_cache_size=25&
    # read_rnd_buffer_size=20480&
    # max_heap_table_size=1048576

    # params[name_list[0]] = random.randint(262144, 3170304)
    # params[name_list[1]] = random.randint(0, 50)
    # params[name_list[2]] = random.sample([0, 1], 1)[0]
    # params[name_list[3]] = random.sample([0, 1, 2], 1)[0]
    # params[name_list[4]] = random.randint(131072, 150994944)
    # params[name_list[5]] = random.randint(131072, 150994944)
    # params[name_list[6]] = random.randint(131072, 150994944)
    # params[name_list[7]] = random.randint(0, 50)
    # params[name_list[8]] = random.randint(0, 131072)
    # params[name_list[9]] = random.randint(262144, 67108864)