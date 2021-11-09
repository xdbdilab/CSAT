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
import numpy as np
import time


# 性能获取
# def get_performance(params, config_params, performance):

def get_3Times(params):
    res = np.ones(3)
    for i in range(3):
        res[i] = get_performance(params)

    return np.mean(res)

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
    req = request.Request('http://47.104.81.179:8080/experiment/redis?%s' % value, headers=headers)  # 这样就能把参数带过去了
    # print('http://47.104.81.179:8080/experiment/redis?%s' % value)

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
    params = {'replBacklogSize': int(10),
              'hashMaxZiplistValue': int(32),
              'hashMaxZiplistEntries': int(1024),
              'listMaxZiplistSize': int(-1),
              'activeDefragIgnoreBytes': int(284),
              'activeDefragThresholdLower': int(9),
              'replDisableTcpNodelay': "yes",
              'hz': int(9)}
    for i in range(10):
        print(get_3Times(params))

    # http://47.104.81.179:8080/experiment/redis?replBacklogSize=10&hashMaxZiplistValue=60&hashMaxZiplistEntries=862&
    # listMaxZiplistSize=3&activeDefragIgnoreBytes=162&activeDefragThresholdLower=15&replDisableTcpNodelay=no&
    # hllSparseMaxBytes=11398&hz=22