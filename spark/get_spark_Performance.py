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
    # http://47.104.101.50:8080/experiment/redis?     redis
    # http://47.104.101.50:9090/experiment/redis?     cassandra
    # http://47.104.101.50:8088/x264/exec?  x264
    # req = request.Request('http://47.104.79.64:8019/experiment/hadoopSor?%s' % value, headers=headers)  # 这样就能把参数带过去了
    # print('http://47.104.79.64:8019/experiment/hadoopSort?%s' % value)
    req = request.Request('http://118.190.211.206:9000/api/spark/run?%s' % value, headers=headers)  # 这样就能把参数带过去了
    # print('http://118.190.211.206:9000/api/spark/run?%s' % value)

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
    params = {'spark.executor.cores': int(1),
              'spark.executor.memory': int(1024),
              'spark.memory.fraction': float(0.6),
              'spark.memory.storageFraction': float(0.6),
              'spark.default.parallelism': int(2),
              'spark.shuffle.compress': "true",
              'spark.shuffle.spill.compress': "true",
              'spark.broadcast.compress': "true",
              'spark.rdd.compress': "true",
              'spark.io.compression.codec': "snappy",
              'spark.reducer.maxSizeInFlight': int(48),
              'spark.shuffle.file.buffer': int(32),
              'spark.serializer': "org.apache.spark.serializer.JavaSerializer"}
    print(get_performance(params))
    # "executorCores": ["int", [1, 4], 1],
    # "executorMemory": ["int", [1024, 4096], 1024],
    # "memoryFraction": ["float", [0.1, 0.9], 0.6],
    # "memoryStorageFraction": ["float", [0.1, 0.9], 0.6],
    # "defaultParallelism": ["int", [1, 12], 2],
    # "shuffleCompress": ["enum", ["true", "false"], "true"],
    # "shuffleSpillCompress": ["enum", ["true", "false"], "true"],
    # "broadcastCompress": ["enum", ["true", "false"], "true"],
    # "rddCompress": ["enum", ["true", "false"], "true"],
    # "ioCompressionCodec": ["enum", ["lz4", "lzf", "snappy"], "snappy"],
    # "reducerMaxSizeInFlight": ["int", [8, 96], 48],
    # "shuffleFileBuffer": ["int", [8, 64], 32],
    # "serializer": ["enum", ["org.apache.spark.serializer.JavaSerializer", "org.apache.spark.serializer.KryoSerializer"],
    #                "org.apache.spark.serializer.JavaSerializer"]
