import random
from urllib import request, parse
import json
import sys
import time
import numpy as np
# 输入格式为字典
def get_performance(params):
    headers = {"user-agent": "Mizilla/5.0"}
    # 下面是需要传递的参数
    # print(params)
    # params['innodb_buffer_pool_size'].astype(np.int64)
    # print(type(params['innodb_buffer_pool_size']))
    # params['innodb_buffer_pool_size']=((np.int64)(params['innodb_buffer_pool_size']))
    # print(type(params['innodb_buffer_pool_size']))
    # print(params)
    value = parse.urlencode(params)
    # print(value)
    # 请求url
    # http://118.190.159.150:8019/experiment/hadoopTerasort Terasort
    # http://118.190.159.150:8019/experiment/hadoopWordcount Wordcount
    # http://118.190.159.150:8019/experiment/hadoopSort Sort
    # req = request.Request('http://118.190.159.150:8019/experiment/hadoopTerasort?%s' % value, headers=headers)  # 这样就能把参数带过去了
    req = request.Request('http://118.190.159.150:8019/experiment/hadoopTerasort?%s' % value, headers=headers)  # 这样就能把参数带过去了

    print('http://118.190.159.150:8019/experiment/hadoopTerasort?%s' % value)
    # 下面是获得响应
    i = 0

    while True:
        try:
            f = request.urlopen(req, timeout=360)
            # print(f)
            if f.getcode() == 200:
                # print("访问成功了\n")
                i = 0
                Data = f.read()
                data = Data.decode('utf-8')
                if data == "error":
                    print("查不到，报errror，重新尝试\n")
                    f.close()
                    return -1.0
                else:
                    # print("get_performance成功\n")
                    f.close()
                    break
            else:
                print("请求访问失败\n")
                f.close()
                return -1.0
        except Exception  as e:
            print(e)
            i = i + 1
            if i < 3:
                time.sleep(5)
            else:
                return -1.0
    # if (float)(data)>=60:
    #     data = -1
    return (float)(data)

if __name__ == "__main__":
    params = {}
    conf_list = ['mapreduce_task_io_sort_factor', 'mapreduce_reduce_shuffle_merge_percent',
                 'mapreduce_output_fileoutputformat_compress', 'mapreduce_reduce_merge_inmem_threshold',
                 'mapreduce_job_reduces', 'mapreduce_map_sort_spill_percent',
                 'mapreduce_reduce_shuffle_input_buffer_percent', 'mapreduce_task_io_sort_mb',
                 'mapreduce_map_output_compress']

    params[conf_list[0]] = (int)(10)

    params[conf_list[1]] = (float)(0.66)
    params[conf_list[2]] = 'false'
    params[conf_list[3]] = (int)(1000)
    params[conf_list[4]] = (int)(1)
    params[conf_list[5]] = (float)(0.8)
    params[conf_list[6]] = (float)(0.7)
    params[conf_list[7]] = (int)(100)
    params[conf_list[8]] = 'false'

    result=get_performance(params=params)
    print(result)