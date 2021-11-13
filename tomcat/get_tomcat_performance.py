import random
from urllib import request, parse
import requests
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
    # 47.104.81.179:8081/jmeter?
    req = requests.post('http://47.104.81.179:8081/jmeter?%s' % value, headers=headers)  # 这样就能把参数带过去了
    # print('http://47.104.81.179:8081/jmeter?%s' % value)
    # 下面是获得响应
    i = 0

    while True:
        try:
            # f = request.urlopen(req, timeout=360)
            # print(f)
            # if f.getcode() == 200:
            if req.status_code == 200:
                # print("访问成功了\n")
                i = 0
                # Data = f.read()
                Data = req.text
                # data = Data.decode('utf-8')
                data = Data
                if data == "error":
                    print("查不到，报errror，重新尝试\n")
                    # f.close()
                    return -1.0
                else:
                    # print("get_performance成功\n")
                    # f.close()
                    break;
            else:
                print("请求访问失败\n")
                # f.close()
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

    # maxThreads=200&minSpareThreads=25&executorTerminationTimeoutMillis=5000&connectionTimeout=30000&maxConnections=20000&maxKeepAliveRequests=100&
    # acceptorThreadCount=1&asyncTimeout=30000&acceptCount=10&socketBuffer=100&processorCache=200&keepAliveTimeout=15
    params = {}
    conf_list = ['maxThreads', 'minSpareThreads', 'executorTerminationTimeoutMillis', 'connectionTimeout', 'maxConnections',
                 'maxKeepAliveRequests', 'acceptorThreadCount', 'asyncTimeout', 'acceptCount', 'socketBuffer', 'processorCache',
                 'keepAliveTimeout']
    default_value = [200, 25, 5000, 30000, 20000, 100, 1, 30000, 10, 100, 200, 15]

    # for i in range(1, len(sys.argv)):
    #     # params[conf_list[i - 1]] = (float)(sys.argv[i])
    #     params[conf_list[i - 1]] = (sys.argv[i])
    # params[conf_list[0]] = (int)(params.get(conf_list[0]))
    # params[conf_list[1]] = (float)(params.get(conf_list[1]))
    # for i in range(3,5):
    #     params[conf_list[i]] = (int)(params.get(conf_list[i]))
    # for i in range(5,7):
    #     params[conf_list[i]] = (float)(params.get(conf_list[i]))
    # params[conf_list[7]] = (int)(params.get(conf_list[7]))

    for i in range(12):
        params[conf_list[i]] = int(default_value[i])

    result = get_performance(params=params)
    print(result)







