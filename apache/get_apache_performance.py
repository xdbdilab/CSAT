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
    req = request.Request('http://47.104.81.179:9000/experiment/httpd?%s' % value, headers=headers)  # 这样就能把参数带过去了
    # print('http://47.104.81.179:9000/experiment/httpd?%s' % value)
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
            else:
                print("请求访问失败\n")
                f.close()
                return -1.0
        except Exception as e:
            print(e)
            i = i + 1
            if i < 3:
                time.sleep(5)
            else:
                return -1.0
        try:
            data = (float)(data)
            return data
        except Exception as e:
            print(e)
            time.sleep(3)
            continue

if __name__ == "__main__":
    params = {}
    conf_list = ['StartServers', 'MinSpareServers', 'MaxSpareServers', 'MaxRequestWorkers', '0']
    # 47.104.81.179:9000/experiment/httpd?StartServers=7&MinSpareServers=7&MaxSpareServers=12&MaxRequestWorkers=252&MaxRequestsPerChild=0
    for i in range(1, len(sys.argv)):
        # params[conf_list[i - 1]] = (float)(sys.argv[i])
        params[conf_list[i - 1]] = (sys.argv[i])

    config = [7,7,12,252,0]
    for i in range(5) :
        params[conf_list[i]] = (int)(config[i])

    result = get_performance(params=params)
    print(result)







