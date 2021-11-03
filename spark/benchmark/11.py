# -*- encoding: utf-8 -*-
"""
 @Time : 2021/6/18 11:49
 @Author : zspp
 @File : 11
 @Software: PyCharm 
"""
def haha():
    while True:
        i=0
        try:

            # print(f)
            if f.getcode() == 200:
                # print("访问成功了\n")
                i = 0

                if data == "error":
                    print("查不到，报errror，重新尝试\n")
                    return -1.0
                else:
                    print("get_performance成功\n")

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