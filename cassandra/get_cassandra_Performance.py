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
    # 47.104.172.188
    # 118.190.211.206
    req = request.Request('http://47.104.235.57:8080/experiment/exec?%s' % value, headers=headers)  # 这样就能把参数带过去了

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
    name_list = ['write_request_timeout_in_ms','read_request_timeout_in_ms','commitlog_total_space_in_mb','key_cache_size_in_mb',
                 'commitlog_segment_size_in_mb','dynamic_snitch_badness_threshold','index_summary_capacity_in_mb',
                 'key_cache_save_period','file_cache_size_in_mb','thrift_framed_transport_size_in_mb','memtable_heap_space_in_mb',
                 'concurrent_writes','index_summary_resize_interval_in_minutes','commitlog_sync_period_in_ms','range_request_timeout_in_ms',
                 'rpc_min_threads','batch_size_warn_threshold_in_kb','concurrent_reads','column_index_size_in_kb','dynamic_snitch_update_interval_in_ms',
                 'memtable_flush_writers','request_timeout_in_ms','cas_contention_timeout_in_ms','permissions_validity_in_ms',
                 'rpc_max_threads','truncate_request_timeout_in_ms','stream_throughput_outbound_megabits_per_sec','memtable_offheap_space_in_mb']
    config_list = [2000,5000,8192,100,32,0.1,100,14400,512,15,2048,32,60,10000,10000,16,5,32,64,100,2,10000,1000,2000,2048,60000,200,2048]
    params = {}
    for i in range(len(name_list)):
        params[name_list[i]] = config_list[i]

    print(get_performance(params))

    # write_request_timeout_in_ms=1&
    # read_request_timeout_in_ms=1&
    # commitlog_total_space_in_mb=1&
    # key_cache_size_in_mb=1&
    # commitlog_segment_size_in_mb=1&
    # dynamic_snitch_badness_threshold=1.0&
    # index_summary_capacity_in_mb=1&
    # key_cache_save_period=1&
    # file_cache_size_in_mb=1&
    # thrift_framed_transport_size_in_mb=1&
    # memtable_heap_space_in_mb=1&
    # concurrent_writes=2&
    # index_summary_resize_interval_in_minutes=1&
    # commitlog_sync_period_in_ms=1&
    # range_request_timeout_in_ms=1&
    # rpc_min_threads=1&
    # batch_size_warn_threshold_in_kb=1&
    # concurrent_reads=2&
    # column_index_size_in_kb=1&
    # dynamic_snitch_update_interval_in_ms=1&
    # memtable_flush_writers=1&
    # request_timeout_in_ms=1&
    # cas_contention_timeout_in_ms=1&
    # permissions_validity_in_ms=1&
    # rpc_max_threads=1&
    # truncate_request_timeout_in_ms=1&
    # stream_throughput_outbound_megabits_per_sec=1&
    # memtable_offheap_space_in_mb=1