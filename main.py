"""
Created on August 1st 12:00:00 2021
main
@author:
"""

from CDAT import ANFIS
from CDAT import CDAT
from CDAT import SAMPLER
import numpy as np
import random
from sympy import *
import pandas as pd
import csv
import time

# 实验前准备（参考x264)：
# 创建以下文件夹：-目标系统名（文件夹）-data（文件夹）
# 准备以下数据：目标系统名.csv (放在目标系统名（文件夹）下）
# 函数：get_Performance：输入配置，输出性能

# 导入get_Performance
from Test.get_Test_Performance import get_performance as test_fun
import x264.main as x264
# from tomcat.Tomcat_performance import getPerformance as tomcat
from spark.benchmark.get_spark_Performance import get_performance as spark
from sqlite.get_sqlite_Performance import get_performance as sqlite
from sqlite.check_sqlite import config_fix as sqlite_fix
from Hadoop.get_hadoop_performance import get_performance as Hadoop
from apache.get_apache_performance import get_performance as apache
from redis.get_redis_Performance import get_3Times as redis
from tomcat.get_tomcat_performance import get_performance as tomcat
from cassandra.get_cassandra_Performance import get_performance as cassandra

SYSTEM = 'Test'# System name (same as file name)
PATH = 'H:/FSE_2022_ACTDS/ACTDS2/' + SYSTEM + '/' # Project path (absolute path)
WORKLOAD = '' # add '_'( e.g., '_Sort')


def Measure(configuration, system = SYSTEM):
    # Need to modify 0 #################################################################################################
    # Input is configuration (list) and output is performance value
    if system == 'Test':
        return test_fun(configuration)
    if system == 'x264':
        no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref = configuration
        return x264.X264.getPerformance(no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref)
    if system == 'tomcat':
        name_list = ['maxThreads', 'minSpareThreads', 'executorTerminationTimeoutMillis', 'connectionTimeout', 'maxConnections',
                 'maxKeepAliveRequests', 'acceptorThreadCount', 'asyncTimeout', 'acceptCount', 'socketBuffer', 'processorCache',
                 'keepAliveTimeout']
        params = {}
        for i in range(len(configuration)):
            params[name_list[i]] = configuration[i]
        return tomcat(params)
    if system == 'spark':
        name_list = ['executorCores', 'executorMemory', 'memoryFraction',
                     'memoryStorageFraction', 'defaultParallelism', 'shuffleCompress',
                     'shuffleSpillCompress', 'broadcastCompress', 'rddCompress', 'ioCompressionCodec',
                     'reducerMaxSizeInFlight', 'shuffleFileBuffer', 'serializer']
        params = {}
        for i in range(len(configuration)):
            params[name_list[i]] = configuration[i]
        return spark(params)
    if system == 'sqlite':
        return sqlite(configuration)
    if system == 'Hadoop':
        name_list = ['mapreduce_task_io_sort_factor', 'mapreduce_reduce_shuffle_merge_percent',
                 'mapreduce_output_fileoutputformat_compress', 'mapreduce_reduce_merge_inmem_threshold',
                 'mapreduce_job_reduces', 'mapreduce_map_sort_spill_percent',
                 'mapreduce_reduce_shuffle_input_buffer_percent', 'mapreduce_task_io_sort_mb',
                 'mapreduce_map_output_compress']
        params = {}
        for i in range(len(configuration)):
            params[name_list[i]] = configuration[i]
        return Hadoop(params)
    if system == 'apache':
        name_list = ['StartServers', 'MinSpareServers', 'MaxSpareServers', 'MaxRequestWorkers', 'MaxRequestsPerChild']
        params = {}
        for i in range(len(configuration)):
            params[name_list[i]] = configuration[i]
        return apache(params)
    if system == 'redis':
        name_list = ['replBacklogSize', 'hashMaxZiplistValue', 'hashMaxZiplistEntries', 'listMaxZiplistSize',
                 'activeDefragIgnoreBytes', 'activeDefragThresholdLower', 'replDisableTcpNodelay', 'hllSparseMaxBytes',
                 'hz']
        params = {}
        for i in range(len(configuration)):
            params[name_list[i]] = configuration[i]
        return redis(params)

    if system == 'cassandra':
        name_list = ['write_request_timeout_in_ms', 'read_request_timeout_in_ms', 'commitlog_total_space_in_mb',
                     'key_cache_size_in_mb',
                     'commitlog_segment_size_in_mb', 'dynamic_snitch_badness_threshold', 'index_summary_capacity_in_mb',
                     'key_cache_save_period', 'file_cache_size_in_mb', 'thrift_framed_transport_size_in_mb',
                     'memtable_heap_space_in_mb',
                     'concurrent_writes', 'index_summary_resize_interval_in_minutes', 'commitlog_sync_period_in_ms',
                     'range_request_timeout_in_ms',
                     'rpc_min_threads', 'batch_size_warn_threshold_in_kb', 'concurrent_reads',
                     'column_index_size_in_kb', 'dynamic_snitch_update_interval_in_ms',
                     'memtable_flush_writers', 'request_timeout_in_ms', 'cas_contention_timeout_in_ms',
                     'permissions_validity_in_ms',
                     'rpc_max_threads', 'truncate_request_timeout_in_ms', 'stream_throughput_outbound_megabits_per_sec',
                     'memtable_offheap_space_in_mb']

        params = {}
        for i in range(len(configuration)):
            params[name_list[i]] = configuration[i]
        return cassandra(params)

# Configuration coding: system side -> algorithm side
def Data_Preprocessing(X):
    Processed_Flag = np.zeros(X.shape[1])
    Map = []
    f_X = np.zeros(X.shape)

    for i in range(X.shape[1]):
        try:
            temp = float(X[0][i])

            Map.append('')
        except:
            if len(X[0][i])>0:
                Processed_Flag[i] = 1
                Map.append(np.unique(X[:,i]))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if Processed_Flag[j] == 1:
                X[i, j] = np.where(Map[j] == X[i,j])[0][0]
            f_X[i,j] = float(X[i,j])
            if pd.isnull(f_X[i,j]):
                f_X[i, j] = -1

    return f_X, Processed_Flag, Map

# Configuration decoding: algorithm side -> system side
def Translation(configuration, Processed_Flag, Map):
    new_configuration = list(configuration)
    for i in range(len(configuration)):
        if Processed_Flag[i] == 1:
            pos = np.min([int(configuration[i]),len(Map[i])-1])
            new_configuration[i] = Map[i][pos]
        else:
            if float(configuration[i])%1 == 0:
                new_configuration[i] = int(configuration[i])
        if configuration[i] == -1:
            new_configuration[i] = -1

    # If the system configuration has special internal constraints, please specify here ################################
    if SYSTEM == 'sqlite':
        new_configuration = sqlite_fix(new_configuration)
    if SYSTEM == 'Hadoop':
        for i in range(len(configuration)):
            if configuration[2] == 1:
                new_configuration[2] = 'true'
            else:
                new_configuration[2] = 'false'
            if configuration[8] == 1:
                new_configuration[8] = 'true'
            else:
                new_configuration[8] = 'false'

            if configuration[0] >= 100:
                new_configuration[0] = 100
            if configuration[0] <= 10:
                new_configuration[0] = 10
            if configuration[1] >= 0.9:
                new_configuration[1] = 0.9
            if configuration[1] <= 0.21:
                new_configuration[1] = 0.21
            if configuration[3] >= 1000:
                new_configuration[3] = 1000
            if configuration[3] <= 10:
                new_configuration[3] = 10
            if configuration[4] >= 1000:
                new_configuration[4] = 1000
            if configuration[4] <= 1:
                new_configuration[4] = 1
            if configuration[5] >= 0.9:
                new_configuration[5] = 0.9
            if configuration[5] <= 0.5:
                new_configuration[5] = 0.5
            if configuration[6] >= 0.8:
                new_configuration[6] = 0.8
            if configuration[6] <= 0.1:
                new_configuration[6] = 0.1
            if configuration[7] >= 260:
                new_configuration[7] = 260
            if configuration[7] <= 100:
                new_configuration[7] = 100



    return new_configuration

# Data file update (recommendation [5 times] for each completion of a set of storage once)
def Data_file_update(XY, Processed_Flag, Map, timestruct):

    file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + '_' + SYSTEM + WORKLOAD + "_Recommended.csv", "a+",
                 newline="")
    content = csv.writer(file1)
    for i in range(len(XY)):
        xy = Translation(XY[i], Processed_Flag, Map)
        content.writerow(xy)
    file1.close()

# Output optimization results (recommended [10] configuration and corresponding performance)
def output(timestruct):
    data = pd.read_csv(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + '_' + SYSTEM + WORKLOAD + "_Recommended.csv")
    name = list(data)
    data = np.array(data)
    # print(data)
    if SYSTEM in ['x264', 'sqlite', 'apache', 'tomcat', 'mysql']:
        data = ANFIS.Union(data)
    if SYSTEM in ['mysql']:
        min_flag = 1
    else:
        min_flag = -1
    data = data[np.argsort(min_flag * data[:, -1])[0:np.min([10, len(data)])]]
    print('Recommended performance & configuration:')
    for member in data:
        print([member[-1]], member[0:-1])
    n_data = {}
    for i in range(len(name)):
        n_data[name[i]] = data[:, i]
    n_data = pd.DataFrame(n_data)
    n_data.to_csv(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + '_' + SYSTEM + WORKLOAD + "_result.csv",
                  index=0)  # Output result file name: Time+system+workload+result.csv

# Subject of the experiment, the parameters are as follows:
# constraint of searches, number of recommendations (each time), initial sampling set size, system name
def Test(Times_Constraint = 90, Recommended_Number = 5, Initial_size = 50, system = SYSTEM):


    # Timestruct
    timestamp = time.time()
    timestruct = time.localtime(timestamp)

    # Need to modify 1 #################################################################################################
    # Add the name of the system to be tested in the list
    if system not in ['Test', 'x264', 'tomcat', 'spark', 'sqlite', 'Hadoop', 'apache', 'redis', 'cassandra']:
        print('Can not do this: ' + system)
        return

    # Need to modify 2 #################################################################################################
    # Overall format reference Test
    # file name
    # List of configuration names (including performance PERF)
    # bound (Closed interval, including upper and lower bounds)
    # int_flag (According to the configuration order, indicate whether it is an integer,
    # enumeration, boolean, and integer are all considered as integers

    # Test
    if system == 'Test':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_Test_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['x1', 'x2', 'PERF'])
        file1.close()

        # CB
        bound = [[0, 2*np.pi], [0, 2*np.pi]]
        int_flag = np.array([0,0])

    # x264
    if system == 'x264':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_x264_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['no-8x8dct', 'no-cabac', 'no-deblock', 'no-fast-pskip', 'no-mbtree', 'no-mixed-refs', 'no-weightb',
             'rc-lookahead', 'ref', 'PERF'])
        file1.close()

        # CB
        bound = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [40, 250], [1, 9]]
        int_flag = np.ones(9)

    # tomcat
    if system == 'tomcat':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_tomcat_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['maxThreads', 'minSpareThreads', 'executorTerminationTimeoutMillis', 'connectionTimeout', 'maxConnections',
             'maxKeepAliveRequests', 'acceptorThreadCount', 'asyncTimeout', 'acceptCount', 'socketBuffer',
             'processorCache',
             'keepAliveTimeout', 'PERF'])
        file1.close()

        # CB
        bound = [[1,500], [1,50], [0,8000], [0,50000],[1,50000],[1,200],[1,10],[1,50000],[1,50],[1,500],[1,500],[1,50]]
        int_flag = np.ones(12)

    # spark
    if system == 'spark':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_spark" + WORKLOAD + "_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(['executorCores', 'executorMemory', 'memoryFraction',
                       'memoryStorageFraction', 'defaultParallelism', 'shuffleCompress',
                       'shuffleSpillCompress', 'broadcastCompress', 'rddCompress', 'ioCompressionCodec',
                       'reducerMaxSizeInFlight', 'shuffleFileBuffer', 'serializer', 'PERF'])
        file1.close()
        # CB
        bound = [[1, 4], [1024, 4096], [0.1, 0.9], [0.1, 0.9], [1, 12], [0, 1], [0, 1], [0, 1], [0, 1], [0, 2],
                 [8, 96], [8, 64], [0, 1]]
        int_flag = np.array([1,1,0,0,1,1,1,1,1,1,1,1,1])

    # sqlite
    if system == 'sqlite':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_sqlite_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['STANDARD_CACHE_SIZE', 'LOWER_CACHE_SIZE', 'HIGHER_CACHE_SIZE', 'STANDARD_PAGE_SIZE',
             'LOWER_PAGE_SIZE', 'HIGHER_PAGE_SIZE', 'HIGHEST_PAGE_SIZE', 'SECURE_DELETE_TRUE',
             'SECURE_DELETE_FALSE', 'SECURE_DELETE_FAST', 'TEMP_STORE_DEFAULT',
             'TEMP_STORE_FILE', 'TEMP_STORE_MEMORY', 'SHARED_CACHE_TRUE', 'SHARED_CACHE_FALSE',
             'READ_UNCOMMITED_TRUE',
             'READ_UNCOMMITED_FALSE', 'FULLSYNC_TRUE', 'FULLSYNC_FALSE', 'TRANSACTION_MODE_DEFERRED',
             'TRANSACTION_MODE_IMMEDIATE',
             'TRANSACTION_MODE_EXCLUSIVE', 'PERF'])
        file1.close()
        # CB
        bound = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                 [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        int_flag = np.array(np.ones(22))

    # Hadoop
    if system == 'Hadoop':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_Hadoop" + WORKLOAD + "_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['mapreduce_task_io_sort_factor', 'mapreduce_reduce_shuffle_merge_percent',
             'mapreduce_output_fileoutputformat_compress', 'mapreduce_reduce_merge_inmem_threshold',
             'mapreduce_job_reduces', 'mapreduce_map_sort_spill_percent',
             'mapreduce_reduce_shuffle_input_buffer_percent', 'mapreduce_task_io_sort_mb',
             'mapreduce_map_output_compress', 'PERF'])
        file1.close()
        # CB
        bound = [[10, 100], [0.21, 0.9], [0, 1], [10, 1000], [1, 1000], [0.5, 0.9], [0.1, 0.8], [100, 260], [0, 1]]
        int_flag = np.array([1,0,1,1,1,0,0,1,1])

    # apache
    if system == 'apache':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_apache_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['StartServers', 'MinSpareServers', 'MaxSpareServers', 'MaxRequestWorkers', 'MaxRequestsPerChild', 'PERF'])
        file1.close()
        # CB
        bound = [[1, 10], [1, 10], [11, 20], [1, 1000], [0, 1000]]
        int_flag = np.array(np.ones(5))

    # redis
    if system == 'redis':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_redis_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['replBacklogSize', 'hashMaxZiplistValue', 'hashMaxZiplistEntries', 'listMaxZiplistSize',
             'activeDefragIgnoreBytes', 'activeDefragThresholdLower', 'replDisableTcpNodelay', 'hllSparseMaxBytes',
             'hz', 'PERF'])
        file1.close()
        # CB
        bound = [[1, 11], [32, 128], [256, 1024], [-5, -1], [100, 300], [5, 20], [0, 1], [0, 5000], [1, 501]]
        int_flag = np.array(np.ones(9))

    # cassandra
    if system == 'cassandra':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_cassandra_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['write_request_timeout_in_ms', 'read_request_timeout_in_ms', 'commitlog_total_space_in_mb',
             'key_cache_size_in_mb',
             'commitlog_segment_size_in_mb', 'dynamic_snitch_badness_threshold', 'index_summary_capacity_in_mb',
             'key_cache_save_period', 'file_cache_size_in_mb', 'thrift_framed_transport_size_in_mb',
             'memtable_heap_space_in_mb',
             'concurrent_writes', 'index_summary_resize_interval_in_minutes', 'commitlog_sync_period_in_ms',
             'range_request_timeout_in_ms',
             'rpc_min_threads', 'batch_size_warn_threshold_in_kb', 'concurrent_reads', 'column_index_size_in_kb',
             'dynamic_snitch_update_interval_in_ms',
             'memtable_flush_writers', 'request_timeout_in_ms', 'cas_contention_timeout_in_ms',
             'permissions_validity_in_ms',
             'rpc_max_threads', 'truncate_request_timeout_in_ms', 'stream_throughput_outbound_megabits_per_sec',
             'memtable_offheap_space_in_mb', 'PERF'])

        file1.close()
        # CB
        bound = [[20, 250000], [20, 250000], [200, 15000], [100, 15000], [30, 2024], [0, 1], [100, 15000], [10, 14400],
                 [250, 15000], [2, 28], [80, 15000], [20, 3800], [2, 60], [2200, 250000], [2000, 250000], [10, 1000],
                 [5, 100000], [31, 2000], [64, 60000], [100, 25000], [2, 28], [1000, 25000], [900, 25000], [1600, 25000],
                 [1048, 3800], [2000, 250000], [100, 8000], [200, 15000]]
        int_flag = np.array(np.ones(28))
        int_flag[5] = 0

    ####################################################################################################################


    #Sample
    data = pd.Series.tolist(pd.read_csv(PATH + system + WORKLOAD + ".csv"))
    data, Processed_Flag, Map = Data_Preprocessing(np.array(data))
    XY = data[random.sample(list(range(len(data))), Initial_size)]

    if system in ['mysql']:
        min_flag = -1
    else:
        min_flag = 1
    XY[:, -1] = XY[:, -1] * min_flag

    # XY = data

    T = len(XY)
    while T < Times_Constraint:
        #Train
        actds = CDAT(XY, bound = bound, int_flag = int_flag)

        #Re-Sample
        print('Reference configuration (T = ', T, '\b): \n', np.max(actds.Y),
              Translation(actds.X[np.argmax(actds.Y)], Processed_Flag, Map))
        X = actds.X[np.argmax(actds.Y)]
        #Recommend
        Recommended_configuration = actds.Generator(X, size = np.min([Recommended_Number, Times_Constraint - T]))
        Recommended_configuration = ANFIS.Union(Recommended_configuration)
        T += len(Recommended_configuration)

        Y = np.zeros((len(Recommended_configuration), 1))

        #Measure
        for i in range(len(Y)):
            Y[i] = Measure(configuration = Translation(Recommended_configuration[i], Processed_Flag, Map), system = system)
            print(Y[i],Translation(Recommended_configuration[i], Processed_Flag, Map))

        #Data_expansion
        XY = np.append(XY, np.append(Recommended_configuration, Y, axis = 1), axis = 0)
        Data_file_update(np.append(Recommended_configuration, min_flag * Y, axis = 1), Processed_Flag, Map, timestruct)

    #Recommended configuration
    actds = CDAT(XY[np.argsort(-XY[:,-1])[0:6]] , bound = bound, int_flag = int_flag)
    Recommended_configuration = actds.Generator(actds.X[np.argmax(actds.Y)], size = 10)
    Recommended_performance = np.zeros((len(Recommended_configuration), 1))
    for i in range(len(Recommended_performance)):
        Recommended_performance[i] = Measure(configuration = Translation(Recommended_configuration[i], Processed_Flag, Map))
    Data_file_update(np.append(Recommended_configuration, min_flag * Recommended_performance, axis=1), Processed_Flag, Map, timestruct)

    output(timestruct)

if __name__ == '__main__':

    # {100,200,300}*3

    # Test Now

    for i in range(1,4):
        print('ours-100-', i, ':')
        Test(Times_Constraint=90, Recommended_Number=5, Initial_size=50)
    for i in range(1,4):
        print('ours-200-', i, ':')
        Test(Times_Constraint=190, Recommended_Number=5, Initial_size=100)
    for i in range(1,4):
        print('ours-300-', i, ':')
        Test(Times_Constraint=290, Recommended_Number=10, Initial_size=150)

    # Test
    # configuration = [0, 200, 10, 1, 30, 1, 1, 2, 1, 2, 2, 50]
    # print(Measure(configuration))