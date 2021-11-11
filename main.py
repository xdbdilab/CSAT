"""
Created on August 1st 12:00:00 2021
DPT main
@author: Lyle
"""

from ACTDS import ANFIS
from ACTDS import ACTDS
from ACTDS import SAMPLER
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
# from Tomcat.Tomcat_performance import getPerformance as Tomcat
from spark.benchmark.get_spark_Performance import get_performance as spark
from sqlite.get_sqlite_Performance import get_performance as sqlite
from sqlite.check_sqlite import config_fix as sqlite_fix
from Hadoop.get_hadoop_performance import get_performance as Hadoop
from apache.get_apache_performance import get_performance as apache
from redis.get_redis_Performance import get_3Times as redis

SYSTEM = 'Hadoop'# System name (same as file name)
PATH = 'H:/FSE_2022_ACTDS/ACTDS2/' + SYSTEM + '/' # Project path (absolute path)
WORKLOAD = '_Wordcount'

def Measure(configuration, system = SYSTEM):
    # 需要修改的地方0 ####################################################################################################
    # 输入为配置（列表）输出为性能值
    if system == 'Test':
        return test_fun(configuration)
    if system == 'x264':
        no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref = configuration
        return x264.X264.getPerformance(no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref)
    if system == 'Tomcat':
        return "UNDONE!"
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
    if SYSTEM in ['x264', 'sqlite', 'apache']:
        data = ANFIS.Union(data)
    data = data[np.argsort(-data[:, -1])[0:np.min([10, len(data)])]]
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

    # Need to modify 1 ####################################################################################################
    # Add the name of the system to be tested in the list
    if system not in ['Test', 'x264', 'Tomcat', 'spark', 'sqlite', 'Hadoop', 'apache', 'redis']:
        print('Can not do this: ' + system)
        return

    # Need to modify 2 ####################################################################################################
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

    # Tomcat
    if system == 'Tomcat':
        file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + "_Tomcat_Recommended.csv", "a+",
                     newline="")
        content = csv.writer(file1)
        content.writerow(
            ['connectionTimeout', 'maxConnections', 'maxKeepAliveRequests', 'acceptorThreadCount', 'asyncTimeout',
             'socketBuffer', 'acceptCount',
             'processorCache', 'keepAliveTimeout', 'maxThreads', 'minSpareThreads',
             'executorTerminationTimeoutMillis', 'PERF'])
        file1.close()

        # CB
        bound = [[0, 500000], [100, 500000], [1, 200], [1, 10], [1, 50000], [1, 50], [1, 200], [1, 500], [1, 30], [1, 500],
                 [1, 50], [0, 8000]]
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


    ####################################################################################################################


    #Sample
    data = pd.Series.tolist(pd.read_csv(PATH + system + WORKLOAD + ".csv"))
    data, Processed_Flag, Map = Data_Preprocessing(np.array(data))
    XY = data[random.sample(list(range(len(data))), Initial_size)]

    # XY = data

    T = len(XY)
    while T < Times_Constraint:
        #Train
        actds = ACTDS(XY, bound = bound, int_flag = int_flag)

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
        Data_file_update(np.append(Recommended_configuration, Y, axis = 1), Processed_Flag, Map, timestruct)

    #Recommended configuration
    actds = ACTDS(XY[np.argsort(-XY[:,-1])[0:6]] , bound = bound, int_flag = int_flag)
    Recommended_configuration = actds.Generator(actds.X[np.argmax(actds.Y)], size = 10)
    Recommended_performance = np.zeros((len(Recommended_configuration), 1))
    for i in range(len(Recommended_performance)):
        Recommended_performance[i] = Measure(configuration = Translation(Recommended_configuration[i], Processed_Flag, Map))
    Data_file_update(np.append(Recommended_configuration, Recommended_performance, axis=1), Processed_Flag, Map, timestruct)

    output(timestruct)

if __name__ == '__main__':

    # {100,200,300}*3
    # Test(Times_Constraint = 90, Recommended_Number = 5, Initial_size = 50)
    # Hadoop_Wordcount Now

    # for i in range(1,4):
    #     print('ours-100-', i, ':')
    #     Test(Times_Constraint=90, Recommended_Number=5, Initial_size=50)
    # for i in range(1,4):
    #     print('ours-200-', i, ':')
    #     Test(Times_Constraint=190, Recommended_Number=5, Initial_size=100)
    for i in range(1,4):
        print('ours-300-', i, ':')
        Test(Times_Constraint=290, Recommended_Number=10, Initial_size=150)

    # Test
    # configuration = [0, 200, 10, 1, 30, 1, 1, 2, 1, 2, 2, 50]
    # print(Measure(configuration))
