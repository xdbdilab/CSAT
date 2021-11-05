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
import x264.main as x264
# from Tomcat.Tomcat_performance import getPerformance as Tomcat
from spark.benchmark.get_spark_Performance import get_performance as spark
from sqlite.get_sqlite_Performance import get_performance as sqlite
from sqlite.check_sqlite import config_fix as sqlite_fix
from Hadoop.get_hadoop_performance import get_performance as Hadoop
from apache.get_apache_performance import get_performance as apache

SYSTEM = 'Hadoop'# 系统名称(和文件名称保持统一）
PATH = 'H:/FSE_2022_ACTDS/ACTDS2/' + SYSTEM + '/' # 项目路径（绝对）
WORKLOAD = '_Terasort'

def Measure(configuration, system = SYSTEM):
    # 需要修改的地方0 ####################################################################################################
    # 输入为配置（列表）输出为性能值
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

# 配置编码：系统侧->算法侧
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

# 配置解码：算法侧->系统侧
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

    # 如果系统配置有特殊的内部约束，请在这里指明 ############################################################################
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


    return new_configuration

# 数据文件更新（每完成一组推荐[10次]存储一次）
def Data_file_update(XY, Processed_Flag, Map, timestruct):

    file1 = open(PATH + 'data/' + time.strftime('%Y%m%d%H%M%S', timestruct) + '_' + SYSTEM + WORKLOAD + "_Recommended.csv", "a+",
                 newline="")
    content = csv.writer(file1)
    for i in range(len(XY)):
        xy = Translation(XY[i], Processed_Flag, Map)
        content.writerow(xy)
    file1.close()

# 输出优化结果（推荐的[10个]配置组合及对应性能）
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
                  index=0)  # 输出结果文件名：时间+系统+后缀+result.csv


# 实验主体，参数依此为：搜索次数、推荐个数（每次）、初始采样集大小、系统名称
def Test(Times_Constraint = 90, Recommended_Number = 5, Initial_size = 60, system = SYSTEM):


    # Timestruct
    timestamp = time.time()
    timestruct = time.localtime(timestamp)

    # 需要修改的地方1 ####################################################################################################
    # 在列表中加入待实验的软件名
    if system not in ['x264', 'Tomcat', 'spark', 'sqlite', 'Hadoop', 'apache']:
        print('Can not do this: ' + system)
        return

    # 需要修改的地方2 ####################################################################################################
    # 整体格式参考x264
    # 文件名
    # 配置名列表（包含性能PERF）
    # bound（闭区间，包含上下界）
    # int_flag（根据配置顺序指明是否为整形，枚举，布尔，整数都算整形

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
    # Hadoop_Terasort Now
    for i in range(1,4):
        print('ours-100-', i, ':')
        Test(Times_Constraint=90, Recommended_Number=5, Initial_size=50)
    for i in range(1,4):
        print('ours-200-', i, ':')
        Test(Times_Constraint=190, Recommended_Number=5, Initial_size=100)
    for i in range(1,4):
        print('ours-300-', i, ':')
        Test(Times_Constraint=290, Recommended_Number=5, Initial_size=150)

    # Test
    # configuration = [0, 200, 10, 1, 30, 1, 1, 2, 1, 2, 2, 50]
    # print(Measure(configuration))
