from spark.benchmark.get_spark_Performance import get_performance as spark
from Hadoop.get_hadoop_performance import get_performance as Hadoop
from redis.get_redis_Performance import get_3Times as redis
from cassandra.get_cassandra_Performance import get_performance as cassandra
from sqlite.get_sqlite_Performance import get_performance as sqlite
import CNFwithSAT as cnf
import x264.main as x264
import csv
import random
import numpy as np
from tqdm import trange
import time
from copy import deepcopy
import pandas as pd


SYSTEM = 'sqlite'# System name (same as file name)
PATH = SYSTEM + '/'  # Project path (absolute path)
WORKLOAD = '_CNF' # add '_'(e.g., '_Sort')



def Random_spark(size = 100, tag = 1):

    params = {}
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    file1 = open(PATH + 'data/' + "random_spark_sort_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                     timestruct) + ".csv",
                 "a+", newline="")
    content = csv.writer(file1)

    # title
    name_list = ['executorCores', 'executorMemory', 'memoryFraction',
         'memoryStorageFraction', 'defaultParallelism', 'shuffleCompress',
         'shuffleSpillCompress', 'broadcastCompress', 'rddCompress', 'ioCompressionCodec',
         'reducerMaxSizeInFlight', 'shuffleFileBuffer', 'serializer', 'PERF']
    content.writerow(name_list)
    file1.close()

    for _ in trange(size):
        file1 = open(PATH + 'data/' + "random_spark_sort_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                         timestruct) + ".csv",
                     "a+", newline="")
        content = csv.writer(file1)
        params[name_list[0]] = random.randint(1, 4)
        params[name_list[1]] = random.randint(1024, 4096)
        params[name_list[2]] = random.uniform(0.1, 0.9)
        params[name_list[3]] = random.uniform(0.1, 0.9)
        params[name_list[4]] = random.randint(1, 12)
        params[name_list[5]] = random.sample(["true", "false"], 1)[0]
        params[name_list[6]] = random.sample(["true", "false"], 1)[0]
        params[name_list[7]] = random.sample(["true", "false"], 1)[0]
        params[name_list[8]] = random.sample(["true", "false"], 1)[0]
        params[name_list[9]] = random.sample(["lz4", "lzf", "snappy"], 1)[0]
        params[name_list[10]] = random.randint(8, 96)
        params[name_list[11]] = random.randint(8, 64)
        params[name_list[12]] = random.sample(["org.apache.spark.serializer.JavaSerializer", "org.apache.spark.serializer.KryoSerializer"], 1)[0]
        pref = spark(params)
        content.writerow(np.append(list(params.values()), pref))
        print([pref], list(params.values()))
        file1.close()
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

def Random_Hadoop(size = 100, tag = 1):
    params = {}
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    file1 = open(
        PATH + 'data/' + "random_Hadoop_Wordcount_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                      timestruct) + ".csv",
        "a+", newline="")
    content = csv.writer(file1)

    # title
    name_list = ['mapreduce_task_io_sort_factor', 'mapreduce_reduce_shuffle_merge_percent',
             'mapreduce_output_fileoutputformat_compress', 'mapreduce_reduce_merge_inmem_threshold',
             'mapreduce_job_reduces', 'mapreduce_map_sort_spill_percent',
             'mapreduce_reduce_shuffle_input_buffer_percent', 'mapreduce_task_io_sort_mb',
             'mapreduce_map_output_compress', 'PERF']
    content.writerow(name_list)
    file1.close()

    for _ in trange(size):
        file1 = open(PATH + 'data/' + "random_Hadoop_Wordcount_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                      timestruct) + ".csv",
                     "a+", newline="")
        content = csv.writer(file1)
        params[name_list[0]] = random.randint(10, 100)
        params[name_list[1]] = random.uniform(0.21, 0.9)
        params[name_list[2]] = random.sample(["true", "false"], 1)[0]
        params[name_list[3]] = random.randint(10, 1000)
        params[name_list[4]] = random.randint(1, 1000)
        params[name_list[5]] = random.uniform(0.5, 0.9)
        params[name_list[6]] = random.uniform(0.1, 0.8)
        params[name_list[7]] = random.randint(100, 260)
        params[name_list[8]] = random.sample(["true", "false"], 1)[0]

        pref = Hadoop(params)
        content.writerow(np.append(list(params.values()), pref))
        print([pref], list(params.values()))
        file1.close()

def Random_redis(size = 100, tag = 1):
    params = {}
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    file1 = open(
        PATH + 'data/' + "random_redis_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                      timestruct) + ".csv",
        "a+", newline="")
    content = csv.writer(file1)

    # title
    name_list = ['replBacklogSize', 'hashMaxZiplistValue', 'hashMaxZiplistEntries', 'listMaxZiplistSize',
                 'activeDefragIgnoreBytes', 'activeDefragThresholdLower', 'replDisableTcpNodelay', 'hllSparseMaxBytes',
                 'hz', 'PERF']
    content.writerow(name_list)
    file1.close()

    for _ in trange(size):
        file1 = open(PATH + 'data/' + "random_redis_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                      timestruct) + ".csv",
                     "a+", newline="")
        content = csv.writer(file1)
        params[name_list[0]] = random.randint(1, 11)
        params[name_list[1]] = random.randint(32, 128)
        params[name_list[2]] = random.randint(256, 1024)
        params[name_list[3]] = random.randint(-5, -1)
        params[name_list[4]] = random.randint(100, 300)
        params[name_list[5]] = random.randint(5, 20)
        params[name_list[6]] = random.sample(["yes", "no"], 1)[0]
        params[name_list[7]] = random.randint(0, 5000)
        params[name_list[8]] = random.randint(1, 501)

        pref = redis(params)
        content.writerow(np.append(list(params.values()), pref))
        print([pref], list(params.values()))
        file1.close()

def Random_cassandra(size = 100, tag = 1):
    params = {}
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    file1 = open(
        PATH + 'data/' + "random_cassandra_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                      timestruct) + ".csv",
        "a+", newline="")
    content = csv.writer(file1)

    # title
    name_list = ['write_request_timeout_in_ms', 'read_request_timeout_in_ms', 'commitlog_total_space_in_mb',
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
                 'memtable_offheap_space_in_mb', 'PERF']

    content.writerow(name_list)
    file1.close()

    for _ in trange(size):
        file1 = open(PATH + 'data/' + "random_cassandra_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                      timestruct) + ".csv",
                     "a+", newline="")
        content = csv.writer(file1)
        params[name_list[0]] = random.randint(20, 250000)
        params[name_list[1]] = random.randint(20, 250000)
        params[name_list[2]] = random.randint(200, 15000)
        params[name_list[3]] = random.randint(100, 15000)
        params[name_list[4]] = random.randint(30, 2024)
        params[name_list[5]] = random.uniform(0, 1)
        params[name_list[6]] = random.randint(100, 15000)
        params[name_list[7]] = random.randint(10, 14400)
        params[name_list[8]] = random.randint(250, 15000)
        params[name_list[9]] = random.randint(2, 28)
        params[name_list[10]] = random.randint(80, 15000)
        params[name_list[11]] = random.randint(20, 3800)
        params[name_list[12]] = random.randint(2, 60)
        params[name_list[13]] = random.randint(2200, 250000)
        params[name_list[14]] = random.randint(2000, 250000)
        params[name_list[15]] = random.randint(10, 1000)
        params[name_list[16]] = random.randint(5, 100000)
        params[name_list[17]] = random.randint(31, 2000)
        params[name_list[18]] = random.randint(64, 60000)
        params[name_list[19]] = random.randint(100, 25000)
        params[name_list[20]] = random.randint(2, 28)
        params[name_list[21]] = random.randint(1000, 25000)
        params[name_list[22]] = random.randint(900, 25000)
        params[name_list[23]] = random.randint(1600, 25000)
        params[name_list[24]] = random.randint(1048, 3800)
        params[name_list[25]] = random.randint(2000, 250000)
        params[name_list[26]] = random.randint(100, 8000)
        params[name_list[27]] = random.randint(200, 15000)


        pref = cassandra(params)
        content.writerow(np.append(list(params.values()), pref))
        print([pref], list(params.values()))
        file1.close()

def Random_x264(size = 100, tag = 1, CNF_flag = False):

    SYSTEM = 'x264'

    bound = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [40, 250], [1, 9]]
    threshold = np.mean(bound, axis=1)
    k = len(bound)
    try:
        clauses = cnf.load_CNF(PATH + SYSTEM + '_CNF.txt')
    except:
        while 1:
            clauses = [cnf.random_clause(k) for i in range(int(k / 2))]
            print('CNF: ', clauses, end=', ')
            model = {}
            result = cnf.dpll(deepcopy(clauses), model)
            print(result[0], end='')
            if result[0]:
                score = cnf.test_CNF(deepcopy(clauses), k)
                print(', ', score)
                if score < 0.15:
                    continue
                else:
                    cnf.save_CNF(PATH + SYSTEM + '_CNF.txt', clauses)
                    break
            print('\n')

    params = {}
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    if CNF_flag:
        file1 = open(
            PATH + 'data/' + "random_x264_CNF_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                               timestruct) + ".csv",
            "a+", newline="")
    else:
        file1 = open(
            PATH + 'data/' + "random_x264_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                               timestruct) + ".csv",
            "a+", newline="")
    content = csv.writer(file1)

    # title
    name_list = ['no-8x8dct', 'no-cabac', 'no-deblock', 'no-fast-pskip', 'no-mbtree', 'no-mixed-refs', 'no-weightb',
             'rc-lookahead', 'ref', 'PERF']

    content.writerow(name_list)
    file1.close()

    for _ in trange(size):
        if CNF_flag:
            file1 = open(
                PATH + 'data/' + "random_x264_CNF_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                       timestruct) + ".csv",
                "a+", newline="")
        else:
            file1 = open(
                PATH + 'data/' + "random_x264_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                   timestruct) + ".csv",
                "a+", newline="")
        content = csv.writer(file1)

        params_list = []
        for i in range(len(bound)):
            value = random.randint(bound[i][0], bound[i][1])
            params[name_list[i]] = value
            params_list.append(value)

        flag = True
        if CNF_flag:
            model = dict()
            for i in range(1, k + 1):
                if params_list[i - 1] >= threshold[i - 1]:
                    model[i] = True
                else:
                    model[i] = False
            flag = cnf.verification(clauses, model)
        if flag:
            pref = x264.X264.getPerformance(*params_list)
        else:
            pref = -1

        content.writerow(np.append(list(params.values()), pref))
        print([pref], list(params.values()))
        file1.close()

def Random_sqlite(size = 100, tag = 1, CNF_flag = False):

    SYSTEM = 'sqlite'

    bound = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
             [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    threshold = np.mean(bound, axis=1)
    k = len(bound)
    try:
        clauses = cnf.load_CNF(PATH + SYSTEM + '_CNF.txt')
    except:
        while 1:
            clauses = [cnf.random_clause(k) for i in range(3)]
            print('CNF: ', clauses, end=', ')
            model = {}
            result = cnf.dpll(deepcopy(clauses), model)
            print(result[0], end='')
            if result[0]:

                score = cnf.test_CNF(deepcopy(clauses), k)
                print(', ', score)
                if score < 0.15:
                    continue
                else:
                    cnf.save_CNF(PATH + SYSTEM + '_CNF.txt', clauses)
                    break
            print('\n')

    params = {}
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    if CNF_flag:
        file1 = open(
            PATH + 'data/' + "random_sqlite_CNF_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                               timestruct) + ".csv",
            "a+", newline="")
    else:
        file1 = open(
            PATH + 'data/' + "random_sqlite_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                               timestruct) + ".csv",
            "a+", newline="")
    content = csv.writer(file1)

    # title
    name_list = ['STANDARD_CACHE_SIZE', 'LOWER_CACHE_SIZE', 'HIGHER_CACHE_SIZE', 'STANDARD_PAGE_SIZE',
             'LOWER_PAGE_SIZE', 'HIGHER_PAGE_SIZE', 'HIGHEST_PAGE_SIZE', 'SECURE_DELETE_TRUE',
             'SECURE_DELETE_FALSE', 'SECURE_DELETE_FAST', 'TEMP_STORE_DEFAULT',
             'TEMP_STORE_FILE', 'TEMP_STORE_MEMORY', 'SHARED_CACHE_TRUE', 'SHARED_CACHE_FALSE',
             'READ_UNCOMMITED_TRUE',
             'READ_UNCOMMITED_FALSE', 'FULLSYNC_TRUE', 'FULLSYNC_FALSE', 'TRANSACTION_MODE_DEFERRED',
             'TRANSACTION_MODE_IMMEDIATE',
             'TRANSACTION_MODE_EXCLUSIVE', 'PERF']

    content.writerow(name_list)
    file1.close()

    for _ in trange(size):
        if CNF_flag:
            file1 = open(
                PATH + 'data/' + "random_sqlite_CNF_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                       timestruct) + ".csv",
                "a+", newline="")
        else:
            file1 = open(
                PATH + 'data/' + "random_sqlite_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                   timestruct) + ".csv",
                "a+", newline="")
        content = csv.writer(file1)

        params_list = []
        for i in range(len(bound)):
            value = random.randint(bound[i][0], bound[i][1])
            params[name_list[i]] = value
            params_list.append(value)

        flag = True
        if CNF_flag:
            model = dict()
            for i in range(1, k + 1):
                if params_list[i - 1] >= threshold[i - 1]:
                    model[i] = True
                else:
                    model[i] = False
            flag = cnf.verification(clauses, model)
        if flag:
            pref = sqlite(params_list)
        else:
            pref = -1

        content.writerow(np.append(list(params.values()), pref))
        print([pref], list(params.values()))
        file1.close()

if __name__ == "__main__":

    for i in range(1,4):
        random.seed(i)
        Random_sqlite(size = 100, tag = i, CNF_flag=True)
    for i in range(1,4):
        random.seed(i)
        Random_sqlite(size = 200, tag = i, CNF_flag=True)
    for i in range(1,4):
        random.seed(i)
        Random_sqlite(size = 300, tag = i, CNF_flag=True)
