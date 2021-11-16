from spark.benchmark.get_spark_Performance import get_performance as spark
from Hadoop.get_hadoop_performance import get_performance as Hadoop
from redis.get_redis_Performance import get_3Times as redis
from mysql.get_mysql_Performance import get_performance as mysql
import csv
import random
import numpy as np
from tqdm import trange
import time

PATH = 'mysql/'
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

def Random_mysql(size = 100, tag = 1):
    params = {}
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    file1 = open(
        PATH + 'data/' + "random_mysql_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                      timestruct) + ".csv",
        "a+", newline="")
    content = csv.writer(file1)

    # title
    name_list = ['innodb_buffer_pool_size', 'innodb_thread_concurrency', 'innodb_adaptive_hash_index',
                 'innodb_flush_log_at_trx_commit', 'join_buffer_size', 'sort_buffer_size', 'tmp_table_size',
                 'thread_cache_size', 'read_rnd_buffer_size', 'max_heap_table_size', 'PERF']
    content.writerow(name_list)
    file1.close()

    for _ in trange(size):
        file1 = open(PATH + 'data/' + "random_mysql_" + str(size) + "_" + str(tag) + "_" + time.strftime('%Y%m%d%H%M%S',
                                                                                                      timestruct) + ".csv",
                     "a+", newline="")
        content = csv.writer(file1)
        params[name_list[0]] = random.randint(262144, 3170304)
        params[name_list[1]] = random.randint(0, 50)
        params[name_list[2]] = random.sample([0, 1], 1)[0]
        params[name_list[3]] = random.sample([0, 1, 2], 1)[0]
        params[name_list[4]] = random.randint(131072, 150994944)
        params[name_list[5]] = random.randint(131072, 150994944)
        params[name_list[6]] = random.randint(131072, 150994944)
        params[name_list[7]] = random.randint(0, 50)
        params[name_list[8]] = random.randint(0, 131072)
        params[name_list[9]] = random.randint(262144, 67108864)

        pref = mysql(params)
        content.writerow(np.append(list(params.values()), pref))
        print([pref], list(params.values()))
        file1.close()

if __name__ == "__main__":

    for i in range(1,4):
        Random_mysql(size = 100, tag = i)
    for i in range(1,4):
        Random_mysql(size = 200, tag = i)
    for i in range(1,4):
        Random_mysql(size = 300, tag = i)

    # Random_Hadoop(size=222, tag=3)