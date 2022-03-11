# CDAT (Configuration Distribution Aware-based Tuning)
Many modern software systems provide numerous configuration options that users can adjust for specific running environments. However, it is always an undue burden on users to configure such a system because of the complex impact of the configuration on the system performance and the lack of understanding of the system. To address this issue, various tuning approaches have been proposed to automatically find the optimal configuration by directly using some search strategies or learning a surrogate model. The essential difference among these methods is the way to exploit and explore the underlying distribution of configuration space. The key idea of CDAT is to **automatically capture the distribution of the optimal configurations and generate the optimal configuration based on such distribution**.  Specifically, CDAT consists of three main steps:
- Step 1: Construct a distribution model to convert the configuration to its corresponding distribution feature
- Step 2: Learn a comparison-based model to find the distribution of better configurations
- Step 3: Generate the potentially promising configurations based on the distribution from Step 2.

# Prerequisites
- Python 3.x
- numpy 1.19.2
- pandas 1.1.5

# Installation
CDAT can be directly executed through source code:
1. Download and install [Python 3.x](https://www.python.org/downloads/).

2. Install numpy

   ``` $ pip install numpy```

3. Install pandas

   ``` $ pip install pandas```

4. Clone CDAT (*Unpublished during double-blind review process*).

   ``` $ clone http://github.com/_____/CDTA.git```

# Subject Systems
We conduct experiments and obtain data on two cloud clusters and a [cloud server](https://www.aliyun.com/), where each cluster has consisted of three servers, and each server has four Intel![](http://latex.codecogs.com/svg.latex?%5CcircledR) Core![](http://latex.codecogs.com/svg.latex?^%5Ctext{TM}) Xeon CPU @ 2.50GHz and 8 GB RAM. The remaining one server is equipped with two Intel![](http://latex.codecogs.com/svg.latex?%5CcircledR) Core![](http://latex.codecogs.com/svg.latex?^%5Ctext{TM})Xeon CPU @ 2.50GHz and 4 GB RAM. CDAT has been evluated on 8 real-world configurable software system:

<table>
    <thead>
        <tr>
            <th align="center">System Under Tune</th>
            <th align="center">Domain</th>
            <th align="center">Benchmark</th>
            <th align="center">Performance</th>
            <th align="center">#tuned/ #all options</th>
            <th align="center">#binary/ #numeric/ #enumerated options</th>
            <th align="center">Tuning Goal</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Hadoop</td>
            <td align="center">Data analysis</td>
            <td align="center">Hibench</td>
            <td align="center">Throughput</td>
            <td align="center">9/316</td>
            <td align="center">2/7/0</td>
            <td align="center">max</td>
        </tr>
        <tr>
            <td align="center">Spark</td>
            <td align="center">Data analysis</td>
            <td align="center">Hibench</td>
            <td align="center">Throughput</td>
            <td align="center">13/89</td>
            <td align="center">5/5/3</td>
            <td align="center">max</td>
        </tr>
        <tr>
            <td align="center">SQLite</td>
            <td align="center">Database</td>
            <td align="center">-</td>
            <td align="center">Throughput</td>
            <td align="center">22/78</td>
            <td align="center">22/0/0</td>
            <td align="center">max</td>
        </tr>
        <tr>
            <td align="center">Redis</td>
            <td align="center">Database</td>
            <td align="center">Redis-Bench</td>
            <td align="center">Requests per second</td>
            <td align="center">9/83</td>
            <td align="center">1/7/1</td>
            <td align="center">max</td>
        </tr>
        <tr>
            <td align="center">Tomcat</td>
            <td align="center">Web server</td>
            <td align="center">Apache-Bench</td>
            <td align="center">Requests per second</td>
            <td align="center">12/69</td>
            <td align="center">0/11/1</td>
            <td align="center">max</td>
        </tr>
        <tr>
            <td align="center">Apache</td>
            <td align="center">Web server</td>
            <td align="center">Apache-Bench</td>
            <td align="center">Requests per second</td>
            <td align="center">5/29</td>
            <td align="center">0/5/0</td>
            <td align="center">max</td>
        </tr>
        <tr>
            <td align="center">x264</td>
            <td align="center">Video encoder</td>
            <td align="center">-</td>
            <td align="center">Coding efficiency</td>
            <td align="center">9/140</td>
            <td align="center">1/7/1</td>
            <td align="center">max</td>
        </tr>
        <tr>
            <td align="center">Cassandra</td>
            <td align="center">Database</td>
            <td align="center">YCSB</td>
            <td align="center">Operations per second</td>
            <td align="center">28/59</td>
            <td align="center">0/28/0</td>
            <td align="center">max</td>
        </tr>
    </tbody>
</table>


Specifically, the configuration options for each subject system are selected as:

- [Hadoop 3.1.1](https://hadoop.apache.org/)
  <table>
    <thead>
        <tr>
            <th>Configuration option</th>
            <th>Type</th>
            <th>Range [Default]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mapreduce_task_io_sort_factor</td>
            <td>int</td>
            <td>[10,100] [10]</td>
        </tr>
        <tr>
            <td>mapreduce_reduce_shuffle_merge_percent</td>
            <td>float</td>
            <td>[0.21,0.9] [0.66]</td>
        </tr>
        <tr>
            <td>mapreduce_output_fileoutputformat_compress</td>
            <td>binary</td>
            <td>true/false [false]</td>
        </tr>
        <tr>
            <td>mapreduce_reduce_merge_inmem_threshold</td>
            <td>int</td>
            <td>[10,1000] [1000]</td>
        </tr>
        <tr>
            <td>mapreduce_job_reduces</td>
            <td>int</td>
            <td>[1,1000] [1]</td>
        </tr>
        <tr>
            <td>mapreduce_map_sort_spill_percent</td>
            <td>float</td>
            <td>[0.5,0.9] [0.8]</td>
        </tr>
        <tr>
            <td>mapreduce_reduce_shuffle_input_buffer_percent</td>
            <td>float</td>
            <td>[0.1,0.8] [0.7]</td>
        </tr>
        <tr>
            <td>mapreduce_task_io_sort_mb</td>
            <td>int</td>
            <td>[100,260] [100]</td>
        </tr>
        <tr>
            <td>mapreduce_map_output_compress</td>
            <td>binary</td>
            <td>true/false [false]</td>
        </tr>
    </tbody>
  </table>
- [Spark 2.4.7](https://Spark.apache.org/)
  <table>
    <thead>
        <tr>
            <th>Configuration option</th>
            <th>Type</th>
            <th>Range [Default]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>executorCores</td>
            <td>enum</td>
            <td>{1,2,3,4} [1]</td>
        </tr>
        <tr>
            <td>executorMemory</td>
            <td>int</td>
            <td>[1024,4096] [1024]</td>
        </tr>
        <tr>
            <td>memoryFraction</td>
            <td>float</td>
            <td>[0.1,0.9] [0.6]</td>
        </tr>
        <tr>
            <td>memoryStorageFraction</td>
            <td>float</td>
            <td>[0.1,0.9] [0.6]</td>
        </tr>
        <tr>
            <td>defaultParallelism</td>
            <td>int</td>
            <td>[1,12] [2]</td>
        </tr>
        <tr>
            <td>shuffleCompress</td>
            <td>binary</td>
            <td>true/false [false]</td>
        </tr>
        <tr>
            <td>shuffleSpillCompress</td>
            <td>binary</td>
            <td>true/false [false]</td>
        </tr>
        <tr>
            <td>broadcastCompress</td>
            <td>binary</td>
            <td>true/false [false]</td>
        </tr>
        <tr>
            <td>rddCompress</td>
            <td>binary</td>
            <td>true/false [false]</td>
        </tr>
        <tr>
            <td>ioCompressionCodec</td>
            <td>enum</td>
            <td>{lz4, lzf, snappy} [snappy]</td>
        </tr>
        <tr>
            <td>reducerMaxSizeInFlight</td>
            <td>int</td>
            <td>[8,96] [48]</td>
        </tr>
        <tr>
            <td>shuffleFileBuffer</td>
            <td>int</td>
            <td>[8,64] [32]</td>
        </tr>
        <tr>
            <td>serializer</td>
            <td>enum</td>
            <td>{org.apache.spark.serializer.JavaSerializer, org.apache.spark.serializer.KryoSerializer]} [org.apache.spark.serializer.JavaSerializer]</td>
        </tr>
    </tbody>
  </table>
- [SQLite 3.36.0](https://www.sqlite.org/)
  <table>
    <thead>
        <tr>
            <th>Configuration option</th>
            <th>Type</th>
            <th>Range [Default]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>STANDARD_CACHE_SIZE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>LOWER_CACHE_SIZE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>HIGHER_CACHE_SIZE</td>
            <td>binary</td>
            <td>1/0 [1]</td>
        </tr>
        <tr>
            <td>STANDARD_PAGE_SIZE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>LOWER_PAGE_SIZE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>HIGHER_PAGE_SIZE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>HIGHEST_PAGE_SIZE</td>
            <td>binary</td>
            <td>1/0 [1]</td>
        </tr>
        <tr>
            <td>SECURE_DELETE_TRUE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>SECURE_DELETE_FALSE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>SECURE_DELETE_FAST</td>
            <td>binary</td>
            <td>1/0 [1]</td>
        </tr>
        <tr>
            <td>TEMP_STORE_DEFAULT</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>TEMP_STORE_FILE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>TEMP_STORE_MEMORY</td>
            <td>binary</td>
            <td>1/0 [1]</td>
        </tr>
        <tr>
            <td>SHARED_CACHE_TRUE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>SHARED_CACHE_FALSE</td>
            <td>binary</td>
            <td>1/0 [1]</td>
        </tr>
        <tr>
            <td>READ_UNCOMMITED_TRUE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>READ_UNCOMMITED_FALSE</td>
            <td>binary</td>
            <td>1/0 [1]</td>
        </tr>
        <tr>
            <td>FULLSYNC_TRUE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>FULLSYNC_FALSE</td>
            <td>binary</td>
            <td>1/0 [1]</td>
        </tr>
        <tr>
            <td>TRANSACTION_MODE_DEFERRED</td>
            <td>binary</td>
            <td>1/0 [1]</td>
        </tr>
        <tr>
            <td>TRANSACTION_MODE_IMMEDIATE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
        <tr>
            <td>FTRANSACTION_MODE_EXCLUSIVE</td>
            <td>binary</td>
            <td>1/0 [0]</td>
        </tr>
    </tbody>
  </table>
- [Redis 4.0.2](https://redis.io/)
  <table>
    <thead>
        <tr>
            <th>Configuration option</th>
            <th>Type</th>
            <th>Range [Default]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>replBacklogSize</td>
            <td>int</td>
            <td>[1,11] [10]</td>
        </tr>
        <tr>
            <td>hashMaxZiplistValue</td>
            <td>int</td>
            <td>[32,128] [60]</td>
        </tr>
        <tr>
            <td>hashMaxZiplistEntries</td>
            <td>int</td>
            <td>[256,1024] [862]</td>
        </tr>
        <tr>
            <td>listMaxZiplistSize</td>
            <td>enum</td>
            <td>{-5,-4,-3,-2,-1} [-3]</td>
        </tr>
        <tr>
            <td>activeDefragIgnoreBytes</td>
            <td>int</td>
            <td>[100,300] [162]</td>
        </tr>
        <tr>
            <td>activeDefragThresholdLower</td>
            <td>int</td>
            <td>[5,20] [15]</td>
        </tr>
        <tr>
            <td>replDisableTcpNodelay</td>
            <td>binary</td>
            <td>no/yes [no]</td>
        </tr>
        <tr>
            <td>hllSparseMaxBytes</td>
            <td>int</td>
            <td>[0,5000] [1139]</td>
        </tr>
        <tr>
            <td>hz</td>
            <td>int</td>
            <td>[1,501] [22]</td>
        </tr>
    </tbody>
  </table>
- [Tomcat 9.0.48](https://tomcat.apache.org/)
  <table>
    <thead>
        <tr>
            <th>Configuration option</th>
            <th>Type</th>
            <th>Range [Default]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>maxThreads</td>
            <td>int</td>
            <td>[1,500] [200]</td>
        </tr>
        <tr>
            <td>minSpareThreads</td>
            <td>int</td>
            <td>[1,50] [25]</td>
        </tr>
        <tr>
            <td>executorTerminationTimeoutMillis</td>
            <td>int</td>
            <td>[0,8000] [5000]</td>
        </tr>
        <tr>
            <td>connectionTimeout</td>
            <td>int</td>
            <td>[0,50000] [30000]</td>
        </tr>
        <tr>
            <td>maxConnections</td>
            <td>int</td>
            <td>[1,50000] [20000]</td>
        </tr>
        <tr>
            <td>maxKeepAliveRequests</td>
            <td>int</td>
            <td>[1,200] [100]</td>
        </tr>
        <tr>
            <td>acceptorThreadCount</td>
            <td>enum</td>
            <td>{1,2,3,4,5,6,7,8,9,10} [1]</td>
        </tr>
        <tr>
            <td>asyncTimeout</td>
            <td>int</td>
            <td>[1,50000] [30000]</td>
        </tr>
        <tr>
            <td>acceptCount</td>
            <td>int</td>
            <td>[1,50] [10]</td>
        </tr>
        <tr>
            <td>socketBuffer</td>
            <td>int</td>
            <td>[1,500] [100]</td>
        </tr>
        <tr>
            <td>processorCache</td>
            <td>int</td>
            <td>[1,500] [200]</td>
        </tr>
        <tr>
            <td>keepAliveTimeout</td>
            <td>int</td>
            <td>[1,50] [15]</td>
        </tr>
    </tbody>
  </table>
- [Apache 2.4.46](https://httpd.apache.org/)
  <table>
    <thead>
        <tr>
            <th>Configuration option</th>
            <th>Type</th>
            <th>Range [Default]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>StartServers</td>
            <td>int</td>
            <td>[1,10] [7]</td>
        </tr>
        <tr>
            <td>MinSpareServers</td>
            <td>int</td>
            <td>[1,10] [7]</td>
        </tr>
        <tr>
            <td>MaxSpareServers</td>
            <td>int</td>
            <td>[11,20] [12]</td>
        </tr>
        <tr>
            <td>MaxRequestWorkers</td>
            <td>int</td>
            <td>[1,1000] [252]</td>
        </tr>
        <tr>
            <td>MaxRequestsPerChild</td>
            <td>int</td>
            <td>[0,1000] [0]</td>
        </tr>
    </tbody>
  </table>
- [x264 r2334](https://www.videolan.org/)
  <table>
    <thead>
        <tr>
            <th>Configuration option</th>
            <th>Type</th>
            <th>Range [Default]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>no-8x8dct</td>
            <td>binary</td>
            <td>0/1 [0]</td>
        </tr>
        <tr>
            <td>no-cabac</td>
            <td>binary</td>
            <td>0/1 [0]</td>
        </tr>
        <tr>
            <td>no-deblock</td>
            <td>binary</td>
            <td>0/1 [0]</td>
        </tr>
        <tr>
            <td>no-fast-pskip</td>
            <td>binary</td>
            <td>0/1 [0]</td>
        </tr>
        <tr>
            <td>no-mbtree</td>
            <td>binary</td>
            <td>0/1 [0]</td>
        </tr>
        <tr>
            <td>no-mixed-refs</td>
            <td>binary</td>
            <td>0/1 [0]</td>
        </tr>
        <tr>
            <td>no-weightb</td>
            <td>binary</td>
            <td>0/1 [0]</td>
        </tr>
        <tr>
            <td>rc-lookahead</td>
            <td>int</td>
            <td>[40,250] [40]</td>
        </tr>
        <tr>
            <td>ref</td>
            <td>int</td>
            <td>[1,9] [1]</td>
        </tr>
    </tbody>
  </table>
- [Cassandra 3.11.6](https://cassandra.apache.org/)
  <table>
    <thead>
        <tr>
            <th>Configuration option</th>
            <th>Type</th>
            <th>Range [Default]</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>write_request_timeout_in_ms</td>
            <td>int</td>
            <td>[20,250000] [2000]</td>
        </tr>
        <tr>
            <td>read_request_timeout_in_ms</td>
            <td>int</td>
            <td>[20,250000] [5000]</td>
        </tr>
        <tr>
            <td>commitlog_total_space_in_mb</td>
            <td>int</td>
            <td>[200, 15000] [8192]</td>
        </tr>
        <tr>
            <td>key_cache_size_in_mb</td>
            <td>int</td>
            <td>[100, 15000] [100]</td>
        </tr>
        <tr>
            <td>commitlog_segment_size_in_mb</td>
            <td>int</td>
            <td>[30, 2024] [32]</td>
        </tr>
        <tr>
            <td>dynamic_snitch_badness_threshold</td>
            <td>float</td>
            <td>[0,1] [0.1]</td>
        </tr>
        <tr>
            <td>index_summary_capacity_in_mb</td>
            <td>int</td>
            <td>[100, 15000] [100]</td>
        </tr>
        <tr>
            <td>key_cache_save_period</td>
            <td>int</td>
            <td>[10, 14400] [14400]</td>
        </tr>
        <tr>
            <td>file_cache_size_in_mb</td>
            <td>int</td>
            <td>[250, 15000] [512]</td>
        </tr>
        <tr>
            <td>thrift_framed_transport_size_in_mb</td>
            <td>int</td>
            <td>[2, 28] [15]</td>
        </tr>
        <tr>
            <td>memtable_heap_space_in_mb</td>
            <td>int</td>
            <td>[80, 15000] [2048]</td>
        </tr>
        <tr>
            <td>concurrent_writes</td>
            <td>int</td>
            <td>[20, 3800] [32]</td>
        </tr>
        <tr>
            <td>index_summary_resize_interval_in_minutes</td>
            <td>int</td>
            <td>[2, 60] [60]</td>
        </tr>
        <tr>
            <td>commitlog_sync_period_in_ms</td>
            <td>int</td>
            <td>[2200, 250000] [10000]</td>
        </tr>
        <tr>
            <td>range_request_timeout_in_ms</td>
            <td>int</td>
            <td>[2000, 250000] [10000]</td>
        </tr>
        <tr>
            <td>rpc_min_threads</td>
            <td>int</td>
            <td>[10, 1000] [16]</td>
        </tr>
        <tr>
            <td>batch_size_warn_threshold_in_kb</td>
            <td>int</td>
            <td>[5, 100000] [5]</td>
        </tr>
        <tr>
            <td>concurrent_reads</td>
            <td>int</td>
            <td>[31, 2000] [32]</td>
        </tr>
        <tr>
            <td>column_index_size_in_kb</td>
            <td>int</td>
            <td>[64, 60000] [64]</td>
        </tr>
        <tr>
            <td>dynamic_snitch_update_interval_in_ms</td>
            <td>int</td>
            <td>[100, 25000] [100]</td>
        </tr>
        <tr>
            <td>memtable_flush_writers</td>
            <td>int</td>
            <td>[2, 28] [2]</td>
        </tr>
        <tr>
            <td>request_timeout_in_ms</td>
            <td>int</td>
            <td>[1000, 25000] [10000]</td>
        </tr>
        <tr>
            <td>cas_contention_timeout_in_ms</td>
            <td>int</td>
            <td>[900, 25000] [1000]</td>
        </tr>
        <tr>
            <td>permissions_validity_in_ms</td>
            <td>int</td>
            <td>[1600, 25000] [2000]</td>
        </tr>
        <tr>
            <td>rpc_max_threads</td>
            <td>int</td>
            <td>[1048, 3800] [2048]</td>
        </tr>
        <tr>
            <td>truncate_request_timeout_in_ms</td>
            <td>int</td>
            <td>[2000, 250000] [60000]</td>
        </tr>
        <tr>
            <td>stream_throughput_outbound_megabits_per_sec</td>
            <td>int</td>
            <td>[100, 8000] [200]</td>
        </tr>
        <tr>
            <td>memtable_offheap_space_in_mb</td>
            <td>int</td>
            <td>[200, 15000] [2048]</td>
        </tr>
    </tbody>
  </table>

# Usage

To run CDAT, users need to prepare before the evaluation and then run the script `main.py`. For details, users can refer to the experimental setup of __sqlite__ (__TargetSystem__), including the following:

- Create folders: __TargetSystem__ /data/

- Prepare the data: __TargetSystem__ /__TargetSystem__.csv

- Prepare the function: get_ __TargetSystem__ _Performance, the input is configuration, and the output is performance.

- Modify experimental parameters in line 35-37.

- If users want to add other systems, add in the script `main.py` (located at line41, line241 and line 247).

Specifically, for target software systems, CDAT will run with three different sample sizes: 100，200，300, and 3 experiments for each sample size. For example, if users want to evaluate CDAT with the system Hadoop with workload Sort, the  modification of lines 35-37 will be:
```
SYSTEM = 'Hadoop'
PATH = 'CDAT/' + SYSTEM + '/'
WORKLOAD = '_Sort'
```
After completing each sample size, the script will output a .csv file in __TargetSystem__/data/ showing the 10 CDAT recommended configurations with measured performance.

The time cost of tuning for each experiment ranges from 2-20 minutes depending on the software system, the sample size, and the user's CPU. Typically, the time cost will be smaller when the software system has a smaller number of configurations or when the sample size is small. Therefore, please be aware that for each sample size, the time cost of evaluating 3 experiments ranges from 0.1 to 1 hour. 

# Experimental Results
To evaluate the performance improvement, we use the ![](http://latex.codecogs.com/svg.latex?%5Crm{Impro}), which is computed as,

![](http://latex.codecogs.com/svg.latex?{%5Crm{Impro}}(SUT,W)=%5Cfrac{P(C_o|SUT,W)-P(C_d|SUT,W)}{P(C_d|SUT,W)}%5Ctimes{100\%})

where![](http://latex.codecogs.com/svg.latex?SUT) is system under tune, W is workload, ![](http://latex.codecogs.com/svg.latex?C_o) is the optimal configuration generated by tuning methods, and ![](http://latex.codecogs.com/svg.latex?C_d) is the default configuration. 

In the table below, we use three different measurement constraints (i.e., 100, 200, 300) to evaluate the impact of measurement effort.  The results are obtained when evaluating CDAT on a Windows 10 computer with Intel![](http://latex.codecogs.com/svg.latex?%5CcircledR) Core![](http://latex.codecogs.com/svg.latex?^%5Ctext{TM}) i7-8700 CPU @ 3.20GHz 16GB RAM.

<table>
    <thead>
        <tr>
            <th rowspan="2" colspan="2" >System Under Tune (Workload)</th>
            <th rowspan="2" >Measurement constraints</th>
            <th colspan="6" >Turn approach</th>
        </tr>
        <tr>
            <th scope="col">Random</th>
            <th scope="col">ACTGAN</th>
            <th scope="col">Hyperopt</th>
            <th scope="col">BestConfig</th>
            <th scope="col">RHFOC</th>
            <th scope="col">CDAT</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="9" align="center">Spark</td>
            <td rowspan="3">Wordcount</td>
            <td>100</td>
            <td><strong>48.8%</strong></td>
            <td>27.7%</td>
            <td>25.6%</td>
            <td>21.4%</td>
            <td>8.5%</td>
            <td>48.2%</td>
        </tr>
        <tr>
            <td>200</td>
            <td>53.6%</td>
            <td>26.4%</td>
            <td>37.5%</td>
            <td>26.8%</td>
            <td>8.8%</td>
            <td><strong>58.3%</strong></td>
        </tr>
        <tr>
            <td>300</td>
            <td>55.0%</td>
            <td>22.0%</td>
            <td>41.3%</td>
            <td>30.6%</td>
            <td>0.6%</td>
            <td><strong>59.7%</strong></td>
        </tr>
        <tr>
            <td rowspan="3">Sort</td>
            <td>100</td>     
            <td>60.9%</td>
            <td>34.2%</td>
            <td>55.0%</td>
            <td>49.6%</td>
            <td>39.1%</td>
            <td><strong>62.5%</strong></td>
        </tr>
        <tr>
            <td>200</td>    
            <td>67.2%</td>
            <td>46.9%</td>
            <td>61.1%</td>
            <td>63.4%</td>
            <td>41.0%</td>
            <td><strong>72.6%</strong></td>
        </tr>
        <tr>
            <td>300</td>     
            <td>69.1%</td>
            <td>45.5%</td>
            <td>69.5%</td>
            <td>30.6%</td>
            <td>38.9%</td>
            <td><strong>76.7%</strong></td>
        </tr>
        <tr>
            <td rowspan="3">Terasort</td>
            <td>100</td> 
            <td>83.1%</td>
            <td>60.2%</td>
            <td>82.8%</td>
            <td>87.3%</td>
            <td>63.1%</td>
            <td><strong>103.6%</strong></td>
        </tr>
        <tr>
            <td>200</td>     
            <td>94.0%</td>
            <td>67.4%</td>
            <td>102.9%</td>
            <td>81.2%</td>
            <td>65.8%</td>
            <td><strong>110.9%</strong></td>
        </tr>
        <tr>
            <td>300</td>    
            <td>105.0%</td>
            <td>72.2%</td>
            <td>98.2%</td>
            <td>92.1%</td>
            <td>68.9%</td>
            <td><strong>113.4%</strong></td>
        </tr>
        <tr>
            <td rowspan="9">Hadoop</td>
            <td rowspan="3">Wordcount</td>
            <td>100</td>     
            <td><strong>5.6%</strong></td>
            <td>1.0%</td>
            <td>4.3%</td>
            <td>2.4%</td>
            <td>-0.9%</td>
            <td>5.0%</td>
        </tr>
        <tr>
            <td>200</td>     
            <td>7.5%</td>
            <td>0.4%</td>
            <td>4.5%</td>
            <td>3.1%</td>
            <td>-5.3%</td>
            <td><strong>8.7%</strong></td>
        </tr>
        <tr>
            <td>300</td>     
            <td>4.0%</td>
            <td>0.9%</td>
            <td>46.1%</td>
            <td>2.3%</td>
            <td>-5.9%</td>
            <td><strong>8.8%</strong></td>
        </tr>
        <tr>
            <td rowspan="3">Sort</td>
            <td>100</td>     
            <td><strong>7.0%</strong></td>
            <td>5.0%</td>
            <td>5.0%</td>
            <td>2.7%</td>
            <td>0.2%</td>
            <td>6.7%</td>
        </tr>
        <tr>     
            <td>200</td>
            <td>9.9%</td>
            <td>6.6%</td>
            <td>5.5%</td>
            <td>4.3%</td>
            <td>2.8%</td>
            <td><strong>10.1%</strong></td>
        </tr>
        <tr>     
            <td>300</td>
            <td>10.8%</td>
            <td>6.4%</td>
            <td>4.2%</td>
            <td>5.9%</td>
            <td>2.8%</td>
            <td><strong>11.2%</strong></td>
        </tr>
        <tr>
            <td rowspan="3">Terasort</td>
            <td>100</td>     
            <td>15.0%</td>
            <td>8.4%</td>
            <td>11.5%</td>
            <td>10.0%</td>
            <td>4.8%</td>
            <td><strong>16.4%</strong></td>
        </tr>
        <tr>
            <td>200</td>   
            <td>18.0%</td>
            <td>8.6%</td>
            <td>12.2%</td>
            <td>11.3%</td>
            <td>5.6%</td>
            <td><strong>16.8%</strong></td>
        </tr>
        <tr>
            <td>300</td>     
            <td>16.9%</td>
            <td>7.2%</td>
            <td>15.5%</td>
            <td>12.3%</td>
            <td>6.8%</td>
            <td><strong>19.0%</strong></td>
        </tr>
        <tr>
            <td rowspan="3" colspan="2"  align="center">Cassandra</td>
            <td>100</td>     
            <td>21.2%</td>
            <td>19.7%</td>
            <td><strong>23.2%</strong></td>
            <td>20.4%</td>
            <td>16.7%</td>
            <td>22.4%</td>
        </tr>
        <tr>     
            <td>200</td>
            <td>21.7%</td>
            <td>23.4%</td>
            <td>23.1%</td>
            <td>20.8%</td>
            <td>14.0%</td>
            <td><strong>116.7%</strong></td>
        </tr>
        <tr>     
            <td>300</td>
            <td>23.0%</td>
            <td>22.3%</td>
            <td>24.3%</td>
            <td>23.0%</td>
            <td>13.5%</td>
            <td><strong>183.3%</strong></td>
        </tr>
        <tr>
            <td rowspan="3" colspan="2"  align="center">Redis</td>
            <td>100</td>     
            <td>35.4%</td>
            <td>9.1%</td>
            <td>23.0%</td>
            <td>22.3%</td>
            <td>2.9%</td>
            <td><strong>40.4%</strong></td>
        </tr>
        <tr>
            <td>200</td>     
            <td>34.2%</td>
            <td>9.6%</td>
            <td>22.2%</td>
            <td>30.2%</td>
            <td>6.0%</td>
            <td><strong>40.1%</strong></td>
        </tr>
        <tr>
            <td>300</td>     
            <td>42.8%</td>
            <td>19.0%</td>
            <td>33.6%</td>
            <td>29.4%</td>
            <td>8.3%</td>
            <td><strong>46.0%</strong></td>
        </tr>
        <tr>
            <td rowspan="3" colspan="2"  align="center">SQLite</td>
            <td>100</td>     
            <td>16.4%</td>
            <td>6.6%</td>
            <td>13.9%</td>
            <td>14.7%</td>
            <td>16.4%</td>
            <td><strong>20.4%</strong></td>
        </tr>
        <tr>
            <td>200</td>     
            <td>21.7%</td>
            <td>15.9%</td>
            <td>16.0%</td>
            <td>12.9%</td>
            <td>14.1%</td>
            <td><strong>22.7%</strong></td>
        </tr>
        <tr>
            <td>300</td>     
            <td>26.2%</td>
            <td>13.7%</td>
            <td>17.0%</td>
            <td>17.3%</td>
            <td>16.2%</td>
            <td><strong>34.6%</strong></td>
        </tr>
        <tr>
            <td rowspan="3" colspan="2"  align="center">Tomcat</td>
            <td>100</td>     
            <td>26.6%</td>
            <td>36.6%</td>
            <td>18.3%</td>
            <td>46.2%</td>
            <td>8.0%</td>
            <td><strong>58.1%</strong></td>
        </tr>
        <tr>     
            <td>200</td>
            <td>35.0%</td>
            <td>26.8%</td>
            <td>50.4%</td>
            <td>40.7%</td>
            <td>10.9%</td>
            <td><strong>74.7%</strong></td>
        </tr>
        <tr>
            <td>300</td>     
            <td>55.7%</td>
            <td>42.4%</td>
            <td>39.9%</td>
            <td>56.5%</td>
            <td>7.2%</td>
            <td><strong>83.4%</strong></td>
        </tr>
        <tr>
            <td rowspan="3" colspan="2"  align="center">x264</td>
            <td>100</td>     
            <td>86.0%</td>
            <td>55.7%</td>
            <td>82.1%</td>
            <td>78.0%</td>
            <td>86.6%</td>
            <td><strong>88.4%</strong></td>
        </tr>
        <tr>     
            <td>200</td>
            <td>87.9%</td>
            <td>76.7%</td>
            <td>86.6%</td>
            <td>80.9%</td>
            <td>87.9%</td>
            <td><strong>88.9%</strong></td>
        </tr>
        <tr>
            <td>300</td>     
            <td>87.7%</td>
            <td>76.6%</td>
            <td>76.4%</td>
            <td>83.9%</td>
            <td>87.2%</td>
            <td><strong>88.9%</strong></td>
        </tr>
        <tr>
            <td rowspan="3" colspan="2"  align="center">Apache</td>
            <td>100</td>     
            <td>49.4%</td>
            <td>33.6%</td>
            <td>48.3%</td>
            <td>48.5%</td>
            <td>18.3%</td>
            <td><strong>54.2%</strong></td>
        </tr>
        <tr>     
            <td>200</td>
            <td>55.3%</td>
            <td>44.0%</td>
            <td>47.0%</td>
            <td><strong>57.0%</strong></td>
            <td>18.8%</td>
            <td>56.8%</td>
        </tr>
        <tr>
            <td>300</td>     
            <td>57.0%</td>
            <td>33.8%</td>
            <td>49.7%</td>
            <td>50.7%</td>
            <td>16.0%</td>
            <td><strong>61.3%</strong></td>
        </tr>
    </tbody>
</table>