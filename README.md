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

# Data
We conduct experiments and obtain data on two cloud clusters and a [cloud server](https://www.aliyun.com/), where each cluster has consisted of three servers, and each server has four Intel![](http://latex.codecogs.com/svg.latex?%5CcircledR) Core![](http://latex.codecogs.com/svg.latex?^%5Ctext{TM})Xeon CPU @ 2.50GHz and 8 GB RAM. The remaining one server is equipped with two Intel![](http://latex.codecogs.com/svg.latex?%5CcircledR) Core![](http://latex.codecogs.com/svg.latex?^%5Ctext{TM})Xeon CPU @ 2.50GHz and 4 GB RAM. CDAT has been evluated on 8 real-world configurable software system:

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
            <td align="center">Hibench</td>
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
            <td align="center">1/27/0</td>
            <td align="center">max</td>
        </tr>
    </tbody>
</table>

Specifically, the configuration options for each software system are selected as:

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
            <td>int</td>
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
            <td>int</td>
            <td>true/false [false]</td>
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
            <td>int</td>
            <td>true/false [false]</td>
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
            <td>int</td>
            <td>true/false [false]</td>
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
            <td>int</td>
            <td>true/false [false]</td>
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
            <td>int</td>
            <td>true/false [false]</td>
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
            <td>int</td>
            <td>true/false [false]</td>
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
            <td>int</td>
            <td>true/false [false]</td>
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

Specifically, for target software systems, CDAT will run with three different sample sizes: 100，200，300, and 3 experiments for each sample size. For example, if users want to evaluate DeepPerf with the system Hadoop with workload Sort, the  modification of lines 35-37 will be:
```
SYSTEM = 'Hadoop'
PATH = 'CDAT/' + SYSTEM + '/'
WORKLOAD = '_Sort'
```
After completing each sample size, the script will output a .csv file in __TargetSystem__/data/ showing the 10 CDAT recommended configurations and measured performance.

The time cost of tuning for each experiment ranges from 2-20 minutes depending on the software system, the sample size, and the user's CPU. Typically, the time cost will be smaller when the software system has a smaller number of configurations or when the sample size is small. Therefore, please be aware that for each sample size, the time cost of evaluating 3 experiments ranges from 0.1 hours to 1 hour. 

# Experimental Results
To evaluate the performance improvement, we use the ![](http://latex.codecogs.com/svg.latex?Impro), which is computed as,
![](http://latex.codecogs.com/svg.latex?{%5Crm{Impro}}(SUT,W)=%5Cfrac{P(C_o|SUT,W)-P(C_d|SUT,W)}{P(C_d|SUT,W)}%5Ctimes{100\%})

where $SUT$ is system under tune, W is workload, $C_o$ is the optimal configuration generated by tuning methods, and $C_d$ is the default configuration. 

In the table below, we use three different measurement constraints (i.e., 100, 200, 300) to evaluate the impact of measurement effort.  The results are obtained when evaluating CDAT on a Windows 10 computer with Intel![](http://latex.codecogs.com/svg.latex?%5CcircledR) Core![](http://latex.codecogs.com/svg.latex?^%5Ctext{TM})i7-8700 CPU @ 3.20GHz 16GB RAM.

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
</table>