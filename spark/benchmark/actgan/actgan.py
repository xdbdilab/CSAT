# -*- encoding: utf-8 -*-
"""
@File    : actgan.py
@Time    : 2021/1/4 10:13
@Author  : zspp, Wei Shouxin
@Software: PyCharm
"""
import pandas as pd
import tensorflow as tf
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import os
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from nfs.colosseum.getPerformance import get_performance
# from nfs.colosseum.report import *
from getPerformance import get_performance

datadir = os.path.dirname((os.path.abspath(__file__)))
# print("actgan_datadir:" + datadir)
time_dict = {}


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='actgan')
    parser.add_argument('--random_sample_file', type=str, required=True, help='random_sample_file')
    parser.add_argument('--actgan_component_params', type=str, required=True, nargs='+',
                        help='actgan_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    args = parser.parse_args()
    return args


def data_in(data, selected_params, columns, performance, min_or_max):
    if min_or_max == 'max':
        data = data.sort_values(by=performance, ascending=False)
    else:
        data = data.sort_values(by=performance, ascending=True)
    data = data.drop(performance, 1)

    ##首先处理没有优先关系的enum参数，即string参数
    char = []  # enum的列名称
    enum_index = []  # enum的原始列索引
    for name in columns[:-1]:
        if selected_params[name][0] == 'string' or selected_params[name][0] == 'enum':
            char.append(name)
            enum_index.append(columns.index(name))

    enum_number = []  # 每个enum参数对应的独热编码的长度
    enum_book = {}  # 每个enum参数的内容，字典形式存储
    m = 0
    for c in char:
        i = enum_index[m]

        new_data = pd.DataFrame({c: selected_params[c][1]})  # 添加几行，为了更好全面编码
        data = data.append(new_data, ignore_index=True)

        enum_book[c] = list(pd.get_dummies(data[c]).columns)
        enum_data = pd.get_dummies(data[c], prefix=c)  # 独热编码后的变量

        data = data.drop(c, 1)

        enum_list = list(enum_data.columns)
        enum_number.append(len(enum_list))

        for k in range(len(enum_list)):
            data.insert(i + k, enum_list[k], enum_data[enum_list[k]])  # 将向量移动到原来枚举值的位置
        m = m + 1
        enum_index = [j + len(enum_data.columns) - 1 for j in enum_index]  # 更新enum_index

        data.drop(data.index[-len(selected_params[c][1]):], inplace=True)  # 删除前3行
        # print(enum_index)
    # print(enum_number)
    # print(data)
    # print(enum_book)

    ##接着处理有优先关系的参数
    # char2 = []
    # enum_index2 = {}
    # for name in columns[:-1]:
    #     if selected_params[name][0] == 'enum':
    #         char2.append(name)
    #         enum_index2[name] = selected_params[name][1]
    # for enum_name in char2:
    #     size_range = list(range(len(enum_index2[enum_name])))
    #     mapping = dict(zip(enum_index2[enum_name],size_range))
    #     data[enum_name] = data[enum_name].map(mapping)

    return data, enum_number, enum_book

def data_out(data, selected_params, columns, enum_number, enum_book):
    if data.empty:
        return pd.DataFrame(columns=columns[:-1])
    data.columns = [i for i in range(len(data.columns))]
    hang = data.iloc[:, 0].size

    # 将data的数据变为符合的类型格式
    m = 0  # 更改columns之后的columns的索引
    index_enum_number = 0  # enum_number的序号索引
    enum_list = []  # enum的列名称
    for name in columns[:-1]:
        if selected_params[name][0] == 'int':
            data[m] = data[m].astype(np.int64)
            m = m + 1
        elif selected_params[name][0] == 'float' or selected_params[name][0] == 'double':
            m = m + 1
            continue
        # elif selected_params[name][0] =='enum':
        #     string_map = dict(zip(list(range(len(selected_params[name][1]))), selected_params[name][1]))
        #     data[[m]] = data[[m]].astype(np.int64)
        #     data[m] = data[m].map(string_map)
        #     m = m + 1
        else:
            enum_list.append(name)
            for i in range(enum_number[index_enum_number]):
                data[m + i] = data[m + i].round().astype(np.int64)
            m = m + enum_number[index_enum_number]
            index_enum_number = index_enum_number + 1
    # print(data)

    # 在数据尾部加入enum的原始列名称
    for i in enum_list:
        data_temp = pd.DataFrame(columns=[i])
        data = pd.concat([data, data_temp], 1)

    for index in range(hang):
        # 将向量转换成枚举值
        # print(data.iloc[[index]].values[0])
        for k in range(len(enum_list)):

            name_index = columns.index(enum_list[k])  # enum变量的原始列索引
            if k != 0:
                for a in range(k):
                    name_index = name_index + enum_number[a] - 1  # 变为enum改变后的索引
            # print(name_index)

            flag = True
            true_index = False
            first_index = -1
            for i in range(enum_number[k]):

                for j in range(i, enum_number[k]):
                    if i == j:

                        if round(data.loc[[index]].values[0][name_index + j]) == 1:
                            first_index = j
                            continue
                        else:
                            if j == enum_number[k] - 1:
                                flag = False
                            break
                    else:
                        if round(data.loc[[index]].values[0][name_index + j]) == 0:

                            if j == enum_number[k] - 1:
                                true_index = True
                                break
                            else:
                                continue
                        else:
                            flag = False
                            break

                if flag == False:
                    break
                if true_index == True:
                    break
            if flag == False:
                data.drop([index], inplace=True)
                break
            if (first_index + 1) != 0 and flag == True:
                data.loc[index, enum_list[k]] = enum_book[enum_list[k]][first_index]
    # print(data)

    # 删去枚举值所在的列
    number = 0
    for i in range(len(enum_list)):
        name_index = number + columns.index(enum_list[i])

        for j in range(enum_number[i]):
            data = data.drop([name_index + j], 1)

        number = number + enum_number[i] - 1
    # print(data)

    # 移动最后含枚举值的几列,对列进行重命名
    for i in range(len(enum_list)):
        orgin_index = columns.index(enum_list[i])
        data.insert(orgin_index, enum_list[i], data.pop(enum_list[i]))
    # print(data)

    col = {}
    for (key, value) in zip(list(data.columns), columns[:-1]):
        col[key] = value
    # print(col)
    data.rename(columns=col, inplace=True)
    # print(data)
    return data



def data_range(data, selected_params, columns, enum_number):
    data = data.values
    # 参数范围（不包含上下界），最后四个是独热编码后的值，只能为0或1，所以设置为-1和2
    MAX_BOUND = []
    MIN_BOUND = []
    m = 0
    for name in columns[:-1]:
        if selected_params[name][0] == 'enum' or selected_params[name][0] == 'string':
            for i in range(enum_number[m]):
                MIN_BOUND.append(0)
                MAX_BOUND.append(1)
            m = m + 1
        # elif selected_params[name][0] =='string':
        #     MIN_BOUND.append(0)
        #     MAX_BOUND.append(len(selected_params[name][1])-1)
        else:
            MIN_BOUND.append(selected_params[name][1][0])
            MAX_BOUND.append(selected_params[name][1][1])
    MIN_BOUND = [i - 1 for i in MIN_BOUND]
    MAX_BOUND = [i + 1 for i in MAX_BOUND]

    result = []
    for each in data:
        flag = True
        for j in range(len(MIN_BOUND)):
            if each[j] <= MIN_BOUND[j] or each[j] >= MAX_BOUND[j]:
                flag = False
                break
        if flag:
            result.append(each)

    return pd.DataFrame(np.array(result))


def gan(data, Epoch, LR_G, LR_D, N_IDEAS, NumOfLine, BATCH_SIZE):
    # GAN超参数设置
    Epoch = Epoch  # 迭代次数
    NumOfLine = NumOfLine  # 每次取多少个样本进行训练
    BATCH_SIZE = BATCH_SIZE  # 总的训练样本数目
    LR_G = LR_G  # 生成器的学习率
    LR_D = LR_D  # 判别器的学习率
    N_IDEAS = N_IDEAS  # 生成器输入随机噪声的维数

    # 小批量抽取训练样本
    def Cwork():
        clist = random.sample(range(BATCH_SIZE), NumOfLine)  # 从1～BATCH_SIZE中随机取NumOfLine个数
        dataused = np.zeros(shape=(NumOfLine, NumOfF))
        j = 0
        for c in clist:
            # 返回某一行   data[i:i+1]
            dataused[j] = data[c]
            j = j + 1
        return dataused

    # 设置tensorflow和numpy随机种子数,保证每次随机量相同
    tf.set_random_seed(1)
    np.random.seed(1)

    NumOfF = data.columns.size  # 软件系统优化参数的数量
    data = data.iloc[:BATCH_SIZE, :NumOfF]  # 取少量样本用来训练
    data = data.values  # 将数据转换成矩阵

    # 数据标准化,去均值和方差归一化
    ss = StandardScaler()
    data = ss.fit_transform(data.astype(float))

    new_points = np.linspace(1, 10, NumOfF)

    # 重置/清除计算图
    tf.reset_default_graph()

    # 建立一个三层神经网络作为GAN的生成器，输入为随机噪声，输出为NumOfF个参数
    with tf.variable_scope('Generator'):  # 返回一个用于定义创建variable（层）的op的上下文管理器
        G_in = tf.placeholder(tf.float32, [None, N_IDEAS])  # random ideas (could from normal distribution)
        G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)  # 创建一个128个神经元的隐藏层，激励函数为relu
        G_out = tf.layers.dense(G_l1, NumOfF)  # 创建一个NumOfF个神经元的输出层

    # 建立一个三层神经网络作为GAN的判别器，输入为一组参数，输出为0-1的数
    with tf.variable_scope('Discriminator'):
        real_f = tf.placeholder(tf.float32, [None, NumOfF], name='real_in')  # 训练样本
        D_l0 = tf.layers.dense(real_f, 128, tf.nn.relu, name='l')
        p_real = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')  # 样本来自训练集的可能性
        # 重复使用上面的两层l和out
        D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)  # 生成样本
        p_fake = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)  # 样本来自训练集的可能性

    # 定义判别器D的损失函数D_loss和生成器G的损失函数G_loss
    D_loss = -tf.reduce_mean(tf.log(p_real) + tf.log(1 - p_fake))
    G_loss = tf.reduce_mean(tf.log(1 - p_fake))

    # 选择AdamOptimizer优化器来最小化损失函数
    train_D = tf.train.AdamOptimizer(LR_D).minimize(
        D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
    train_G = tf.train.AdamOptimizer(LR_G).minimize(
        G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

    # 开始使用tensorflow训练GAN
    sess = tf.Session()  # 创建tensorflow会话
    sess.run(tf.global_variables_initializer())  # 对上述所有定义的变量进行初始化

    outp = [[0] * NumOfF] * 300  # 输出300个生成值
    i = 0  # 表示第几个生成值
    loss_dict = {'dloss': [], 'gloss': []}
    for step in range(Epoch + 1):  # 进行Epoch次迭代训练
        dataused = Cwork()  # 多次采集不同数量的训练样本
        G_ideas = np.random.randn(NumOfLine, N_IDEAS)  # G的输入是NumOfLine个N_IDEAS维随机噪声
        # 训练GAN并得到相应输出
        results, pa0, Dl, Gl = sess.run([G_out, p_fake, D_loss, G_loss, train_D, train_G],
                                        {G_in: G_ideas, real_f: dataused})[:4]
        if step%2000==0:
            print("判别器损失：", -Dl, "，生成器损失：", Gl,"----step",step)
        loss_dict['dloss'].append(-Dl)
        loss_dict['gloss'].append(Gl)

        if step > (Epoch - 300):  # 将最后300次生成的结果第一条作为最终生成集
            outp[i] = results[0]
            i = i + 1

    outp = ss.inverse_transform(outp)  # 将标准化的数据转换成原本格式
    return pd.DataFrame(outp).dropna(), loss_dict


def actgan(train_data, hyper_params, columns, selected_params, performance, min_or_max,
           actgan_component_params, config_params,sample_num):
    """
    调优算法
    :param columns: 列名
    :param performance: 性能指标名称
    :param min_or_max: 最大化还是最小化
    :param train_data: 训练数据
    :param hyper_params: 超参数设置
    :param actgan_component_params: actgan_component_params
    :param selected_params: 待调整参数名称列表
    :param config_params: config_params
    :return: 最优的一组参数 {param1: value1, param2: value2, ... , perfomance: value}
    """
    actgan_32_start_time = time.time()
    print("开始actgan产生32组样本的时间：", time.ctime(actgan_32_start_time))

    Epoch = hyper_params['Epoch']
    LR_G = hyper_params['LR_G']
    LR_D = hyper_params['LR_D']
    N_IDEAS = hyper_params['N_IDEAS']
    NumOfLine = hyper_params['NumOfLine']
    BATCH_SIZE = hyper_params['BATCH_SIZE']

    newdata, enum_number, enum_book = data_in(train_data.iloc[:-BATCH_SIZE, :], selected_params, columns,
                                              performance, min_or_max)

    gan_result, loss_dict = gan(newdata, Epoch, LR_G, LR_D, N_IDEAS, NumOfLine, BATCH_SIZE)
    print(gan_result)
    gan_result = data_range(gan_result, selected_params, columns, enum_number)
    print(gan_result)
    res = data_out(gan_result, selected_params, columns, enum_number, enum_book)
    print(res)

    if res.empty:
        pd.DataFrame(columns=columns)

    while True:  # 保证生成想要的样本数量
        if res.shape[0] >= BATCH_SIZE:
            res = res.iloc[:BATCH_SIZE, :]
            break
        else:
            gan_results, loss_dict = gan(newdata, Epoch, LR_G, LR_D, N_IDEAS, NumOfLine, BATCH_SIZE)
            gan_results = data_range(gan_results, selected_params, columns, enum_number)
            gan_results = data_out(gan_results, selected_params, columns, enum_number, enum_book)
            if not gan_results.empty:
                res = pd.concat([res, gan_results], axis=0)
                res.dropna(axis=0, how='any', inplace=True)
    # res[performance] = np.NAN
    res.reset_index()
    two = np.zeros(res.shape[0])

    fig = plt.figure()
    plt.plot(loss_dict['dloss'], color='red', label='D')
    plt.plot(loss_dict['gloss'], color='blue', label='G')
    plt.title('loss')
    plt.legend()
    plt.savefig('./actgan/loss.png')

    actgan_32_end_time = time.time()
    print("结束actgan产生32组样本的时间：", time.ctime(actgan_32_end_time))
    print("actgan产生32组样本的时间：{}s".format(actgan_32_end_time - actgan_32_start_time))
    time_dict["actgan产生32组样本的时间"] = actgan_32_end_time - actgan_32_start_time

    actgan_performance_32_config_start_time = time.time()
    print("开始32组样本获取性能的时间：", time.ctime(actgan_performance_32_config_start_time))

    for i in range(res.shape[0]):
        x = res.to_dict('records')[i]
        p = get_performance(x, config_params, performance)
        two[i] = p

    res.insert(res.shape[1], performance, two)
    actgan_performance_32_config_end_time = time.time()
    print("结束32组样本获取性能的时间：", time.ctime(actgan_performance_32_config_end_time))
    time_dict["32组样本获取性能的时间"] = actgan_performance_32_config_end_time - actgan_performance_32_config_start_time

    c1 = time.time()
    res = pd.concat([train_data.iloc[:-BATCH_SIZE, :], res], axis=0)

    # 保存actgan相关数据
    nameactgan = '{}actgan.csv'.format(sample_num)
    name = os.path.join(datadir, nameactgan)
    res.to_csv(name, index=False)

    if min_or_max == 'max':
        res = res.sort_values(by=performance, ascending=False)
    else:
        res = res.sort_values(by=performance, ascending=True)
    res = res.iloc[0, :]

    c2 = time.time()
    time_dict['确定{}组参数中最优参数的时间'.format(sample_num)] = c2 - c1

    time_txt = os.path.join(datadir, 'time.txt')
    file_out = open(time_txt, 'w+', encoding='utf-8')
    file_out.write(str(time_dict))
    file_out.close()
    # output_dir = config_params['output_dir']
    # namecsv = uuid.uuid1().__str__() + '.csv'
    # name = os.path.join(output_dir, namecsv)
    # res.to_csv(name, index=False)

    # name2 = os.path.join(datadir, 'actgan_file.txt')
    # file_out = open(name2, 'w+')
    # file_out.write(str(name))
    # file_out.close()
    # print("actgan_result:" + str(name))

    # report_data(experimentRunId, conversationId, 'actgan', name)
    data = pd.DataFrame(dict(res), index=[0])
    namecsv = 'best.csv'
    name = os.path.join(datadir, namecsv)
    data.to_csv(name, index=False)
    # best_data = {}
    # best_data[performance] = res.pop(performance)
    # best_data['paramList'] = res

    # print(best_data)
    # data(best_data, config_params)

    # name2 = os.path.join(datadir, 'actgan_file.txt')
    # file_out = open(name2, 'w+')
    # file_out.write(str(best_data))
    # file_out.close()
    # print("actgan_result:" + str(res))


def run():
    args = parse_arguments()
    path = args.random_sample_file
    file_read = open(path, 'r', encoding='utf-8')
    params = file_read.read()
    # print("actgan_shu_ru_nei_rong:" + params)
    params = dict(eval(params))
    # print(params)

    temp = params.copy()
    performance = temp.pop('performance')
    columns = temp.pop('columns')
    path = temp.pop('random_sample_result_path')
    min_or_max = temp.pop('min_or_max')
    sample_num = temp.pop('sample_num')
    selected_params = temp

    actgan_component_params = eval("".join(args.actgan_component_params))
    # print("actgan_component_params:" , actgan_component_params)
    hyper_params = {}
    hyper_params['Epoch'] = actgan_component_params['Epoch']  # 算法迭代次数本为150000
    hyper_params['LR_G'] = actgan_component_params['LR_G']  # 生成器的学习率 0.0001
    hyper_params['LR_D'] = actgan_component_params['LR_D']  # 判别器的学习率 0.0001
    hyper_params['N_IDEAS'] = actgan_component_params['N_IDEAS']  # 生成器输入随机噪声的维数
    hyper_params['NumOfLine'] = actgan_component_params['NumOfLine']  # 每次取多少个样本进行训练本为16
    hyper_params['BATCH_SIZE'] = actgan_component_params['BATCH_SIZE']  # 总的训练样本数目本为32

    config_params = eval("".join(args.config_params))

    train_data = pd.read_csv(path)
    train_data = train_data.replace(True, "true")
    train_data = train_data.replace(False, "false")
    actgan(train_data, hyper_params, columns, selected_params, performance, min_or_max,
           actgan_component_params, config_params,sample_num)
    file_read.close()


if __name__ == '__main__':
    run()
