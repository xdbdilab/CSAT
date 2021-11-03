# -*- encoding: utf-8 -*-
"""
 @Time : 2021/3/12 14:31
 @Author : zspp
 @File : space_search
 @Software: PyCharm 
"""
import random
import pandas as pd
import argparse
import sys
import os
import math
import operator
import copy
import time
import numpy as np
# from nfs.colosseum.getPerformance import get_performance
# from nfs.colosseum.report import *
from getPerformance import get_performance

datadir = os.path.dirname((os.path.abspath(__file__)))
# print("space_search_datadir:" + datadir)
time_dict = {}



def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='space_search')
    parser.add_argument('--select_all_file', type=str, required=True, help='select_all\'result_filename')
    parser.add_argument('--space_search_component_params', type=str, required=True, nargs='+',
                        help='space_search_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    args = parser.parse_args()
    return args


def inAlready(sets, permNew):
    """
    查询样本集permNew是否已经在sets中
    :param sets: [permNew1,permNew2,.....]
    :param permNew: 每一行为同一种配置的随机序列,每一列为一次配置
    :return:
    """
    noSame = True
    notEqual = False
    for i in range(len(sets)):
        if sets[i]:
            notEqual = False
            for attR in range(1, len(permNew)):
                for p in range(len(permNew[attR])):
                    if (sets[i][attR][p] != permNew[attR][p]):
                        notEqual = True
                        break
                if notEqual:
                    break

            if not notEqual:
                noSame = False
                break
        else:
            break

    return not noSame


def uniBoundsGeneration(bounds, crntAttr, sampleSetSize):
    """
    划分属性区间
    :param bounds:[bound0,boud1,boud2,....,bound_samplesetsize]
    :param crntAttr: 参数的属性，['int',[1,10],0]
    :param sampleSetSize: 样本集大小
    :return:
    """
    bounds[0] = crntAttr[1][0]
    bounds[-1] = crntAttr[1][1]
    pace = (bounds[sampleSetSize] - bounds[0]) / sampleSetSize
    for j in range(1, sampleSetSize + 1):
        bounds[j] = bounds[j - 1] + pace
    return bounds


def positionSwitch(sets, dists, pos1, pos2):
    """
    交换位置
    :param sets: 样本集的集合
    :param dists: 距离列表
    :param pos1: 位置2
    :param pos2: 位置2
    :return:
    """
    tempSet = sets[pos1]
    sets[pos1] = sets[pos2]
    sets[pos2] = tempSet

    tempVal = dists[pos1]
    dists[pos1] = dists[pos2]
    dists[pos2] = tempVal

    return sets, dists


def eucDistForPairs(sampleA, sampleB):
    """
    :param sampleA: 一组参数配置[conf1,conf2,conf3,....]
    :param sampleB: 一组参数配置[conf1,conf2,conf3,....]
    :return: 距离
    """
    dist = 0
    for i in range(len(sampleA)):
        dist += (sampleA[i] - sampleB[i]) * (sampleA[i] - sampleB[i])
    return dist


def minDistForSet(setPerm):
    """
    :param setPerm: 每一行为同一种配置的随机序列,每一列为一次配置
    :return: 样本集setPerm中两个样本之间最短的距离
    """
    min_distance = sys.maxsize
    sampleSetSize = len(setPerm[0])
    attrNum = len(setPerm)
    sampleA = [None] * attrNum  # 一组参数配置[conf1,conf2,conf3,....]
    sampleB = [None] * attrNum  # 另一组参数配置[conf1,conf2,conf3,....]
    for i in range(sampleSetSize - 1):
        for j in range(attrNum):
            sampleA[j] = setPerm[j][i]

        for k in range(i + 1, sampleSetSize):
            for j in range(attrNum):
                sampleB[j] = setPerm[j][k]

            distance = eucDistForPairs(sampleA, sampleB)
            min_distance = distance if min_distance > distance else min_distance

    return min_distance


def generateOneSampleSet(sampleSetSize, attrNum):
    """
    :return: setPerm,每一行为同一种配置的随机序列,每一列为一次配置
    :param sampleSetSize: 样本集大小
    :param attrNum: 参数数量
    """
    setPerm = [None] * attrNum  # 每一行为同一种配置的随机序列,每一列为一次配置

    for i in range(1, attrNum):
        setPerm[i] = [None] * sampleSetSize

        # 随机生成 （0 - sampleSetSize - 1） 的序列
        for j in range(sampleSetSize):
            crntRand = random.randint(0, sampleSetSize - 1)

            while crntRand in setPerm[i]:
                crntRand = random.randint(0, sampleSetSize - 1)

            setPerm[i][j] = crntRand

    # 第一个参数属性始终使用自然序列
    setPerm[0] = [None] * sampleSetSize
    for j in range(sampleSetSize):
        setPerm[0][j] = j
    return setPerm


sets = None


def sampleMultiDimContinuous(selected_params, sampleSetSize, useMid, sampleSetToGet):
    """
    采样样本集
    :param selected_params: 参数属性{'param1': 10, 'param2': 's1', ...}字典形式
    :param sampleSetSize: 样本集大小
    :param useMid: 是否使用中间值
    :param sets: 样本集的集合，注意！！
    :return: data 每一行为一条数据，即一组参数，总共sampleSetSize行
    """
    # only initialize once
    # 分散得更远的配置放在第一位,即覆盖空间范围更广的
    global sets

    if not sets:
        # possible number of sample sets will not exceed $sampleSetSize to the power of 2
        length = len(selected_params)
        if length > 2:
            temp = math.pow(sampleSetSize, length - 1)
        elif length > 1:
            temp = sampleSetSize
        else:
            temp = 1
        L = int(min(RRSMaxRounds, temp))
        dists = []
        sets = []
        for i in range(L):
            dists.append(-1)
            sets.append(None)

        maxMinDist = -1
        posWithMaxMinDist = -1
        # 生成L个大小的样本集
        for i in range(L):
            setPerm = generateOneSampleSet(sampleSetSize, length)
            while inAlready(sets, setPerm):
                setPerm = generateOneSampleSet(sampleSetSize, length)
            sets[i] = setPerm

            dists[i] = minDistForSet(setPerm)
            if dists[i] > maxMinDist:
                posWithMaxMinDist = i
                maxMinDist = dists[i]

        sets, dists = positionSwitch(sets, dists, 0, posWithMaxMinDist)

    # 获取第几次配置
    crntSetPerm = sets[sampleSetToGet]
    # 划分每个参数属性间隔
    bounds = [[None] * (sampleSetSize + 1)] * len(selected_params)
    roundToInt = [False] * len(selected_params)

    for i in range(len(bounds)):
        crntAttr = selected_params[(list(selected_params.keys()))[i]]
        bounds[i] = uniBoundsGeneration(bounds[i], crntAttr, sampleSetSize)[:]

        temp_divide = bounds[i][-1] - bounds[i][0]
        if temp_divide > sampleSetSize:
            roundToInt[i] = True

    # 根据划分区间与排列来构造数据集
    data = [[None] * len(selected_params)] * sampleSetSize  # 一行为一条数据
    for i in range(sampleSetSize):
        vals = [None] * len(selected_params)
        for j in range(len(selected_params)):
            if useMid:
                vals[j] = (bounds[j][crntSetPerm[j][i]] + bounds[j][crntSetPerm[j][i] + 1]) / 2
            else:
                vals[j] = bounds[j][crntSetPerm[j][i]] + (
                        bounds[j][crntSetPerm[j][i] + 1] - bounds[j][crntSetPerm[j][i]]) * random.uniform(0.01, 0.99)
            if roundToInt[j]:
                vals[j] = int(vals[j])

        data[i] = vals[:]
    return data


def getMultiDimContinuous(selected_params, sampleSetSize, useMid, sampleSetToGet):
    """

    :param selected_params: 参数属性
    :param sampleSetSize: 初始样本集大小
    :param useMid: 是否取中间值
    :return:  采样数据集，形式目前定义为：[[1,2,3,...参数长度],[],,,,[]]
    """
    data = sampleMultiDimContinuous(selected_params, sampleSetSize, useMid, sampleSetToGet)
    while len(data) < sampleSetSize:
        temp = sampleMultiDimContinuous(selected_params, sampleSetSize, useMid, sampleSetToGet)
        data.extend(temp)

    # 删除多余的数据，只保留样本集大小的数据
    while len(data) > sampleSetSize:
        data.pop(len(data) - 1)

    return data


def setCurrentRound(round):
    """
    设置采样sets的哪一个样本集，sets是样本集的集合
    :param round:第几个
    :return:
    """
    sampleSetToGet = 0
    if (sets != None and round < len(sets)):
        sampleSetToGet = round
    return sampleSetToGet


class MidParams:
    currentround = 0
    subround = 0
    currentBest = None


def runExp(samplePoints, performance, selected_params, config_params):
    """
    获取性能指标值
    :param selected_params: 原始参数设置
    :param samplePoints: 采样样本集
    :param performance: 性能名称
    :param config_params: config_params
    :return: 采样样本集+性能 [[conf1,conf2,,,,,,,performance],.....]
    """

    for i in range(len(samplePoints)):
        columns = list(selected_params.keys())
        # 需要还原enum,string,参数
        for j, key in zip(list(range(len(columns))), columns):
            if selected_params[key][0] == 'enum' or selected_params[key][0] == 'string':
                samplePoints[i][j] = selected_params[key][1][math.floor(samplePoints[i][j])]
            elif selected_params[key][0] == 'int':
                samplePoints[i][j] = int(samplePoints[i][j])

        value = samplePoints[i]
        params = dict(zip(columns, value))
        y = get_performance(params, config_params, performance)
        # y= i #测试用
        samplePoints[i].append(y)
    return samplePoints


def findBestPerf(trainingsamplePoints, min_or_max):
    """
    找到样本集中最优的性能的一条
    :param trainingsamplePoints:样本集[[conf1,conf2,,,,,,,performance],.....]
    :param min_or_max:最大化or最小化
    :return:最优的一条样本[conf1,conf2,,,,,,,performance]
    """
    hang = len(trainingsamplePoints)
    index = -1
    if min_or_max == 'max':
        bestPerf = 1 - sys.maxsize
        for i in range(hang):
            if trainingsamplePoints[i][-1] > bestPerf:
                index = i
                bestPerf = trainingsamplePoints[i][-1]
        return trainingsamplePoints[index]
    else:
        bestPerf = sys.maxsize
        for i in range(hang):
            if trainingsamplePoints[i][-1] < bestPerf:
                index = i
                bestPerf = trainingsamplePoints[i][-1]
        return trainingsamplePoints[index]


def defltSettings(selected_params):
    """
    获取一组参数默认值
    :param selected_params: 原始参数属性
    :return: [conf1,conf2,,,,,]
    """
    length = len(selected_params)
    default_data = [None] * length
    for i in range(length):
        key = (list(selected_params.keys()))[i]
        if selected_params[key][0] == 'enum' or selected_params[key][0] == 'string':
            default_data[i] = selected_params[key][1].index(selected_params[key][2])
        else:
            default_data[i] = selected_params[key][2]

    return default_data


def scaleDownDetour(trainingSet, best_one_data, selected_params):
    """
        更改参数的范围大小
        :param trainingSet: 参数集
        :param best_one_data: 最优的那条参数
        :param selected_params: 原始参数范围
        :return: 更改后的参数范围
        """
    localAtts = copy.deepcopy(selected_params)
    attNum = len(selected_params)
    minDists = [None] * 2
    for i in range(attNum):

        minDists[0] = 1 - sys.maxsize
        minDists[1] = sys.maxsize
        haha_ENUM = False
        for j in range(len(trainingSet)):

            if not operator.eq(trainingSet[j], best_one_data):
                if type(trainingSet[j][i]) != int and type(trainingSet[j][i]) != float:
                    haha_ENUM =True
                    break

                else:
                    val = trainingSet[j][i] - best_one_data[i]
                    if val < 0:
                        minDists[0] = max((float)((int)(val * 1000)) / 1000.0, minDists[0])
                    else:
                        minDists[1] = min((float)((int)(val * 1000)) / 1000.0, minDists[1])
        key = (list(selected_params.keys()))[i]
        if haha_ENUM ==True:
            upper = selected_params[key][1][1]
            lower = selected_params[key][1][0]
        else:

            upper = best_one_data[i] + minDists[1]
            lower = best_one_data[i] + minDists[0]

        detourSet = set()
        detourSet.add(upper)
        detourSet.add(lower)
        key = (list(selected_params.keys()))[i]
        detourSet.add(selected_params[key][1][1])
        detourSet.add(selected_params[key][1][0])
        detourSet = sorted(list(detourSet))

        length = len(detourSet)
        if length == 1:
            upper = lower = detourSet[0]

        elif length == 2:
            # 针对dataset为3的特殊情况，且在边边
            upper = detourSet[-1]
            lower = detourSet[0]

        elif length == 3:
            # 针对与原始范围上下界一个值，且dataset只有它一个
            upper = lower = detourSet[1]

        else:
            upper = detourSet[-2]
            lower = detourSet[1]

        localAtts[key][1][1] = upper
        localAtts[key][1][0] = lower

    return localAtts


def RBSoDDSOptimization(selected_params, performance, min_or_max, defaultSettings, default_params, config_params):
    """
    RBS搜索最优空间
    :param defaultSettings: 默认参数，enum、string属性已经数字化后的
    :param selected_params: 参数属性,enum、string属性已经数字化后的
    :param config_params: config_params
    :param performance: 性能名称
    :param min_or_max: 最大化性能还是最小化，值为字符串，“min”或者“max”
    :return: 最优参数设置,performance
    """
    best_csv = []
    temp_csv = []  # 所有采样结果
    middle_csv = []
    samplePoints = None
    trainingsamplePoints = None
    total_num = RRSMaxRounds * InitialSampleSetSize
    time_dict["{}个参数的生成时间".format(total_num)] = 0
    time_dict["{}个参数的运行时间".format(total_num)] = 0

    number = 0
    params = copy.deepcopy(selected_params)
    opParams = MidParams()
    if min_or_max == 'min':
        opParams.currentBest = sys.maxsize
    else:
        opParams.currentBest = 1 - sys.maxsize

    while opParams.currentround < RRSMaxRounds:
        if number >= RRSMaxRounds * InitialSampleSetSize:
            break
        if opParams.currentround != 0 or opParams.subround != 0:
            temp_round = setCurrentRound(opParams.currentround)
            c1 = time.time()
            samplePoints = getMultiDimContinuous(params, InitialSampleSetSize, False, temp_round)
            number = number + 50
            c2 = time.time()
            time_dict['{}个参数的生成时间'.format(total_num)] = time_dict['{}个参数的生成时间'.format(total_num)] + c2 - c1
            trainingsamplePoints = runExp(samplePoints, performance, default_params, config_params)
            c3 = time.time()
            time_dict['{}个参数的运行时间'.format(total_num)] = time_dict['{}个参数的运行时间'.format(total_num)] + c3 - c2

            temp_csv.extend(trainingsamplePoints)
        else:
            temp_round = setCurrentRound(opParams.currentround)
            c1 = time.time()
            samplePoints = getMultiDimContinuous(params, InitialSampleSetSize, False, temp_round)
            number = number + 50
            samplePoints[0] = defaultSettings
            c2 = time.time()
            time_dict['{}个参数的生成时间'.format(total_num)] = time_dict['{}个参数的生成时间'.format(total_num)] + c2 - c1
            trainingsamplePoints = runExp(samplePoints, performance, default_params, config_params)
            c3 = time.time()
            time_dict['{}个参数的运行时间'.format(total_num)] = time_dict['{}个参数的运行时间'.format(total_num)] + c3 - c2
            temp_csv.extend((trainingsamplePoints))

        best_one_data = findBestPerf(trainingsamplePoints, min_or_max)
        print(best_one_data)
        middle_csv.append(best_one_data)

        tempBest = best_one_data[-1]

        if (tempBest > opParams.currentBest and min_or_max == 'max') or (
                tempBest < opParams.currentBest and min_or_max == 'min'):
            print("Previous best is ", opParams.currentBest, "; Current best is ", tempBest)
            opParams.currentBest = tempBest

            best_csv.append(best_one_data)

            props = scaleDownDetour(trainingsamplePoints, best_one_data, selected_params)  # 更改参数范围
            params = props

            opParams.subround += 1
        else:
            samplePoints = None
            opParams.currentround += 1
            opParams.subround = 0
            params = copy.deepcopy(selected_params)
            print("Entering into round ", opParams.currentround)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("We are ending the optimization experiments!")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("The best performance is : ", opParams.currentBest)
    print("=========================================")

    columns2 = list(default_params.keys())
    columns2.append(performance)

    temp = [best_csv[-1][:]]
    data_write = pd.DataFrame(temp, columns=columns2)
    # 所有数据
    all_data = pd.DataFrame(np.array(temp_csv), columns=columns2)
    namecsv = '{}bestconfig.csv'.format(total_num)
    name = os.path.join(datadir, namecsv)
    all_data.to_csv(name, index=False)

    # 保存中间数据

    temp_data = pd.DataFrame(np.array(middle_csv), columns=columns2)
    namecsv = '中间搜索过程点.csv'
    name = os.path.join(datadir, namecsv)
    temp_data.to_csv(name, index=False)

    temp_data = pd.DataFrame(np.array(best_csv), columns=columns2)
    namecsv = '最优搜索过程点.csv'
    name = os.path.join(datadir, namecsv)
    temp_data.to_csv(name, index=False)

    time_txt = os.path.join(datadir, 'time.txt')
    file_out = open(time_txt, 'w+', encoding='utf-8')
    file_out.write(str(time_dict))
    file_out.close()
    return data_write, best_csv[-1][-1]


def space_search(params, space_search_component_params, config_params):
    """
    bestconfig搜索最优参数
    :param space_search_component_params: 模块参数
    :param config_params: config_params  java接口一些相关参数
    :param params: 原始参数设置
    """
    temp = params.copy()
    performance = temp.pop('performance')
    min_or_max = temp.pop('min_or_max')
    selected_params = temp

    default_params = copy.deepcopy(selected_params)
    defaultSettings = defltSettings(selected_params)

    # 考虑到bestconfig自身的情况，需要对数据进行处理，将enum，string类型的参数数字化
    for key in list(selected_params.keys()):
        if selected_params[key][0] == 'enum' or selected_params[key][0] == 'string':
            selected_params[key][1] = list([0, len(selected_params[key][1])])

    Bdata, perf = RBSoDDSOptimization(selected_params, performance, min_or_max, defaultSettings, default_params,
                                      config_params)

    # namecsv = uuid.uuid1().__str__() + '.csv'
    # name = os.path.join(datadir, namecsv)
    # data.to_csv(name, index=False)
    #
    # name2 = os.path.join(datadir, 'space_search_file.txt')
    # file_out = open(name2, 'w+')
    # file_out.write(str(name))
    # file_out.close()
    # print("space_search_result:" + str(name))

    df_samples = dict(Bdata.iloc[0, :])

    best_data = {}
    best_data[performance] = perf
    best_data.update(df_samples)

    # print(best_data)
    # data(best_data, config_params)
    # df_samples[performance] = perf
    data = pd.DataFrame(dict(best_data), index=[0])
    namecsv = 'best.csv'
    name = os.path.join(datadir, namecsv)
    data.to_csv(name, index=False)

    # name2 = os.path.join(datadir, 'space_search_file.txt')
    # file_out = open(name2, 'w+')
    # file_out.write(str(best_data))
    # file_out.close()


def run():
    args = parse_arguments()
    path = args.select_all_file
    file_read = open(path, 'r')
    params = file_read.read()
    # print("space_search_shu_ru_nei_rong:" + params)
    params = dict(eval(params))
    # print(params)

    space_search_component_params = eval("".join(args.space_search_component_params))
    config_params = eval("".join(args.config_params))

    # print("space_search_component_params:", space_search_component_params)
    # print("config_params:", config_params)

    # bestconfig相关参数设置
    global InitialSampleSetSize
    InitialSampleSetSize = space_search_component_params['InitialSampleSetSize']
    global RRSMaxRounds
    RRSMaxRounds = space_search_component_params['RRSMaxRounds']

    # print("InitialSampleSetSize:", InitialSampleSetSize)
    # print("RRSMaxRounds:", RRSMaxRounds)

    space_search(params, space_search_component_params, config_params)
    file_read.close()


if __name__ == '__main__':
    run()
