import argparse
import os
import time
import uuid
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
from sklearn import preprocessing

datadir = os.path.dirname((os.path.abspath(__file__)))
print("lasso_select_datadir:" + datadir)
print()
time_dict = {}


def parse_arguments():
    """
    参数解析
    :return:
    """

    parser = argparse.ArgumentParser(description='lasso_select')
    parser.add_argument('--lasso_select_component_params', type=str, required=True, nargs='+',
                        help='lasso_select_component_params,dict_style')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    parser.add_argument('--random_sample_file', type=str, required=True, help='random_sample_file')
    parser.add_argument('--common_params', type=str, required=True, nargs='+', help='common_params')

    args = parser.parse_args()
    return args


def lasso_select(input_file, all_params, white_list, performance, min_or_max, config_params, alpha=1):
    """
    :param input_file: 采样得到的csv文件，适用lasso回归进行特征选择
    :param config_params:
    :param alpha: lasso超参数
    :return: 保存特征选择后的csv文件
    """
    lasso_start_time = time.time()

    res = {}

    inputfile = input_file  # 输入的数据文件
    data = pd.read_csv(inputfile, encoding='utf-8')  # 读取数据
    data = data.replace(True, "true")
    data = data.replace(False, "false")
    data1 = data.select_dtypes(include='number')
    data2 = data.select_dtypes(include='object')
    X, y = data1.iloc[:, :-1], data1[performance]

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X.values)
    y_train = y.values[:, np.newaxis]
    lasso = Lasso(alpha)  # 调用Lasso()函数，设置λ的值为1000

    lasso.fit(X_train, y_train)
    print('相关系数为：', np.round(lasso.coef_, 5))  # 输出结果，保留五位小数

    ## 计算相关系数非零的个数
    print('相关系数非零个数为：', np.sum(lasso.coef_ != 0))
    mask = lasso.coef_ != 0  # 返回一个相关系数是否为零的布尔数组
    print('相关系数是否为零：', mask)
    if np.sum(lasso.coef_ != 0) == 0:
        new_data = data1.iloc[:, :-1]
    else:
        new_data = X.iloc[:, mask]  # 返回相关系数非零的数据

    outputfile = os.path.join(datadir, 'new_train_data.csv')  # 输出的数据文件

    new_data = pd.concat((data2, new_data), axis=1)
    features = list(new_data.columns)
    for i in white_list:
        if i not in features:
            features.append(i)
            new_data[i] = data1[i]
    new_data[performance] = data[performance]
    res['features'] = features
    print('lasso选择的特征：', features)

    for f in features:
        res[f] = all_params[f]

    new_data.to_csv(outputfile, index=False)  # 存储数据
    print('输出数据的维度为：', new_data.shape)  # 查看输出数据的维度
    print()

    res['lasso_select_result_path'] = outputfile
    res['performance'] = performance
    res['min_or_max'] = min_or_max

    name = os.path.join(datadir, "result.txt")
    file_out = open(name, 'w+', encoding='utf-8')
    # print("select_call_context:" + name)
    file_out.write(str(res))
    file_out.close()

    name2 = os.path.join(datadir, 'lasso_select_file.txt')
    file_out = open(name2, 'w+', encoding='utf-8')
    file_out.write(str(name))
    file_out.close()

    lasso_end_time = time.time()
    print("lasso特征选择的开始时间：", time.ctime(lasso_start_time))
    print("lasso特征选择的结束时间：", time.ctime(lasso_end_time))
    print("lasso特征选择的耗费时间：{}s".format(lasso_end_time - lasso_start_time))
    time_dict['lasso特征选择的耗费时间'] = lasso_end_time - lasso_start_time
    print()
    # print(time_dict)
    time_txt = os.path.join(datadir, 'time.txt')
    file_out = open(time_txt, 'w+', encoding='utf-8')
    file_out.write(str(time_dict))
    file_out.close()


def run():
    args = parse_arguments()

    args.lasso_select_component_params = eval("".join(args.lasso_select_component_params))
    args.config_params = eval("".join(args.config_params))
    args.common_params = eval("".join(args.common_params))
    path = args.random_sample_file

    file_read = open(path, 'r', encoding='utf-8')
    params = file_read.read()
    # print("actgan_shu_ru_nei_rong:" + params)
    params = dict(eval(params))
    # print(params)

    temp = params.copy()
    columns = temp.pop('columns')
    path = temp.pop('random_sample_result_path')

    # print("lasso_select_component_params:", args.lasso_select_component_params)
    # print("config_params:" , args.config_params)
    # print("common_params:" , args.common_params)

    alpha = args.lasso_select_component_params['alpha']
    black_list = args.common_params['black_list']
    white_list = args.common_params['white_list']
    performance = args.common_params['performance']
    all_params = args.common_params['all_params']

    min_or_max = performance[-3:]
    performance = performance[:-4]

    print('alpha:', alpha)
    print('random_sample_result_path:', path)
    print("min_or_max:", min_or_max)
    print("performance:", performance)
    print("black_list:", black_list)
    print("white_list:", white_list)
    print()

    lasso_select(path, all_params, white_list, performance, min_or_max, args.config_params, alpha)


if __name__ == '__main__':
    run()
