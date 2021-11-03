# -*- encoding: utf-8 -*-
"""
@File    : select_all.py
@Time    : 2021/3/8 9:14
@Author  : zspp
@Software: PyCharm
"""
import argparse
import os
import uuid

datadir = os.path.dirname((os.path.abspath(__file__)))


# print("select_all_datadir:" + datadir)


def parse_arguments():
    """
    参数解析
    :return:
    """

    parser = argparse.ArgumentParser(description='select_all')
    parser.add_argument('--select_all_component_params', type=str, required=True, nargs='+',
                        help='select_all_component_params,dict_style')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
    parser.add_argument('--common_params', type=str, required=True, nargs='+', help='common_params')

    args = parser.parse_args()
    return args


def select_all(all_params, black_list, white_list, performance, min_or_max, config_params):
    """
    选择输入的所有参数（黑名单内的除外）

    :param min_or_max: 最大化性能指标值，还是最小化
    :param config_params: config_params
    :param performance: 性能指标值
    :param all_params: python dict格式，key是参数名， value中包含类型、取值范围、默认值
    :param black_list: python list格式，存储不选择的参数名，可为空
    :param white_list: python list格式，存储必选择的参数名，可为空
    :return: 选择后的参数，存储到文件
    """
    res = {}
    for each in white_list:
        res[each] = all_params[each]

    for key in all_params.keys():
        if key in black_list or key in white_list:
            continue

        res[key] = all_params[key]

    res['performance'] = performance
    res['min_or_max'] = min_or_max

    # output_dir = config_params['output_dir']

    name = os.path.join(datadir, "result.txt")
    file_out = open(name, 'w+', encoding='utf-8')
    # print("select_call_context:" + name)
    file_out.write(str(res))
    file_out.close()

    name2 = os.path.join(datadir, 'select_all_file.txt')
    file_out = open(name2, 'w+', encoding='utf-8')
    file_out.write(str(name))
    file_out.close()
    # print("select_all_file_context:" + str(name))


def run():
    args = parse_arguments()

    args.select_all_component_params = eval("".join(args.select_all_component_params))
    args.config_params = eval("".join(args.config_params))

    args.common_params = eval("".join(args.common_params))

    # print("select_all_component_params:", args.select_all_component_params)
    # print("config_params:", args.config_params)
    # print("common_params:", args.common_params)
    all_params = (args.common_params['all_params'])
    black_list = (args.common_params['black_list'])
    white_list = (args.common_params['white_list'])
    performance = args.common_params['performance']

    min_or_max = performance[-3:]
    performance = performance[:-4]

    print("min_or_max:", min_or_max)
    print("performance:", performance)
    print("all_params:", all_params)
    print("black_list:", black_list)
    print("white_list:", white_list)

    select_all(all_params, black_list, white_list,
               performance, min_or_max, args.config_params)


if __name__ == '__main__':
    run()
