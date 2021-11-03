# # -*- encoding: utf-8 -*-
# """
#  @Time : 2021/3/9 10:38
#  @Author : zspp
#  @File : GA
#  @Software: PyCharm
# """
# import random
# import numpy as np
# import math
# import pandas as pd
# import argparse
# import os
# import copy
# import pickle
# import joblib
#
# # from nfs.colosseum.report import *
# # from nfs.colosseum.getPerformance import get_performance
# from getPerformance import get_performance
#
# datadir = os.path.dirname((os.path.abspath(__file__)))
#
#
# # print("GA_datadir:" + datadir)
#
#
# def parse_arguments():
#     """
#     参数解析
#     :return:
#     """
#     parser = argparse.ArgumentParser(description='GA')
#     parser.add_argument('--random_forest_file', type=str, required=True, help='random_forest_file')
#     parser.add_argument('--genetic_algorithm_component_params', type=str, required=True, nargs='+',
#                         help='genetic_algorithm_component_params')
#     parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
#
#     args = parser.parse_args()
#     return args
#
#
# def data_in(data, selected_params, columns, performance, min_or_max):  # 每次只传入一组参数，所以与acgan、bogan的数据预处理不一样
#     # print(data)
#     # if min_or_max == 'max':
#     #     data = data.sort_values(by=performance, ascending=False)
#     # else:
#     #     data = data.sort_values(by=performance, ascending=True)
#     data = data.drop(performance, 1)
#     # print(data)
#     ##首先处理没有优先关系的enum参数，即string参数
#     char = []  # enum的列名称
#     enum_index = []  # enum的原始列索引
#     for name in columns[:-1]:
#         if selected_params[name][0] == 'string' or selected_params[name][0] == 'enum':
#             char.append(name)
#             enum_index.append(columns.index(name))
#
#     enum_number = []  # 每个enum参数对应的独热编码的长度
#     enum_book = {}  # 每个enum参数的内容，字典形式存储
#     m = 0
#     for c in char:
#         i = enum_index[m]
#
#         new_data = pd.DataFrame({c: selected_params[c][1]})  # 添加几行，为了更好全面编码
#         data = data.append(new_data, ignore_index=True)
#
#         enum_book[c] = list(pd.get_dummies(data[c]).columns)
#         enum_data = pd.get_dummies(data[c], prefix=c)  # 独热编码后的变量
#
#         data = data.drop(c, 1)
#
#         enum_list = list(enum_data.columns)
#         enum_number.append(len(enum_list))
#
#         for k in range(len(enum_list)):
#             data.insert(i + k, enum_list[k], enum_data[enum_list[k]])  # 将向量移动到原来枚举值的位置
#         m = m + 1
#         enum_index = [j + len(enum_data.columns) - 1 for j in enum_index]  # 更新enum_index
#
#         data.drop(data.index[-len(selected_params[c][1]):], inplace=True)  # 删除前3行
#
#         # print(enum_index)
#     # print(enum_number)
#     # print(data)
#     # print(enum_book)
#
#     ##接着处理有优先关系的参数
#     # char2 = []
#     # enum_index2 = {}
#     # for name in columns[:-1]:
#     #     if selected_params[name][0] == 'enum':
#     #         char2.append(name)
#     #         enum_index2[name] = selected_params[name][1]
#     # for enum_name in char2:
#     #     size_range = list(range(len(enum_index2[enum_name])))
#     #     mapping = dict(zip(enum_index2[enum_name], size_range))
#     #     data[enum_name] = data[enum_name].map(mapping)
#     return data
#
#
# def get_performance_rt(params, sc, selected_params, columns, performance, min_or_max):
#     params = data_in(params, selected_params, columns, performance, min_or_max)
#     params = params.iloc[:, :].values
#     params = sc.transform(params.astype(float))
#     y_pred = rt.predict(params)
#     return y_pred
#
#
# class Individual:
#     def __init__(self, selected_params, fitness=0.0):
#         self.genes = [-1] * len(selected_params)  # double
#         self.fitness = fitness
#
#     def clone(self, selected_params):
#         individual = Individual(selected_params)
#         individual.setGenes(self.genes)
#         return individual
#
#     def generateIndividual(self, selected_params, columns):
#         while True:
#             for i in range(len(selected_params)):
#                 gene = random.random()
#                 self.setGene(i, gene)
#             if Individual.isLegal(self.transfromGene(selected_params, columns)):
#                 break
#
#     def getGenes(self):
#         return self.genes
#
#     def setGenes(self, genes):
#         self.genes = copy.deepcopy(genes)
#
#     def getFitness(self, selected_params, columns, sc, performance, min_or_max):
#         if self.fitness == 0:
#             parameter = self.transfromGene(selected_params, columns)
#             params = pd.DataFrame(parameter, index=[0], columns=columns)
#             try:
#                 self.fitness = get_performance_rt(params, sc, selected_params, columns, performance,
#                                                   min_or_max)
#             except Exception as e:
#                 print(e)
#         return self.fitness
#
#     def setFitness(self, fitness):
#         self.fitness = fitness
#
#     def getGene(self, index):
#         return self.genes[index]
#
#     def setGene(self, index, value):
#         self.genes[index] = value
#
#     @classmethod
#     def isLegal(cls, parameters):
#         return True
#
#     def transfromGene(self, selected_params, columns):
#
#         parameters = {}
#         # 'sort_buffer_size': ['int', [131072, 16777216], 262144],
#         j = 0
#         for i in columns[:-1]:
#             para_type = selected_params[i][0]
#             if para_type == 'int':
#                 parameters[i] = int(
#                     math.floor(self.getGene(j) * (selected_params[i][1][1] - selected_params[i][1][0] + 1)) +
#                     selected_params[i][1][0])
#             elif para_type == 'float' or para_type == 'double':
#                 parameters[i] = float(
#                     math.floor(self.getGene(j) * (selected_params[i][1][1] - selected_params[i][1][0] + 1)) +
#                     selected_params[i][1][0])
#             elif para_type == 'enum' or 'string':
#                 choice = selected_params[i][1]  # 离散变量内容
#                 length = len(choice)  # 离散种类数目
#                 gene_choice = np.linspace(0.0, 1.0, num=length + 1)
#                 index = np.where(gene_choice > self.getGene(j))
#                 parameters[i] = choice[index[0][0] - 1]
#             j += 1
#         return parameters
#
#
# class Population:
#
#     def __init__(self, populationSize, initialise, selected_params, columns):
#         # 创建一个种群
#         self.individuals = [Individual(selected_params)] * populationSize
#         if initialise:
#             # 初始化种群
#             for i in range(populationSize):
#                 newIndividual = Individual(selected_params)
#                 newIndividual.generateIndividual(selected_params, columns)
#                 self.saveIndividual(i, newIndividual)
#
#     def saveIndividual(self, index, indiv):
#         self.individuals[index] = indiv
#
#     def getIndividual(self, index):
#         return self.individuals[index]
#
#     def getFittest(self, min_or_max, selected_params, columns, sc, performance):
#         fittest = self.individuals[0]
#         if min_or_max == 'max':
#             # 最大化性能
#             for i in range(1, len(self.individuals)):
#                 temp = fittest.getFitness(selected_params, columns, sc, performance, min_or_max)
#                 if temp < self.getIndividual(i).getFitness(selected_params, columns, sc, performance,
#                                                            min_or_max):
#                     fittest = self.getIndividual(i)
#
#
#         else:
#             # 最小化性能
#             for i in range(len(self.individuals)):
#                 temp = fittest.getFitness(selected_params, columns, sc, performance, min_or_max)
#                 if temp >= self.getIndividual(i).getFitness(selected_params, columns, sc, performance,
#                                                             min_or_max):
#                     fittest = self.getIndividual(i)
#         return fittest
#
#     def size(self):
#         return len(self.individuals)
#
#
# class Algorithm:
#     # /* GA 算法的参数*/
#     uniformRate = 0.5  # 交叉概率
#     mutationRate = 0.015  # 突变概率
#     tournamentSize = 5  # 淘汰数组的大小
#     elitism = True  # 精英主义
#
#     # 进化一个种群
#     @classmethod
#     def evolvePopulation(cls, pop, selected_params, columns, min_or_max, sc, performance):
#         # 存放新一代的种群
#         newPopulation = Population(pop.size(), False, selected_params, columns)
#         # 把最优秀的那个放在第一个位置
#         if (Algorithm.elitism):
#             newPopulation.saveIndividual(0, pop.getFittest(min_or_max, selected_params, columns, sc,
#                                                            performance))
#         # Crossover population
#         if (Algorithm.elitism):
#             elitismOffset = 1
#         else:
#             elitismOffset = 0
#
#         for i in range(elitismOffset, pop.size()):
#             # 随机选择两个优秀的个体
#             indiv1 = Algorithm.tournamentSelection(pop, min_or_max, selected_params, columns, sc,
#                                                    performance)
#             indiv2 = Algorithm.tournamentSelection(pop, min_or_max, selected_params, columns, sc,
#                                                    performance)
#             # 进行交叉
#             while True:
#                 newIndiv = Algorithm.crossover(indiv1, indiv2, selected_params)
#                 if Individual.isLegal(parameters=newIndiv.transfromGene(selected_params, columns)):
#                     break
#             newPopulation.saveIndividual(i, newIndiv)
#
#         # Mutate population  突变
#         for i in range(elitismOffset, newPopulation.size()):
#             while True:
#                 tempIndiv = newPopulation.getIndividual(i).clone(selected_params)
#                 Algorithm.mutate(tempIndiv, selected_params)
#                 if Individual.isLegal(parameters=tempIndiv.transfromGene(selected_params, columns)):
#                     break
#             newPopulation.getIndividual(i).setGenes(tempIndiv.getGenes())
#
#         return newPopulation
#
#     # 进行两个个体的交叉，交叉的概率为uniformRate
#     @classmethod
#     def crossover(cls, indiv1, indiv2, selected_params):
#         newSol = Individual(selected_params)
#         # 随机的从两个个体中选择
#         for i in range(len(selected_params)):
#             if (random.random() <= Algorithm.uniformRate):
#                 newSol.setGene(i, indiv1.getGene(i))
#             else:
#                 newSol.setGene(i, indiv2.getGene(i))
#         return newSol
#
#     # 突变个体，突变的概率为 mutationRate
#     @classmethod
#     def mutate(cls, indiv, selected_params):
#         for i in range(len(selected_params)):
#             if (random.random() <= Algorithm.mutationRate):
#                 gene = random.random()
#                 indiv.setGene(i, gene)
#
#     # 随机选择一个较优秀的个体，用于进行交叉
#     @classmethod
#     def tournamentSelection(cls, pop, min_or_max, selected_params, columns, sc, performance):
#         # Create a tournament population
#         tournamentPop = Population(Algorithm.tournamentSize, False, selected_params, columns)
#         # 随机选择tournamentSize 个放入tournamentPop中
#         for i in range(Algorithm.tournamentSize):
#             randomId = (int)(random.random() * pop.size())
#             tournamentPop.saveIndividual(i, pop.getIndividual(randomId))
#
#         # 找到淘汰数组中最优秀的
#         fittest = tournamentPop.getFittest(min_or_max, selected_params, columns, sc, performance)
#         return fittest
#
#
# def genetic_algorithm(columns, selected_params, performance, min_or_max, config_params, model_path, std_path,
#                       genetic_algorithm_component_params):
#     Algorithm.uniformRate = genetic_algorithm_component_params['uniformRate']  # 交叉概率
#     Algorithm.mutationRate = genetic_algorithm_component_params['mutationRate']  # 突变概率
#     Algorithm.tournamentSize = genetic_algorithm_component_params['tournamentSize']  # 淘汰数组的大小
#     if genetic_algorithm_component_params['elitism'] == 'True':
#         Algorithm.elitism = True
#     else:
#         Algorithm.elitism = False  # 精英主义
#     Population_SIZE = genetic_algorithm_component_params['Population_SIZE']
#     EPOCH = genetic_algorithm_component_params['EPOCH']
#
#     global rt
#     rt = joblib.load(model_path)
#
#     sc = pickle.load(open(std_path, 'rb'))
#
#     myPop = Population(Population_SIZE, True, selected_params, columns)
#     if min_or_max == 'min':
#         fittestIndividual = Individual(selected_params, float('inf'))
#     else:
#         fittestIndividual = Individual(selected_params, float("-inf"))
#     for generationCount in range(EPOCH):
#         # if generationCount % 30 == 0:
#         print("generationCount:", generationCount)
#         temp = myPop.getFittest(min_or_max, selected_params, columns, sc, performance)
#         if min_or_max == 'max' and (temp.getFitness(selected_params, columns, sc, performance,
#                                                     min_or_max) > fittestIndividual.getFitness(
#             selected_params, columns, sc, performance, min_or_max)):
#             fittestIndividual.setGenes(temp.getGenes())
#             fittestIndividual.setFitness(temp.getFitness(selected_params, columns, sc, performance, min_or_max))
#         elif min_or_max == 'min' and (temp.getFitness(selected_params, columns, sc, performance,
#                                                       min_or_max) < fittestIndividual.getFitness(
#             selected_params, columns, sc, performance, min_or_max)):
#             fittestIndividual.setGenes(temp.getGenes())
#             fittestIndividual.setFitness(temp.getFitness(selected_params, columns, sc, performance, min_or_max))
#         # print(fittestIndividual.genes)
#         myPop = Algorithm.evolvePopulation(myPop, selected_params, columns, min_or_max, sc, performance)
#
#     fitParameter = fittestIndividual.transfromGene(selected_params, columns)
#     y = get_performance(fitParameter, config_params, performance)
#     # y=0 #测试使用
#     fitParameter[performance] = y
#
#     # 保存结果
#     res = pd.DataFrame(fitParameter, index=[0], columns=columns)
#     fileName = 'best.csv'
#
#     name = os.path.join(datadir, fileName)
#     res.to_csv(name, index=False)
#     #
#     # name2 = os.path.join(datadir, 'rfhoc_file.txt')
#     # file_out = open(name2, 'w+')
#     # file_out.write(str(name))
#     # file_out.close()
#     # print("rfhoc_result:" + str(name))
#
#     # df_samples = fitParameter
#     #
#     # best_data = {}
#     # best_data[performance] = df_samples.pop(performance)
#     # best_data['paramList'] = df_samples
#     #
#     # print(best_data)
#     # data(best_data, config_params)
#
#     # namecsv = uuid.uuid1().__str__() + '.csv'
#     # name = os.path.join(output_dir, namecsv)
#     # df_samples.to_csv(name, index=False)
#
#     name2 = os.path.join(datadir, 'GA_file.txt')
#     file_out = open(name2, 'w+', encoding='utf-8')
#     file_out.write(str(res))
#     file_out.close()
#
#
# def run():
#     args = parse_arguments()
#     path = args.random_forest_file
#     file_read = open(path, 'r', encoding='utf-8')
#     params = file_read.read()
#     # print("GA_shu_ru_nei_rong:" + params)
#     params = dict(eval(params))
#     # print(params)
#
#     temp = params.copy()
#     performance = temp.pop('performance')
#     columns = temp.pop('columns')
#     path = temp.pop('random_sample_result_path')
#     min_or_max = temp.pop('min_or_max')
#     std_path = temp.pop('std_path')
#     model_path = temp.pop('model_path')
#     selected_params = temp
#
#     genetic_algorithm_component_params = eval("".join(args.genetic_algorithm_component_params))
#     config_params = eval("".join(args.config_params))
#
#     # print("genetic_algorithm_component_params:", genetic_algorithm_component_params)
#     # print("config_params:", config_params)
#
#     genetic_algorithm(columns, selected_params, performance, min_or_max,
#                       config_params, model_path, std_path, genetic_algorithm_component_params)
#
#     file_read.close()
#
#
# if __name__ == '__main__':
#     run()
"""上面是第一版代码"""

# # -*- encoding: utf-8 -*-
# """
#  @Time : 2021/3/9 10:38
#  @Author : zspp
#  @File : GA
#  @Software: PyCharm
# """
# import random
# import numpy as np
# import math
# import pandas as pd
# import argparse
# import os
# import copy
# import pickle
# import joblib
# import time
#
# # from nfs.colosseum.report import *
# # from nfs.colosseum.getPerformance import get_performance
# from getPerformance import get_performance
#
# datadir = os.path.dirname((os.path.abspath(__file__)))
#
#
# # print("GA_datadir:" + datadir)
#
#
# def parse_arguments():
#     """
#     参数解析
#     :return:
#     """
#     parser = argparse.ArgumentParser(description='GA')
#     parser.add_argument('--random_forest_file', type=str, required=True, help='random_forest_file')
#     parser.add_argument('--genetic_algorithm_component_params', type=str, required=True, nargs='+',
#                         help='genetic_algorithm_component_params')
#     parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')
#
#     args = parser.parse_args()
#     return args
#
#
# def data_in(data, selected_params, columns, performance, min_or_max):  # 每次只传入一组参数，所以与acgan、bogan的数据预处理不一样
#     # print(data)
#     # if min_or_max == 'max':
#     #     data = data.sort_values(by=performance, ascending=False)
#     # else:
#     #     data = data.sort_values(by=performance, ascending=True)
#     data = data.drop(performance, 1)
#     # print(data)
#     ##首先处理没有优先关系的enum参数，即string参数
#     char = []  # enum的列名称
#     enum_index = []  # enum的原始列索引
#     for name in columns[:-1]:
#         if selected_params[name][0] == 'string' or selected_params[name][0] == 'enum':
#             char.append(name)
#             enum_index.append(columns.index(name))
#
#     enum_number = []  # 每个enum参数对应的独热编码的长度
#     enum_book = {}  # 每个enum参数的内容，字典形式存储
#     m = 0
#     for c in char:
#         i = enum_index[m]
#
#         new_data = pd.DataFrame({c: selected_params[c][1]})  # 添加几行，为了更好全面编码
#         data = data.append(new_data, ignore_index=True)
#
#         enum_book[c] = list(pd.get_dummies(data[c]).columns)
#         enum_data = pd.get_dummies(data[c], prefix=c)  # 独热编码后的变量
#
#         data = data.drop(c, 1)
#
#         enum_list = list(enum_data.columns)
#         enum_number.append(len(enum_list))
#
#         for k in range(len(enum_list)):
#             data.insert(i + k, enum_list[k], enum_data[enum_list[k]])  # 将向量移动到原来枚举值的位置
#         m = m + 1
#         enum_index = [j + len(enum_data.columns) - 1 for j in enum_index]  # 更新enum_index
#
#         data.drop(data.index[-len(selected_params[c][1]):], inplace=True)  # 删除前3行
#
#         # print(enum_index)
#     # print(enum_number)
#     # print(data)
#     # print(enum_book)
#
#     ##接着处理有优先关系的参数
#     # char2 = []
#     # enum_index2 = {}
#     # for name in columns[:-1]:
#     #     if selected_params[name][0] == 'enum':
#     #         char2.append(name)
#     #         enum_index2[name] = selected_params[name][1]
#     # for enum_name in char2:
#     #     size_range = list(range(len(enum_index2[enum_name])))
#     #     mapping = dict(zip(enum_index2[enum_name], size_range))
#     #     data[enum_name] = data[enum_name].map(mapping)
#     return data
#
#
# def get_performance_rt(pop, sc, selected_params, columns, performance, min_or_max):
#     params = data_in(pop, selected_params, columns, performance, min_or_max)
#     params = params.values
#     params = sc.transform(params.astype(float))
#     y_pred = rt.predict(params)
#     pop.iloc[:, -1] = y_pred
#     return pop
#
#
# def generatepop(selected_params, popsize, columns):
#     pop = []
#     for i in range(popsize):
#         gene = [random.random() for i in range(len(selected_params) + 1)]
#         pop.append(gene)
#     pop = pd.DataFrame(pop, columns=columns)
#     return pop
#
#
# def transfromGene(pop, selected_params, columns):
#     for i in columns[:-1]:
#         para_type = selected_params[i][0]
#         if para_type == 'int':
#             pop[i] = pop[i].apply(lambda x: int(
#                 math.floor(x * (selected_params[i][1][1] - selected_params[i][1][0] + 1)) +
#                 selected_params[i][1][0]))
#
#         elif para_type == 'float' or para_type == 'double':
#             pop[i] = pop[i].apply(lambda x: float(
#                 math.floor(x * (selected_params[i][1][1] - selected_params[i][1][0] + 1)) +
#                 selected_params[i][1][0]))
#         elif para_type == 'enum' or 'string':
#             choice = selected_params[i][1]  # 离散变量内容
#             length = len(choice)  # 离散种类数目
#             gene_choice = np.linspace(0.0, 1.0, num=length + 1)
#             pop[i] = pop[i].apply(lambda x: choice[(np.where(gene_choice > x))[0][0] - 1])
#     return pop
#
#
# def getFittest(pop, min_or_max, sc, performance, selected_params, columns):
#     dt_pop = transfromGene(pop, selected_params, columns)
#     dt_pop = get_performance_rt(dt_pop, sc, selected_params, columns, performance, min_or_max)
#     if min_or_max == 'max':
#         index = np.argmax(pop.iloc[:, -1])
#     else:
#         index = np.argmin(pop.iloc[:, -1])
#     return index, dt_pop.iloc[index, -1], dt_pop
#
#
# # 进化一个种群
# def evolvePopulation(pop, selected_params, columns, min_or_max, sc, performance, index):
#     # 存放新一代的种群
#     newPopulation = pd.DataFrame()
#     # 把最优秀的那个放在第一个位置
#     if elitism:
#         newPopulation = newPopulation.append(pd.DataFrame(dict(pop.iloc[index, :]),index=[0] ,columns=columns),ignore_index=True)
#     # Crossover population
#     if elitism:
#         elitismOffset = 1
#     else:
#         elitismOffset = 0
#
#     for i in range(elitismOffset, pop.shape[0]):
#         # 随机选择两个优秀的个体
#         indiv1 = tournamentSelection(pop, min_or_max)
#         indiv2 = tournamentSelection(pop, min_or_max)
#         # 进行交叉
#         newIndiv = crossover(indiv1, indiv2, selected_params)
#         # Mutate population  突变
#         tempIndiv = mutate(newIndiv, selected_params)
#         newPopulation = newPopulation.append(pd.DataFrame(tempIndiv, index=[0]), ignore_index=True)
#
#     return newPopulation
#
#
# # 进行两个个体的交叉，交叉的概率为uniformRate
# def crossover(indiv1, indiv2, selected_params):
#     newSol = {}
#     # 随机的从两个个体中选择
#     for i, key in zip(range(len(selected_params)), selected_params.keys()):
#         if (random.random() <= uniformRate):
#             newSol[key] = indiv1[key]
#         else:
#             newSol[key] = indiv2[key]
#     return newSol
#
#
# # 突变个体，突变的概率为 mutationRate
# def mutate(indiv, selected_params):
#     random_id = [random.random() for _ in range(len(selected_params))]
#     for i, key in zip(random_id, selected_params.keys()):
#         if (i <= mutationRate):
#             indiv[key] = random.random()
#     return indiv
#
#
# # 随机选择一个较优秀的个体，用于进行交叉
# def tournamentSelection(pop, min_or_max):
#     # Create a tournament population
#     randomId = [random.randint(0, pop.shape[0] - 1) for _ in range(tournamentSize)]
#     if min_or_max == 'max':
#         index = np.argmax(np.array(pop.iloc[randomId, -1:]))
#     else:
#         index = np.argmin(np.array(pop.iloc[randomId, -1:]))
#
#     return dict(pop.iloc[randomId[index], :])
#
#
# def genetic_algorithm(columns, selected_params, performance, min_or_max, config_params, model_path, std_path,
#                       genetic_algorithm_component_params):
#     Population_SIZE = genetic_algorithm_component_params['Population_SIZE']
#     EPOCH = genetic_algorithm_component_params['EPOCH']
#
#     global rt
#     rt = joblib.load(model_path)
#
#     sc = pickle.load(open(std_path, 'rb'))
#
#     myPop = generatepop(selected_params, Population_SIZE, columns)
#     if min_or_max == 'min':
#         fittestIndividual = float("inf")
#     else:
#         fittestIndividual = float("-inf")
#
#     best_data = pd.DataFrame(index=[0], columns=columns)
#     for generationCount in range(EPOCH):
#         if generationCount % 100 == 0:
#             print("generationCount:", generationCount)
#
#         index, value, dt_pop = getFittest(myPop.copy(deep=True), min_or_max, sc, performance, selected_params, columns)
#         if min_or_max == 'max' and (value > fittestIndividual):
#             best_data.iloc[0, :] = dt_pop.iloc[index, :]
#             fittestIndividual = value
#
#         elif min_or_max == 'min' and (value < fittestIndividual):
#             best_data.iloc[0, :] = dt_pop.iloc[index, :]
#             fittestIndividual = value
#
#         myPop.iloc[:, -1] = dt_pop.iloc[:, -1]
#         myPop = evolvePopulation(myPop, selected_params, columns, min_or_max, sc, performance, index)
#     y = get_performance(dict(best_data.iloc[:, -1]), config_params, performance)
#     best_data.iloc[0, performance] = y
#
#     # 保存结果
#     fileName = 'best.csv'
#     name = os.path.join(datadir, fileName)
#     best_data.to_csv(name, index=False)
#
#     name2 = os.path.join(datadir, 'GA_file.txt')
#     file_out = open(name2, 'w+', encoding='utf-8')
#     file_out.write(str(best_data))
#     file_out.close()
#
#
# def run():
#     args = parse_arguments()
#     path = args.random_forest_file
#     file_read = open(path, 'r', encoding='utf-8')
#     params = file_read.read()
#     # print("GA_shu_ru_nei_rong:" + params)
#     params = dict(eval(params))
#     # print(params)
#
#     temp = params.copy()
#     performance = temp.pop('performance')
#     columns = temp.pop('columns')
#     path = temp.pop('random_sample_result_path')
#     min_or_max = temp.pop('min_or_max')
#     std_path = temp.pop('std_path')
#     model_path = temp.pop('model_path')
#     selected_params = temp
#
#     genetic_algorithm_component_params = eval("".join(args.genetic_algorithm_component_params))
#     config_params = eval("".join(args.config_params))
#
#     global uniformRate, mutationRate, tournamentSize, elitism
#     uniformRate = genetic_algorithm_component_params['uniformRate']  # 交叉概率
#     mutationRate = genetic_algorithm_component_params['mutationRate']  # 突变概率
#     tournamentSize = genetic_algorithm_component_params['tournamentSize']  # 淘汰数组的大小
#     if genetic_algorithm_component_params['elitism'] == 'True':
#         elitism = True
#     else:
#         elitism = False  # 精英主义
#
#     # print("genetic_algorithm_component_params:", genetic_algorithm_component_params)
#     # print("config_params:", config_params)
#
#     genetic_algorithm(columns, selected_params, performance, min_or_max,
#                       config_params, model_path, std_path, genetic_algorithm_component_params)
#
#     file_read.close()
#
#
# if __name__ == '__main__':
#     run()
"""上面是第二版代码，还是太慢"""

# -*- encoding: utf-8 -*-
"""
 @Time : 2021/3/9 10:38
 @Author : zspp
 @File : GA
 @Software: PyCharm
"""
import random
import numpy as np
import math
import pandas as pd
import argparse
import os
import copy
import pickle
import joblib
import time

# from nfs.colosseum.report import *
# from nfs.colosseum.getPerformance import get_performance
from getPerformance import get_performance

datadir = os.path.dirname((os.path.abspath(__file__)))


# print("GA_datadir:" + datadir)


def parse_arguments():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='GA')
    parser.add_argument('--random_forest_file', type=str, required=True, help='random_forest_file')
    parser.add_argument('--genetic_algorithm_component_params', type=str, required=True, nargs='+',
                        help='genetic_algorithm_component_params')
    parser.add_argument('--config_params', type=str, required=True, nargs='+', help='config_params')

    args = parser.parse_args()
    return args


def data_in(data, selected_params, columns, performance, min_or_max):  # 每次只传入一组参数，所以与acgan、bogan的数据预处理不一样
    # print(data)
    # if min_or_max == 'max':
    #     data = data.sort_values(by=performance, ascending=False)
    # else:
    #     data = data.sort_values(by=performance, ascending=True)
    data = data.drop(performance, 1)
    # print(data)
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
    #     mapping = dict(zip(enum_index2[enum_name], size_range))
    #     data[enum_name] = data[enum_name].map(mapping)
    return data


def get_performance_rt(pop, sc, selected_params, columns, performance, min_or_max):
    params = data_in(pop, selected_params, columns, performance, min_or_max)
    params = params.values

    params = sc.transform(params.astype(float))
    y_pred = rt.predict(params)
    pop.iloc[:, -1] = y_pred
    return pop


def generatepop(selected_params, popsize, columns):
    pop = np.array([random.random() for i in range(len(selected_params) + 1)])
    for i in range(1, popsize):
        gene = np.array([random.random() for i in range(len(selected_params) + 1)])
        pop = np.row_stack((pop, gene))
    return pop


def transfromGene(pop, selected_params, columns):
    pop = pd.DataFrame(pop, columns=columns)
    for j, i in zip(range(len(columns) - 1), columns[:-1]):
        para_type = selected_params[i][0]
        if para_type == 'int':
            pop[i] = pop[i].apply(lambda x: int(
                math.floor(x * (selected_params[i][1][1] - selected_params[i][1][0] - 1)) +
                selected_params[i][1][0] + 1))

        elif para_type == 'float' or para_type == 'double':
            pop[i] = pop[i].apply(lambda x: float(
                (x * (selected_params[i][1][1] - selected_params[i][1][0]-0.1)) +
                selected_params[i][1][0] + 0.1))
        elif para_type == 'enum' or 'string':
            choice = selected_params[i][1]  # 离散变量内容
            length = len(choice)  # 离散种类数目
            gene_choice = np.linspace(0.0, 1.0, num=length + 1)
            pop[i] = pop[i].apply(lambda x: choice[(np.where(gene_choice > x))[0][0] - 1])
    return pop


def getFittest(pop, min_or_max, sc, performance, selected_params, columns):
    dt_pop = transfromGene(pop, selected_params, columns)
    dt_pop = get_performance_rt(dt_pop, sc, selected_params, columns, performance, min_or_max)
    if min_or_max == 'max':
        index = np.argmax(dt_pop.iloc[:, -1])
    else:
        index = np.argmin(dt_pop.iloc[:, -1])
    return index, dt_pop.iloc[index, -1], dt_pop


# 进化一个种群
def evolvePopulation(pop, selected_params, columns, min_or_max, sc, performance, index):
    # 存放新一代的种群
    newPopulation = []
    # 把最优秀的那个放在第一个位置
    if elitism:
        newPopulation = np.array(pop[index])
        newPopulation = np.expand_dims(newPopulation, 0)
    # Crossover population
    if elitism:
        elitismOffset = 1
    else:
        elitismOffset = 0

    for i in range(elitismOffset, pop.shape[0]):
        # 随机选择两个优秀的个体

        indiv1 = tournamentSelection(pop, min_or_max)  # np.array
        indiv2 = tournamentSelection(pop, min_or_max)  # np.array
        # 进行交叉
        newIndiv = crossover(indiv1, indiv2, selected_params)  # np.array
        # Mutate population  突变
        tempIndiv = np.array(mutate(newIndiv, selected_params))  # np.array
        tempIndiv = tempIndiv[:] + [0]
        tempIndiv = np.expand_dims(tempIndiv, 0)
        newPopulation = tempIndiv if newPopulation == [] else np.concatenate((newPopulation, tempIndiv))

    return newPopulation


# 进行两个个体的交叉，交叉的概率为uniformRate
def crossover(indiv1, indiv2, selected_params):
    # 随机的从两个个体中选择
    random_id = [random.random() for _ in range(len(selected_params) + 1)]
    newSol = [indiv1[i] if j <= uniformRate else indiv2[i] for i, j in zip(range(len(random_id)), random_id)]

    return newSol


# 突变个体，突变的概率为 mutationRate
def mutate(indiv, selected_params):
    random_id = [random.random() for _ in range(len(selected_params) + 1)]
    indiv = [random.random() if j <= mutationRate else indiv[i] for i, j in zip(range(len(random_id)), random_id)]
    return indiv


# 随机选择一个较优秀的个体，用于进行交叉
def tournamentSelection(pop, min_or_max):
    # Create a tournament population
    randomId = [random.randint(0, pop.shape[0] - 1) for _ in range(tournamentSize)]
    if min_or_max == 'max':
        index = np.argmax(pop[randomId, -1:])
    else:
        index = np.argmin(pop[randomId, -1:])

    return pop[randomId[index], :]


def genetic_algorithm(columns, selected_params, performance, min_or_max, config_params, model_path, std_path,
                      genetic_algorithm_component_params):
    Population_SIZE = genetic_algorithm_component_params['Population_SIZE']
    EPOCH = genetic_algorithm_component_params['EPOCH']

    global rt
    rt = joblib.load(model_path)

    sc = pickle.load(open(std_path, 'rb'))

    myPop = generatepop(selected_params, Population_SIZE, columns)
    if min_or_max == 'min':
        fittestIndividual = float("inf")
    else:
        fittestIndividual = float("-inf")

    best_data = pd.DataFrame(index=[0], columns=columns)
    for generationCount in range(EPOCH):
        if generationCount % 300 == 0:
            print("generationCount:", generationCount)
        index, value, dt_pop = getFittest(np.copy(myPop), min_or_max, sc, performance, selected_params, columns)

        if min_or_max == 'max' and (value > fittestIndividual):
            best_data.iloc[0, :] = dt_pop.iloc[index, :]
            fittestIndividual = value

        elif min_or_max == 'min' and (value < fittestIndividual):
            best_data.iloc[0, :] = dt_pop.iloc[index, :]
            fittestIndividual = value

        myPop[:, -1] = dt_pop.iloc[:, -1]

        myPop = evolvePopulation(myPop, selected_params, columns, min_or_max, sc, performance, index)

    dict_best = dict(best_data.iloc[0, :-1])
    for i in selected_params.keys():
        if selected_params[i][0] == 'int':
            dict_best[i] = int(dict_best[i])

    y = get_performance(dict_best, config_params, performance)
    best_data[performance] = y

    # 保存结果
    fileName = 'best.csv'
    name = os.path.join(datadir, fileName)
    best_data.to_csv(name, index=False)

    name2 = os.path.join(datadir, 'GA_file.txt')
    file_out = open(name2, 'w+', encoding='utf-8')
    file_out.write(str(best_data))
    file_out.close()


def run():
    args = parse_arguments()
    path = args.random_forest_file
    file_read = open(path, 'r', encoding='utf-8')
    params = file_read.read()
    # print("GA_shu_ru_nei_rong:" + params)
    params = dict(eval(params))
    # print(params)

    temp = params.copy()
    performance = temp.pop('performance')
    columns = temp.pop('columns')
    path = temp.pop('random_sample_result_path')
    min_or_max = temp.pop('min_or_max')
    std_path = temp.pop('std_path')
    model_path = temp.pop('model_path')
    sample_num = temp.pop('sample_num')
    selected_params = temp

    genetic_algorithm_component_params = eval("".join(args.genetic_algorithm_component_params))
    config_params = eval("".join(args.config_params))

    global uniformRate, mutationRate, tournamentSize, elitism
    uniformRate = genetic_algorithm_component_params['uniformRate']  # 交叉概率
    mutationRate = genetic_algorithm_component_params['mutationRate']  # 突变概率
    tournamentSize = genetic_algorithm_component_params['tournamentSize']  # 淘汰数组的大小
    if genetic_algorithm_component_params['elitism'] == 'True':
        elitism = True
    else:
        elitism = False  # 精英主义

    # print("genetic_algorithm_component_params:", genetic_algorithm_component_params)
    # print("config_params:", config_params)

    genetic_algorithm(columns, selected_params, performance, min_or_max,
                      config_params, model_path, std_path, genetic_algorithm_component_params)

    file_read.close()


if __name__ == '__main__':
    run()
