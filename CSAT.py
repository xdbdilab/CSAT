"""
Created on August 1st 12:00:05 2021
CSAT
@author:
"""
import numpy as np
import random
from time import time
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import *

# Configuration_Name: Interpretation

class CSAT:

    def __init__(self, XY, bound = -1, int_flag = []):
        self.XY = np.array(XY)
        for i in range(len(self.XY)):
            self.XY[i, -1] = self.XY[i, -1] + random.random()*2 - 1
        self.X = self.XY[:, 0:-1]
        self.Y = self.XY[:, -1]

        if len(int_flag) == 0:
            self.int_flag = np.ones(self.X.shape[1])
        else:
            self.int_flag = int_flag

        if bound == -1:
            bound = np.array([[]])
            for i in range(len(self.X)):
                bound = np.append(bound, [np.array([0, 10000])], axis = 1)
            self.bound = bound
        else:
            self.bound = np.array(bound)


        self.Lock_value_mask()
        self.anfis = ANFIS(XY)
        self.Perceive()
        self.Comperators_1()

    def Perceive(self):
        # Initial distribution number
        Initial_distribution_number = 0
        Dominance_distribution_number = 6
        XY = ANFIS.Union(self.XY)
        temp = np.sort(-XY[:, -1])
        negpos = np.where(temp < 0)[0]
        try:
            Dominance_distribution_number = min([Dominance_distribution_number, negpos[-1]])
            Initial_distribution_number += 6-Dominance_distribution_number
            MR = np.zeros((np.min([self.X.shape[0], Initial_distribution_number]), self.X.shape[1]))
            pos = list(range(len(self.XY)))
            random.shuffle(pos)
            for i in range(len(MR)):
                MR[i] = self.anfis.x2MR(self.X[pos[i]])
            X = XY[np.argsort(-XY[:, -1])[0:min([Dominance_distribution_number, negpos[-1]])], 0:-1]
            for i in range(len(X)):
                MR = np.append(MR, [self.anfis.x2MR(X[i])], axis=0)
        except:
            Initial_distribution_number = 6
            MR = np.zeros((np.min([self.X.shape[0], Initial_distribution_number]), self.X.shape[1]))
            pos = list(range(len(self.XY)))
            random.shuffle(pos)
            for i in range(len(MR)):
                MR[i] = self.anfis.x2MR(self.X[pos[i]])

        # MR[MR.shape[0]-1] = self.anfis.x2MR(self.X[np.argmax(self.Y)])

        # self.anfis.MR = self.anfis.Update_MR(MR)
        self.anfis.MR = MR


        self.anfis.ANFIS(Train_index=np.array(range(len(self.XY))), epoch=50)
        self.X_p = self.anfis.Perceive(self.X).transpose()

    def Lock_value_mask(self):
        self.mask = np.ones(self.X.shape[1])*-233
        b = []
        # for i in range(len(self.mask)):
        #     if len(np.unique(self.X[:, i])) == 1:
        #         self.mask[i] = self.X[0, i]
        #         b.append(i)

        self.X = np.delete(self.X, b, axis = 1)
        self.XY = np.delete(self.XY, b, axis = 1)
        self.int_flag = np.delete(self.int_flag, b, axis=0)

    def Comperators_1(self):
        # Linear regression
        X_p = self.X_p
        # Sample noise
        sigma = 0.05
        # Parameter prior
        Sigma = np.eye(len(self.Interactive(X_p[0],X_p[0])))

        # Data processing
        V = []
        U = []
        for i in range(len(X_p)):
            for j in range(len(X_p)):
                V = np.append(V, self.Interactive(X_p[i], X_p[j]))
                U = np.append(U, self.Comparison_function(self.Y[i], self.Y[j]))
        V = V.reshape([len(X_p)*len(X_p), len(self.Interactive(X_p[0],X_p[0]))])
        U = U.reshape([len(U), 1])

        # Parameter Estimation
        w = np.linalg.pinv((np.dot(V.transpose(), V)) + sigma ** 2 * np.linalg.pinv(Sigma))
        self.w = np.dot(np.dot(w, V.transpose()), U)

        # Parameter posterior
        self.Sigma_w = np.linalg.pinv((np.dot(V.transpose(), V)) / sigma ** 2 + np.linalg.pinv(Sigma))
        mu_w = np.dot(np.dot(self.Sigma_w, V.transpose()), U) / sigma ** 2
        self.sigma = sigma

    def Compare_1(self, x, x_B):
        x = self.anfis.Perceive(np.matrix(x))
        x_B = self.anfis.Perceive(np.matrix(x_B))
        v = self.Interactive(x, x_B)

        # Distribution parameter
        mu_y = np.dot(v.transpose(), self.w)
        sigma_y = np.dot(np.dot(v.transpose(), self.Sigma_w), v) + self.sigma ** 2

        return float(norm.cdf(mu_y / np.sqrt(sigma_y)))

    def Compare_1_o(self, o, o_B):
        v = self.Interactive(o, o_B)

        # Distribution parameter
        mu_y = np.dot(v.transpose(), self.w)
        sigma_y = np.dot(np.dot(v.transpose(), self.Sigma_w), v) + self.sigma ** 2

        return float(norm.cdf(mu_y / np.sqrt(sigma_y)))

    def Comperators_2(self):
        # Logistic regression
        X_p = self.X_p

        # Data processing
        V = []
        U = []
        for i in range(len(X_p)):
            for j in range(len(X_p)):

                u = self.Comparison_function(self.Y[i], self.Y[j])
                if u == 0:
                    continue
                elif u > 0:
                    V = np.append(V, self.Interactive(X_p[i], X_p[j]))
                    U = np.append(U, 1)
                else:
                    V = np.append(V, self.Interactive(X_p[i], X_p[j]))
                    U = np.append(U, 0)
        self.V = V.reshape([len(X_p) * len(X_p) - len(X_p), len(self.Interactive(X_p[0], X_p[0]))])
        self.U = U.reshape([len(U), 1])

    def Comperators_3(self):
        # Classification And Regression Tree
        X_p = self.X_p

    def Recommended_fix(self, nx):
        bound = self.bound
        for i in range(len(nx)):
            for j in range(len(nx[i])):
                if nx[i][j] >= bound[j, 1]:
                    nx[i][j] = bound[j, 1]
                elif nx[i][j] <= bound[j, 0]:
                    nx[i][j] = bound[j, 0]

        return nx

    def Generator_parameters(self, o_B):
        m, n = self.anfis.MR.shape

        # Full probability
        Variable_space_size = np.sqrt(m) / np.math.factorial(m - 1)
        Accumulative_value = 0

        for i in range(5*n):
            o = self.Sum2one(m)
            Accumulative_value += self.Compare_1_o(o, o_B) * Variable_space_size
        Full_P = Accumulative_value / (i + 1)

        # P_1
        P_1 = np.zeros(m)
        for k in range(m):
            o = np.zeros(m)
            o[k] = 1
            P_1[k] = self.Compare_1_o(o, o_B) / Full_P

        # P_0
        P_0 = np.zeros(m)
        Variable_space_size = np.sqrt(m - 1) / np.math.factorial(np.max([2, m]) - 2)
        for k in range(m):
            Accumulative_value = 0
            for i in range(50):
                o = np.zeros(m)
                temp = self.Sum2one(m - 1)
                o[0:k] = temp[0:k]
                o[k + 1:] = temp[k:]
                Accumulative_value += self.Compare_1_o(o, o_B) * Variable_space_size
            P_0[k] = Accumulative_value / (i + 1)

        P_1, P_0 = P_1 / np.sum(P_1), P_0 / np.sum(P_0)
        P_1[np.where(P_1 <= 0.001)] = 0.001
        P_0[np.where(P_0 <= 0.001)] = 0.001

        # Generator parameters
        mu, sigma = np.zeros(m), np.zeros(m)
        for k in range(m):
            mu[k], sigma[k] = self.Gaussian_equation_solving(P_0[k], P_1[k])
            if mu[k] < 0:
                mu[k] = 0.001

        self.G_p = [mu, sigma]

    def Generator(self, x_B, size = 1):
        self.Generator_parameters(self.anfis.Perceive(np.matrix(x_B)))
        m = self.anfis.MR.shape[0]
        o = np.zeros((size, m))
        for i in range(m):
            o[:, i] = np.random.randn(size) * self.G_p[1][i] + self.G_p[0][i]
        for i in range(size):
            o[i] = o[i]/np.sum(o[i])

        return self.Decoder(o, x_B)

    def Decoder(self, o, x_B):
        m = self.anfis.MR.shape[0]
        n = self.anfis.MR.shape[1]
        s = o.shape[0]
        nx = np.zeros((s,n))

        for i in range(n):
            nx[:, i] = x_B[i]
            for j in range(m):
                delta = o[:, j]*(self.anfis.mf[i].mf[int(self.anfis.MR[j][i])].config[1] - x_B[i] + (random.random()-0.5)*x_B[i]*0.4)
                nx[:, i] += delta

        for i in range(n):
            if self.int_flag[i] == 1:
                nx[:, i] = np.round(nx[:, i])

        # Recommended fix (for bound)
        nx = self.Recommended_fix(nx)
        xx = np.zeros((nx.shape[0], len(self.mask)))
        t = 0
        for i in range(len(self.mask)):
            if self.mask[i] == -233:
                xx[:,i] = nx[:,t]
                t += 1

        return xx

    @staticmethod
    def Interactive(x, x_B):
        x = np.matrix(x)
        x_B = np.matrix(x_B)
        x = x.flatten()
        x_B = x_B.flatten()

        #y = x'Ax+x'Bx_B+x_B'Cx_B+d'x+e'x_B
        v = []
        A_x = np.dot(x.transpose(), x)
        v = np.append(v,np.reshape(A_x.transpose(), A_x.shape[0] * A_x.shape[1]))
        B_x = np.dot(x.transpose(), x_B)
        v = np.append(v,np.reshape(B_x.transpose(), B_x.shape[0] * B_x.shape[1]))
        C_x = np.dot(x_B.transpose(), x_B)
        v = np.append(v,np.reshape(C_x.transpose(), C_x.shape[0] * C_x.shape[1]))
        v = np.append(v,x)
        v = np.append(v,x_B)

        return v.reshape([len(v), 1])

    @staticmethod
    def Comparison_function(y, y_B):
        return np.arctan((y-y_B))/np.pi*2

    @staticmethod
    def Sum2one(m):
        r = np.abs(np.random.randn(m))**1.34
        r = r/np.sqrt(np.sum(r**2))
        x = r/np.sum(r)
        return x

    @staticmethod
    def Gaussian_equation_solving(P_0,P_1):
        k = Symbol('k')
        b = Symbol('b')
        c = Symbol('c')
        res = solve([(k * (0 + b)) ** 2 - (P_0 - c) ** 2, (k * (1 + b)) ** 2 - (P_1 - c) ** 2,
                     (k ** 2 * b * (1 + b)) ** 2 - (
                                 ((c * (2 + 4 * b) - 2) ** 2 - (k * b ** 2) ** 2 - (k * (1 + b) ** 2) ** 2) / (
                                     2 * b * (1 + b))) ** 2], [k, b, c])

        # print(res)
        if len(res) == 0:
            res = [(1,1,1)]
        k = np.real(complex(res[-1][0]))
        b = np.real(complex(res[-1][1]))
        c = np.real(complex(res[-1][2]))
        x = Symbol('x')

        mu = integrate(x * (-abs(k * (x + b)) + c), (x, 0, 1))


        sigma2 = integrate((x - mu) ** 2 * (-abs(k * (x + b)) + c), (x, 0, 1))

        if mu < 0:
            mu = -1
        if sigma2 < 0:
            sigma2 = np.exp(float(sigma2))

        return float(mu) +0.1*(random.random()-0.5), np.sqrt(float(sigma2))


class ANFIS:

    def __init__(self, XY, *MR, seed=int(time())):
        # np.random.seed(seed)
        # random.seed(seed)
        if len(MR) > 0:
            self.MR = np.array(MR[0])
        else:
            self.MR = np.array([])
        self.XY = np.array(XY)

        self.Setup()

    def Setup(self):
        self.X = self.XY[:, 0:-1]
        self.Y = self.XY[:, -1]

        Invalid_Data_Index = []
        # for i in range(self.X.shape[1]):
        #     if len(np.unique(self.X[:, i])) == 1:
        #         Invalid_Data_Index.append(i)
        self.X = np.delete(self.X, Invalid_Data_Index, axis=1)

        self.MeanShift()
        self.X2mf()

    def ANFIS(self, Train_index, epoch):

        vdw = sdw = 0
        for t in range(1, epoch + 1):
            self.mFun(self.X[Train_index])
            bMR = self.reMR()

            O_1 = np.copy(self.F)
            O_2 = np.exp(np.dot(bMR.transpose(), np.log(O_1 + 0.0001)))
            O_3 = O_2 / np.dot(np.ones([O_2.shape[0], O_2.shape[0]]), O_2)

            self.Structure_after(self.X[Train_index], self.Y[Train_index], O_3)

            O_4 = np.multiply(np.dot(self.Ac.transpose(), np.append(self.X[Train_index], np.ones(
                [self.X[Train_index].shape[0], 1]), axis=1).transpose()), O_3)

            O_5 = np.sum(O_4, axis=0)

            if self.MRE(O_5, self.Y[Train_index]) < 0.001:
                break

            vdw, sdw = self.reMF(Train_index, bMR, O_5, vdw, sdw, t)

    def Perceive(self, Test_X):
        self.mFun(Test_X)
        bMR = self.reMR()

        O_1 = np.copy(self.F)
        O_2 = np.exp(np.dot(bMR.transpose(), np.log(O_1 + 0.0001)))
        O_3 = O_2 / np.dot(np.ones([O_2.shape[0], O_2.shape[0]]), O_2)
        # O_4 = np.multiply(
        #     np.dot(self.Ac.transpose(), np.append(Test_X, np.ones([Test_X.shape[0], 1]), axis=1).transpose())
        #     , O_3)

        return O_3

    def prediction(self, Test_X):
        self.mFun(Test_X)
        bMR = self.reMR()

        O_1 = np.copy(self.F)
        O_2 = np.exp(np.dot(bMR.transpose(), np.log(O_1 + 0.0001)))
        O_3 = O_2 / np.dot(np.ones([O_2.shape[0], O_2.shape[0]]), O_2)
        O_4 = np.multiply(
            np.dot(self.Ac.transpose(), np.append(Test_X, np.ones([Test_X.shape[0], 1]), axis=1).transpose())
            , O_3)
        O_5 = np.sum(O_4, axis=0)

        return O_5

    def GenMR(self, Test_MR, Initial_number_of_rules, Val_index, *Number_generated_each_time):
        if len(Test_MR) == 0:
            MR = []
            Random_index = list(range(len(Val_index)))
            np.random.shuffle(Random_index)
            for i in range(np.min([len(Val_index), Initial_number_of_rules])):
                MR.append(self.x2MR(self.X[Val_index[Random_index[i]]]))
            return self.Union(np.array(MR))
        else:
            Temp_MR = []
            for i in range(Number_generated_each_time[0]):
                MR = np.copy(Test_MR[random.sample(range(Test_MR.shape[0]), 1)[0]])
                Before_mutated_fragment = random.sample(range(len(MR)), 1)[0]
                Mutatable_set = np.copy(self.CluRe.C[Before_mutated_fragment])
                Mutatable_set = np.delete(Mutatable_set, MR[Before_mutated_fragment], axis=0)
                MR[Before_mutated_fragment] = random.sample(list(Mutatable_set), 1)[0]
                Temp_MR.append(MR)
            return np.array(Temp_MR)

    def x2MR(self, x):
        MR = np.zeros(x.shape)
        for i in range(len(x)):
            m = np.zeros(len(self.mf[i].mf))
            for j in range(len(m)):
                m[j] = self.MF(x[i], self.mf[i].mf[j].type, self.mf[i].mf[j].config)
            MR[i] = np.argmax(m)
        return MR

    def reMF(self, Train_index, bMR, Predicted_Y, vdw, sdw, t):
        s, beta1, beta2, alpha, epsilon = 0, 0.9, 0.9, 0.001, 0.001
        error = Predicted_Y - self.Y[Train_index]
        F = self.F + 0.001

        D_1 = np.dot(bMR, np.multiply(np.dot(self.Ac.transpose(),
                                             np.append(self.X[Train_index], np.ones([len(Train_index), 1]),
                                                       axis=1).transpose()),
                                      np.exp(np.dot(bMR.transpose(), np.log(F))))) / F
        D_2 = np.dot(1 - bMR, np.multiply(
            np.dot(self.Ac.transpose(),
                   np.append(self.X[Train_index], np.ones([len(Train_index), 1]), axis=1).transpose()),
            np.exp(np.dot(bMR.transpose(), np.log(F)))))
        D_3 = np.dot(bMR, np.exp(np.dot(bMR.transpose(), np.log(F)))) / F
        D_4 = np.dot(1 - bMR, np.exp(np.dot(bMR.transpose(), np.log(F))))

        q = np.multiply(D_2, D_3) / (D_1 + 0.001) - D_4
        a = D_3
        b = D_4
        dYdF = np.multiply(q, a) / (np.multiply(q, F) + b + 0.001) ** 2

        dc = []
        dsig = []

        for i in range(len(self.mf)):
            x = self.X[Train_index, i]
            for j in range(len(self.mf[i].mf)):
                sig = self.mf[i].mf[j].config[0]
                c = self.mf[i].mf[j].config[1]
                dFdsig = np.multiply(np.exp(-(x - c) ** 2 / 2 / sig ** 2), (x - c) ** 2) / sig ** 3
                dFdc = np.multiply(np.exp(-(x - c) ** 2 / 2 / sig ** 2), (x - c)) / sig ** 2
                dsig.append(np.dot(error, np.multiply(dYdF[s, :], dFdsig).transpose()))
                dc.append(np.dot(error, np.multiply(dYdF[s, :], dFdc).transpose()))
                s += 1
        dw = np.append(dsig, dc)
        vdw = beta1 * vdw + (1 - beta1) * dw
        sdw = beta2 * sdw + (1 - beta2) * dw ** 2
        vdwc = vdw / (1 - beta1 ** t)
        sdwc = sdw / (1 - beta2 ** t)

        delta = alpha * vdwc / (np.sqrt(sdwc) + epsilon)
        dsig = delta[:len(dc)]
        dc = delta[len(dc):]

        s = 0
        for i in range(len(self.mf)):
            for j in range(len(self.mf[i].mf)):
                self.mf[i].mf[j].config[0] -= dsig[s]
                self.mf[i].mf[j].config[1] -= dc[s]
                s += 1
        return vdw, sdw

    def Structure_after(self, X, Y, O_3):
        x = y = []
        for k in range(X.shape[0]):
            x_t = []
            for i in range(O_3.shape[0]):
                x_t = np.append(x_t, np.append(X[k], 1) * O_3[i, k])
            x_t = np.array(x_t).reshape(1, len(x_t))
            if len(x) == 0:
                x = x_t
            else:
                x = np.append(x, x_t, axis=0)
            y = np.append(y, Y[k])

        y = y.reshape(len(y), 1)
        self.Ac = np.dot(np.dot(np.linalg.pinv(np.dot(x.transpose(), x)), x.transpose()), y).reshape(
            O_3.shape[0], X.shape[1] + 1).transpose()

    def Update_MR(self, Updated_MR):
        socer = self.Rules_Socer(self.X[np.random.choice(list(range(self.X.shape[0])), size=100, replace=True)],
                                 Updated_MR)
        Effective_rules = np.array(np.where(socer > 0.9))
        if len(Effective_rules[0]) > 0:
            Updated_MR = np.delete(Updated_MR, np.where(socer < 0.9), axis=0)
        return self.Union(Updated_MR)

    def Rules_Socer(self, X_sample, MR):
        O_2, O_3 = self.First_Three_Layers(X_sample, MR)
        socer = np.zeros(MR.shape[0])
        for i in range(len(socer)):
            socer[i] = np.max(O_3[i])

        return socer

    def First_Three_Layers(self, X_sample, MR):
        self.mFun(X_sample)
        bMR = self.reMR(MR)

        O_1 = np.copy(self.F)
        O_2 = np.exp(np.dot(bMR.transpose(), np.log(O_1 + 0.0001)))
        O_3 = O_2 / np.dot(np.ones([O_2.shape[0], O_2.shape[0]]), O_2)

        return O_2, O_3

    def mFun(self, X):

        F = Ft = Aux = []
        for x_n in range(X.shape[0]):
            Aux = []
            Ft = []
            for i in range(len(self.mf)):
                Aux.append(len(Ft) + 1)
                for j in range(len(self.mf[i].mf)):
                    Ft.append(self.MF(X[x_n, i],
                                      self.mf[i].mf[j].type,
                                      self.mf[i].mf[j].config))
            F.append(Ft)
        self.F = np.reshape(F, [X.shape[0], len(Ft)]).transpose()
        self.Aux = Aux

    def reMR(self, *MR):
        if len(MR) == 0:
            MR = self.MR
        else:
            MR = MR[0]

        bMR = np.zeros([self.F.shape[0], MR.shape[0]])
        for i in range(MR.shape[0]):
            for j in range(MR.shape[1]):
                bMR[int(np.min([self.F.shape[0], self.Aux[j] + MR[i, j]])) - 1, i] = 1

        return bMR

    def MeanShift(self):
        class CLURE:
            def __init__(self):
                self.C = []

            def append(self, C):
                self.C.append(C)

        self.CluRe = CLURE()

        for i in range(self.X.shape[1]):
            CluRe = self.MS(self.X[:, i])
            if len(CluRe) == 0:
                CluRe = np.array([np.min(self.X[0, i]), np.max(self.X[0, i]) + 0.1])
            self.CluRe.append(CluRe)

    def X2mf(self):

        class MUFUN:
            def __init__(self, type, config):
                self.type = type
                self.config = config

        class MF:
            def __init__(self):
                self.mf = []

            def append(self, mf):
                self.mf.append(mf)

        CC = []
        for i in range(self.X.shape[1]):
            C = np.cov([self.X[:, i], self.Y])
            CC.append(C[0, 1])

        self.mf = []
        for i in range(self.X.shape[1]):
            self.mf.append(MF())
            for j in range(len(self.CluRe.C[i])):
                type = 'gaussmf'
                # config = [np.sqrt((self.CluRe.C[i][1] - self.CluRe.C[i][0])/0.01) / np.abs(CC[i]), self.CluRe.C[i][j]]
                config = [0.01, self.CluRe.C[i][j]]
                self.mf[i].append(MUFUN(type, config))

    @classmethod
    def infocls(cls):
        print(cls)

    @staticmethod
    def MS(x):
        m = 1
        if np.exp(np.max(x)) >= 1e5:
            m = np.max(x)
            x = x / m

        min_x = np.min(x)
        max_x = np.max(x)

        r = 1 / 9 * (max_x - min_x)
        CluRe = np.linspace(min_x, max_x, 10)
        pCluRe = CluRe

        while 1:
            dis = np.abs(np.log(np.dot(np.exp(-np.matrix(x).transpose()), np.exp(np.matrix(CluRe)))))
            ab = np.array(np.where(dis < r))
            a = ab[0, :]
            b = ab[1, :]
            label = np.unique(b)

            for i in label:
                CluRe[i] = np.mean(x[a[np.where(b == i)]])

            if np.max(np.abs(CluRe - pCluRe)) < 1e-10:
                CluRe = np.unique(CluRe[label]) * m
                return CluRe
            CluRe = np.unique(CluRe[label])
            pCluRe = CluRe

    @staticmethod
    def MF(x, type, config, *fun):
        if len(fun) == 0:
            fun = 1

        if type == 'gaussmf':
            if fun == 0:
                return ['exp(-(x-c).^2/2/sig^2', {'c', 'sig'}]
            sig = config[0]
            c = config[1]
            return np.exp(-(x - c) ** 2 / 2 / sig ** 2)

    @staticmethod
    def Union(MR):
        Check_array = np.dot(np.exp(-MR), np.exp(MR.transpose())) - MR.shape[1]
        ij = np.array(np.where(Check_array == 0))
        i, j = ij[0, :], ij[1, :]
        j = np.delete(j, np.where(i - j >= 0), axis=0)
        return np.delete(MR, j, axis=0)

    @staticmethod
    def MRE(Predicted_Y, Real_Y):
        return np.sum(np.abs(Predicted_Y - Real_Y) / Real_Y) / len(Real_Y) * 100

    @staticmethod
    def Sample(Available_index_set, Val_size, Train_size):

        if len(Available_index_set) < Val_size + Train_size:
            print('Check the data set size!')

        Available_index_set = np.array(Available_index_set)
        Val_index_pos = random.sample(list(range(len(Available_index_set))), Val_size)
        Val_index = Available_index_set[Val_index_pos]
        Available_index_set = np.delete(Available_index_set, Val_index_pos)

        Train_index_pos = random.sample(list(range(len(Available_index_set))), Train_size)
        Train_index = Available_index_set[Train_index_pos]
        Rest_index = np.delete(Available_index_set, Train_index_pos)

        return Val_index, Train_index, Rest_index
class SAMPLER:
    def __init__(self, lb = 0, ub = 1, pool = -1, Score_fun = np.sin, epsilon = 0.01):
        self.lb = lb
        self.ub = ub
        self.fun = Score_fun
        self.epsilon = epsilon
        self.pool = pool
        if pool == -1:
            self.Pool()
        else:
            self.pool = pool

        self.Score()

    def Pool(self):
        self.pool = np.random.random(size = 100)*(self.ub - self.lb) + self.lb

    def Score(self):
        self.score = np.copy(self.pool)
        for i in range(len(self.pool)):
            self.score[i] = self.fun(self.pool[i])
            if self.score[i] <= self.epsilon:
                self.score[i] = self.epsilon

    def rands(self, size = 1):
        xita = np.random.random(size)
        cum = np.cumsum(self.score)/np.sum(self.score)
        xita = np.matrix(xita)
        cum = np.matrix(cum)
        Q = 1/np.log(np.dot(np.exp(cum).T, np.exp(-1*xita)))
        return self.pool[np.argmax(Q,axis=0)]