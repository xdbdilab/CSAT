import subprocess
import csv
import random
import pandas as pd
import numpy as np

PATH = "./x264/"
    #Location of software and video (the same path)

class X264:
    def __init__(self, size = 300):
        self.x264_data(size)
        print('Data is saved to: ' + PATH)

    @staticmethod
    def getPerformance(no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref):

        config_name = ['no-8x8dct','no-cabac','no-deblock','no-fast-pskip','no-mbtree','no-mixed-refs','no-weightb','rc-lookahead','ref']
        config = [no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref]

        project = PATH + 'x264.exe '
        for i in range(len(config)-2):
            if config[i] == 1:
                project += ('--' + config_name[i] + ' ')
        for i in list([-1,-2]):
            project += ('--' + config_name[i] + ' '+str(config[i])+' ')

        project += '-o '+ PATH + 'video2.mp4 ' + PATH + 'video1.mp4'


        res,out = subprocess.getstatusoutput(project)
        try:
            out = float(out[-13:-5])
        except:
            out = -1
        return out

    def x264_data(self, size = 1000, file_name = "actgan_x264_300.csv"):
        file1 = open(PATH+"x264.csv", "a", newline="")
        content = csv.writer(file1)
        configuration = pd.read_csv(PATH+file_name)

        for i in range(size):
            project = list([])
            no_8x8dct = configuration['no_8x8dct'][i]
            project.append(no_8x8dct)
            no_cabac = configuration['no_cabac'][i]
            project.append(no_cabac)
            no_deblock = configuration['no_deblock'][i]
            project.append(no_deblock)
            no_fast_pskip = configuration['no_fast_pskip'][i]
            project.append(no_fast_pskip)
            no_mbtree = configuration['no_mbtree'][i]
            project.append(no_mbtree)
            no_mixed_refs = configuration['no_mixed_refs'][i]
            project.append(no_mixed_refs)
            no_weightb = configuration['no_weightb'][i]
            project.append(no_weightb)
            rc_lookahead = configuration['rc_lookahead'][i]
            project.append(rc_lookahead)
            ref = configuration['ref'][i]
            project.append(ref)


            result = self.getPerformance(no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref)

            project.append(result)
            print(i, ':', project)
            content.writerow(project)

        file1.close()

        # Illegal configuration removal
        data = pd.read_csv(PATH+"x264.csv")
        b = []
        for i in range(len(data['PERF'])):
            if data['PERF'][i] == -1:
                b.append(i)
        data = data.drop(b)
        data.to_csv(PATH + "x264.csv", index=0)

if __name__ == "__main__":
    X264.x264_data()
