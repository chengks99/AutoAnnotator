import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

class Evaluation(object):
    def __init__(self, modelf):
        print ('Inherit Evaluation Module')
        self.outPrefix = modelf.replace('.h5', '')
    
    def _tpr_tnr (self, matrix):
        _tnr = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][0]))
        _tpr = float(matrix[1][1]) / (float(matrix[0][1]) + float(matrix[1][1])) 
        return _tpr, _tnr

    def _basic_cf (self, act, pred):
        conf_matrix = confusion_matrix(act.argmax(axis=1), pred.argmax(axis=1))
        print(conf_matrix)
        outList = np.array(range(len(act[0])))
        multiLabel_matrix = multilabel_confusion_matrix(act.argmax(axis=1), pred.argmax(axis=1), labels=outList)
        print(multiLabel_matrix)

        resDic = {}
        for x in range(len(multiLabel_matrix)):
            labelling = self.labelling[x]
            if not labelling in resDic:
                resDic[labelling] = {}
            resDic[labelling]['tpr'], resDic[labelling]['tnr'] = self._tpr_tnr(multiLabel_matrix[x])
        
        
        df = pd.DataFrame(resDic)
        print (df)


    def _out_by_class (self, data, act, pred, eClass):
        if not eClass is None:
            objType = data[eClass].to_numpy().tolist()
        else:
            objType = data[self.objLabelHead].to_numpy().tolist()
        actLabel = act.argmax(axis=1)
        predLabel = pred.argmax(axis=1)
        resDic = {}
        for x in range(len(act)):
            ac = int(actLabel[x])
            pe = int(predLabel[x])
            if not ac in resDic:
                resDic[ac] = {}
            t = objType[x]
            if t is None: continue
            if not t in resDic[ac]:
                resDic[ac][t] = {'correct': 0, 'wrong': 0}
            if ac == pe:
                resDic[ac][t]['correct'] += 1
            else:
                resDic[ac][t]['wrong'] += 1
        for k, v in resDic.items():
            for tk, tv in v.items():
                pct = tv['correct'] / (tv['correct'] + tv['wrong'])
                print ('{}, type: {}, acc: {}'.format(k, tk, pct))

    def classification_result (self, data, act, pred, eClass, modelf, labelling):\
        self.labelling = labelling
        self._basic_cf(act, pred)
        self._out_by_class(data, act, pred, eClass)

    def regression_result (self, act, pred, eClass, modelf):
        resList = []
        for a, p in zip(list(act), list(pred)):
            print ('{:.2f} -> {:.2f}, diff: {:.2f}'.format(a, p[0], a-p[0]))
            resList.append(a - p[0])
        resList = np.array(resList)

        mean = np.mean(resList)
        std = np.std(resList)

        print ('Regression Result: Mean: {:.2f}%, Std: {:.2f}%'.format(mean, std))