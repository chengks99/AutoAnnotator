import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

class Evaluation(object):
    def __init__(self, modelf):
        print ('Inherit Evaluation Module')
        self.outPrefix = modelf.replace('.h5', '')
    
    # TPR and TNR rate calculation
    def _tpr_tnr (self, matrix):
        _tnr = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][0]))
        _tpr = float(matrix[1][1]) / (float(matrix[0][1]) + float(matrix[1][1])) 
        return _tpr, _tnr

    # seaborn category plotting
    #   df = dataframe
    #   x = header for x-axis
    #   y = header for y-axis
    #   hue = header for hue subject
    #   kind = graph type. Default='bar'
    #   outSuffix = graph output suffix
    #   ylim = range for y-axis
    #   value = True to print out value for each bar
    def sns_catPlot (self, df, x, y, hue, kind='bar', outSuffix='graph.png', ylim=(0,1), value=True):
        if not '.' in outSuffix:
            outSuffix = '{}.png'.format(outSuffix)
        outf = '{}-{}'.format(self.outPrefix, outSuffix)
        title = os.path.basename(outf).split('.')[0]
        splot = sns.catplot(data=df, x=x, y=y, hue=hue, kind=kind, height=8.27, aspect=11.7/8.27).set(title=title)
        splot.set(ylim=ylim)

        if value:
            ax = splot.facet_axis(0,0)
            for p in ax.patches:
                ax.text(p.get_x() + 0.02, 
                        p.get_height() * 1.02, 
                        '{0:.2f}'.format(p.get_height()),
                        color='black',
                        rotation='horizontal',
                        size='large')
        splot.savefig(outf)

    # basic confusion matrix print out and plot preparation
    def _basic_cf (self, act, pred):
        conf_matrix = confusion_matrix(act.argmax(axis=1), pred.argmax(axis=1))
        print(conf_matrix)
        outList = np.array(range(len(act[0])))
        multiLabel_matrix = multilabel_confusion_matrix(act.argmax(axis=1), pred.argmax(axis=1), labels=outList)
        print(multiLabel_matrix)

        resDic = []
        for x in range(len(multiLabel_matrix)):
            labelling = self.labelling[x]
            res = {}
            res['tpr'], res['tnr'] = self._tpr_tnr(multiLabel_matrix[x])
            for k, v in res.items():
                dic = {}
                dic['type'] = labelling
                dic['kind'] = k
                dic['sensitivity'] = v
                resDic.append(dic)
       
        df = pd.DataFrame(resDic)
        self.sns_catPlot(df, x='kind', y='sensitivity', hue='type', outSuffix='sensitivity.png')

    # calculate accuracy for each class and plot the output
    def _out_by_class (self, data, act, pred, eClass):
        labelType = data[self.objLabelHead].to_numpy().tolist()
        objType = []
        if not eClass is None:
            eType = data[eClass].to_numpy().tolist()
            for e, l in zip(eType, labelType):
                objType.append(l if e is None else e)
        else:
            objType = labelType
        actLabel = act.argmax(axis=1)
        predLabel = pred.argmax(axis=1)
        resDic = {}
        for x in range(len(act)):
            ac = int(actLabel[x])
            pe = int(predLabel[x])
            if not ac in resDic:
                resDic[ac] = {}
            t = objType[x]
            if not t in resDic[ac]:
                resDic[ac][t] = {'correct': 0, 'wrong': 0}
            if ac == pe:
                resDic[ac][t]['correct'] += 1
            else:
                resDic[ac][t]['wrong'] += 1
        resPlot = []
        for k, v in resDic.items():
            for tk, tv in v.items():
                pct = tv['correct'] / (tv['correct'] + tv['wrong'])
                print ('{}, type: {}, precision: {}/{} ({:.3f})'.format(self.labelling[k], tk, tv['correct'], (tv['correct'] + tv['wrong']), pct))
                dic = {}
                dic['type'] = self.labelling[k]
                dic['label'] = tk
                dic['acc'] = pct
                resPlot.append(dic)
        df = pd.DataFrame(resPlot)
        self.sns_catPlot(df, x='type', y='acc', hue='label', outSuffix='precision-by-type.png')        

    # evaluator for classification result
    def classification_result (self, data, act, pred, eClass, modelf, labelling):
        self.labelling = labelling
        self._basic_cf(act, pred)
        self._out_by_class(data, act, pred, eClass)

    # evaluator for regression result
    def regression_result (self, act, pred, eClass, modelf):
        resList = []
        for a, p in zip(list(act), list(pred)):
            print ('{:.2f} -> {:.2f}, diff: {:.2f}'.format(a, p[0], a-p[0]))
            resList.append(a - p[0])
        resList = np.array(resList)

        mean = np.mean(resList)
        std = np.std(resList)

        print ('Regression Result: Mean: {:.2f}%, Std: {:.2f}%'.format(mean, std))