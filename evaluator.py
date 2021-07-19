import numpy as np

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

class Evaluation(object):
    def __init__(self):
        print ('Inherit Evaluation Module')
    
    def _basic_cf (self, act, pred):
        conf_matrix = confusion_matrix(act.argmax(axis=1), pred.argmax(axis=1))
        print(conf_matrix)
        outList = np.array(range(len(act[0])))
        multiLabel_matrix = multilabel_confusion_matrix(act.argmax(axis=1), pred.argmax(axis=1), labels=outList)
        print(multiLabel_matrix)

    def _out_by_class (self, data, act, pred, eClass):
        if not eClass is None:
            objType = data[eClass].to_numpy().tolist()
        else:
            objType = data[self.objLabelHead].to_numpy().tolist()
        resDic = {}
        actLabel = act.argmax(axis=1)
        predLabel = pred.argmax(axis=1)
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
        for k, v in resDic.items():
            for tk, tv in v.items():
                pct = tv['correct'] / (tv['correct'] + tv['wrong'])
                print ('{}, type: {}, acc: {}'.format(k, tk, pct))

    def classification_result (self, data, act, pred, eClass):
        self._basic_cf(act, pred)
        self._out_by_class(data, act, pred, eClass)


    def regression_result (self, act, pred, eClass):
        resList = []
        for a, p in zip(list(act), list(pred)):
            print ('{:.2f} -> {:.2f}, diff: {:.2f}'.format(a, p[0], a-p[0]))
            resList.append(a - p[0])
        resList = np.array(resList)

        mean = np.mean(resList)
        std = np.std(resList)

        print ('Regression Result: Mean: {:.2f}%, Std: {:.2f}%'.format(mean, std))