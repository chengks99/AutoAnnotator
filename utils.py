import os, json, copy, pickle, uuid
import numpy as np
import pandas as pd

class LabelMeExtractor (object):
    def __init__(self, lmDir):
        self.dir = lmDir
        if not os.path.isdir(self.dir):
            raise ValueError('Unable to locate labelme directory: {}'.format(self.dir))
        self.outf = os.path.join(self.dir, '{}.pickle'.format(os.path.basename(self.dir)))
        print (self.outf)
    
    def _extract_obj_data (self, jsonf, isKitti=False):
        with open(jsonf, 'r') as f: data = json.load(f)
        if isKitti:
            objShape = data.pop('shape', None)
        else:
            objShape = data.pop('shapes', None)
        if objShape is None: return None
        if 'flags' in data:
            fDict = data.pop('flags')
            data.update(fDict)
        data['imagePath'] = os.path.join(os.path.dirname(jsonf), data['imagePath'])
        objList = []
        for obj in objShape:
            oDict = copy.deepcopy(data)
            if 'flags' in obj and not isKitti:
                flags = obj.pop('flags')
                oDict.update(flags)
            oDict.update(obj)
            oDict['indexID'] = str(uuid.uuid4())
            objList.append(oDict)
        return objList

    def extraction (self):
        objList = []
        for root, dirs, files in os.walk(self.dir):
            for f in files:
                if not f.endswith('.json'): continue
                jList = self._extract_obj_data(os.path.join(root, f))
                if not jList is None:
                    objList.extend(jList)
        df = pd.DataFrame(objList)
        return df
    
    def save_file (self, df):
        with open(self.outf, 'wb') as handle:
            pickle.dump(df, handle)
        return self.outf

class TrainTestSplitter (object):
    def __init__(self, dfPath):
        if not os.path.isfile(dfPath):
            raise ValueError('Unable to locate datafile {}'.format(dfPath))
        with open(dfPath, 'rb') as handle:
            self.df = pickle.load(handle)
    
    def _single_data_checker (self, imgList, lblList):
        from collections import Counter
        _counter = Counter(lblList)

        acceptList, ignoreList = [], []
        for k, v in Counter(lblList).items():
            if v == 1: 
                ignoreList.append(k)
            else:
                acceptList.append(k)

        _imgList, _lblList = [], []
        for i, l in zip(imgList, lblList):
            if l in acceptList:
                _imgList.append(i)
                _lblList.append(l)
        imgList = np.array(_imgList)
        lblList = np.array(_lblList)
        return imgList, lblList, np.array(ignoreList)

    def data_checker (self, traindf, testdf, objType, ratio):
        print ('***********************')
        maxTrain = float(ratio['max'] + ratio['step']) / 100.
        minTrain = float(ratio['max'] - ratio['step']) / 100.
        for o in objType:
            _train = traindf[traindf.label == o]
            _test = testdf[testdf.label == o]

            _trainSize = float(_train.shape[0])
            _testSize = float(_test.shape[0])
            _totalSize = _trainSize + _testSize

            _trainRatio = _trainSize / _totalSize
            _testRatio = _testSize / _totalSize

            if _trainRatio > maxTrain or _trainRatio < minTrain:
                print ('{} ratio not met {} [{}-{}]'.format(0, _trainRatio, maxTrain, minTrain))
                return False

            print ('{}: Train: {}({:.2f}%), Test: {}({:.2f}%)'.format(o, _trainSize, _trainRatio * 100., _testSize, _testRatio * 100.))
        return True
    
    def _save_output (self, traindf, testdf):
        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        trainf = os.path.join(os.getcwd(), '{}-traindf.pickle'.format(dt_string))
        testf = os.path.join(os.getcwd(), '{}-testdf.pickle'.format(dt_string))
        with open(trainf, 'wb') as handle:
            pickle.dump(traindf, handle)
        with open(testf, 'wb') as handle:
            pickle.dump(testdf, handle)
        return trainf, testf

    def train_test_split (self, ratio={'max': 70, 'min': 30, 'step': 10}):
        # get imgList and it unique object list
        imgList = self.df.imagePath.unique()
        lblList = []
        for il in self.df.imagePath.unique():
            _df = self.df[self.df.imagePath == il]
            _labelList = sorted(_df.label.unique().tolist())
            lblList.append('{}'.format('-'.join(l for l in _labelList)))
        lblList = np.array(lblList)

        # ignore sample which only contains 1 data
        imgList, lblList, ignoreList = self._single_data_checker(imgList, lblList)

        # split train test
        from sklearn.model_selection import StratifiedShuffleSplit
        ss = StratifiedShuffleSplit(n_splits=5, test_size=float(ratio['min'])/100., random_state=16)
        SPLIT = True
        trainf, testf = None, None
        while (SPLIT):
            for train_index, test_index in ss.split(imgList, lblList):
                X_train, X_test = imgList[train_index], imgList[test_index]
                y_train, y_test = lblList[train_index], lblList[test_index]

                traindf = self.df[self.df.imagePath.isin(X_train)]
                testdf = self.df[self.df.imagePath.isin(X_test)]
                res = self.data_checker(traindf, testdf, self.df.label.unique(), ratio)
                if res:
                    trainf, testf = self._save_output(traindf, testdf)
                    SPLIT = False
                    break
        return trainf, testf


            


