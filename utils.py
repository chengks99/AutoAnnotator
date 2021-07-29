import os, json, copy, pickle, uuid
import numpy as np
import pandas as pd
from datetime import datetime

# Extract LabelMe format json into dataframe
class LabelMeExtractor (object):
    def __init__(self, lmDir, isKitti=False):
        for lm in lmDir:
            if not os.path.isdir(lm):
                raise ValueError('Unable to locate labelme directory of {}'.format(lm))
        self.dir = lmDir
        self.isKitti = isKitti
    
    # extract data for each objects
    def _extract_obj_data (self, jsonf):
        with open(jsonf, 'r') as f: data = json.load(f)
        if self.isKitti:
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
            if 'flags' in obj and not self.isKitti:
                flags = obj.pop('flags')
                oDict.update(flags)
            oDict.update(obj)
            oDict['indexID'] = str(uuid.uuid4())
            objList.append(oDict)
        return objList

    # start extraction for each files
    def extraction (self):
        objList = []
        for lm in self.dir:
            for root, dirs, files in os.walk(lm):
                for f in files:
                    if not f.endswith('.json'): continue
                    jList = self._extract_obj_data(os.path.join(root, f))
                    if not jList is None:
                        objList.extend(jList)
        df = pd.DataFrame(objList)
        return df
    
    # save dataframe
    def save_file (self, df, outPrefix=''):
        if outPrefix == '':
            now = datetime.now()
            outPrefix = now.strftime("%Y%m%d_%H%M%S")
        outf = os.path.join(os.getcwd(), '{}-labelMe.pickle'.format(outPrefix))
        with open(outf, 'wb') as handle:
            pickle.dump(df, handle)
        return outf

# train and test set splitting
class TrainTestSplitter (object):
    def __init__(self, dfPath):
        if not os.path.isfile(dfPath):
            raise ValueError('Unable to locate datafile {}'.format(dfPath))
        self.dfPath = dfPath
        with open(dfPath, 'rb') as handle:
            self.df = pickle.load(handle)
    
    # some unique objectList contain only 1 sample. This will cause train/test splitting return error. we need to filter out those sample and add into train set after that
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

    # checking ratio between train/test return True if met ratio setting, False otherwise
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
                print ('{} ratio not met {:.2f} [{}-{}]'.format(o, _trainRatio, maxTrain, minTrain))
                return False

            print ('{}: Train: {}({:.2f}%), Test: {}({:.2f}%)'.format(o, _trainSize, _trainRatio * 100., _testSize, _testRatio * 100.))
        return True
    
    # save train/test dataset
    def _save_output (self, traindf, testdf, outPrefix=''):
        if outPrefix == '':
            outPrefix = os.path.basename(self.dfPath).replace('-labelMe.pickle', '')
        trainf = os.path.join(os.getcwd(), '{}-trainSetInfo.pickle'.format(outPrefix))
        testf = os.path.join(os.getcwd(), '{}-testSetInfo.pickle'.format(outPrefix))
        with open(trainf, 'wb') as handle:
            pickle.dump(traindf, handle)
        with open(testf, 'wb') as handle:
            pickle.dump(testdf, handle)
        return trainf, testf

    # split train test dataset
    def train_test_split (self, ratio={'max': 70, 'min': 30, 'step': 5}, outPrefix=''):
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

        # continue looping if data splitting not met ratio requirement
        while (SPLIT):
            for train_index, test_index in ss.split(imgList, lblList):
                X_train, X_test = imgList[train_index], imgList[test_index]
                y_train, y_test = lblList[train_index], lblList[test_index]

                if ignoreList.shape[0] > 0:
                    X_train = np.concatenate((X_train, ignoreList), axis=0)

                traindf = self.df[self.df.imagePath.isin(X_train)]
                testdf = self.df[self.df.imagePath.isin(X_test)]
                res = self.data_checker(traindf, testdf, self.df.label.unique(), ratio)
                if res:
                    trainf, testf = self._save_output(traindf, testdf, outPrefix=outPrefix)
                    SPLIT = False
                    break
        return trainf, testf


            


