import os, json, copy, pickle, uuid
import pandas as pd

class LabelMeExtractor (object):
    def __init__(self, lmDir):
        self.dir = lmDir
        if not os.path.isdir(self.dir):
            raise ValueError('Unable to locate labelme directory: {}'.format(self.dir))
        self.outf = os.path.join(self.dir, '{}.pickle'.format(os.path.basename(self.dir)))
    
    def _extract_obj_data (self, jsonf):
        with open(jsonf, 'r') as f: data = json.load(f)
        objShape = data.pop('shapes', None)
        if objShape is None: return None
        objList = []
        for obj in objShape:
            oDict = copy.deepcopy(data)
            if 'flags' in obj:
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
        self.ratio = ratio
    
    def train_test_split (self, ratio={'max': 70, 'min': 30, 'step': 10}):
        pass
