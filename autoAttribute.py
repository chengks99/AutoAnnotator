import pickle, os, argparse, sys, copy
import numpy as np
import pandas as pd

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from featureExtractor import FeatureExtractor

# main Auto Attributor
class AutoAttributeDetector (FeatureExtractor):
    def __init__(self, **kwargs):
        self.trainf = kwargs.pop('trainf', None)
        self.testf = kwargs.pop('testf', None)
        self.evalf = kwargs.pop('evalf', None)

        self.data = {
            'train': self._get_data(self.trainf),
            'test': self._get_data(self.testf),
            'eval': self._get_data(self.evalf)
        }
    
    # set evaluation data
    def set_eval_data (self, evalf):
        self.data = {'eval': self._get_data(self.evalf)}

    # read data from pickle file
    def _get_data (self, dataf):
        if dataf is None: return None
        dataf = os.path.abspath(dataf)
        fname, fext = os.path.splitext(dataf)
        if not os.path.isfile(dataf):
            raise ValueError('Unable to locate datafile @ {}'.format(dataf))
        if fext != '.pickle':
            raise ValueError('{} is not a pickle format'.format(os.path.basename(dataf)))
        data = None
        with open(dataf, 'rb') as handle: data = pickle.load(handle)
        if data is None:
            raise ValueError('{} contains None type of data'.foramt(dataf))
        return data
    
    # preprocess the data
    #   pad_rate = padding area over image size. [default: 0.2]
    #               if obj cropped area is 100px, it will pad additional 20% of cropped
    #               area and return a 120px img
    #   grey_scale = padding area background. [default: original] 
    #                original if use actual image area as padding background
    #                black if pad as black color, white if pad as white color
    def data_preprocessing (self, imgPathHeader='imagePath', pad_rate=0.2, grey_scale='original', save_img=False):
        for key, data in self.data.items():
            if data is None: continue
            print ('Data preprocessing: Process {} data'.format(key))
            df_file = '{}df.pickle'.format(key)
            if not os.path.isfile(df_file):
                data = self.img_padding(data, imgPathHeader, pad_rate=pad_rate, grey_scale=grey_scale, save_img=save_img)
                with open(df_file, 'wb') as handle: pickle.dump(data, handle)
            else:
                with open(df_file, 'rb') as handle:
                    self.data[key] = pickle.load(handle)
    
    # convert data into dictionary type: {'train': {'data': [], 'occluded': [], 'label': [] ...}}
    def get_data_output (self, outDict={}, objLabelHead='label', indexHead='indexID', target_size=(224,224), outf={'train': 'trainData.pickle', 'test': 'testData.pickle', 'eval': 'evalData.pickle'}):
        self.objLabelHead = objLabelHead
        self.indexHead = indexHead
        self.fvData = {}
        outHeader = outDict.keys()
        for key, data in self.data.items():
            if data is None: continue
            print ('Get Data & Output for {} data'.format(key))
            if not os.path.isfile(outf[key]):
                self.fvData[key] = self.img2arr(data, outHeader=outHeader, outDict=outDict, objLabelHead=objLabelHead, indexHead=indexHead, target_size=target_size)
                with open(outf[key], 'wb') as handle: pickle.dump(self.fvData[key], handle)
            else:
                with open(outf[key], 'rb') as handle:
                    self.fvData[key] = pickle.load(handle)
   
    # filter data
    def _data_filter (self, data, filter):
        df = copy.deepcopy(data)
        for k, v in filter.items():
            df = df[df[k].isin(v)]
        return df

   # form dataframe for classification
    def _form_encoding_input_data (self, outHeader, filter):
        _data = {}
        for k in self.fvData.keys():
            _data[k] = self._data_filter(self.fvData[k], filter)
            _data[k] = self.get_output_encoder(_data[k], outHeader, self.objLabelHead, self.indexHead)
        return _data
    
    # form dataframe for regre
    def _form_regressor_input_data (self, outHeader, feature_range):
        _data = {}
        for k in self.fvData.keys():
            _data[k] = self._data_filter(self.fvData[k], filter)
            _data[k] = self.get_output_scaler(data[k], outHeader, feature_range, self.objLabelHead, self.indexHead)
        return _data
    
    # print output decoder
    def _print_decoded_output (self, key):
        for k in self.fvData.keys():
            self.print_output_encoder(self.fvData[k], key)

    # get default configuration dictionary
    def _get_cfg (self, **kwargs):
        cfg = {}
        cfg['augmentation'] = kwargs.pop('augmentation', False)
        cfg['nn'] = kwargs.pop('nn', 'mobileNet')
        cfg['epochs'] = kwargs.pop('epochs', 200)
        cfg['batch_size'] = kwargs.pop('batch_size', 32)
        cfg['input_tensor'] = kwargs.pop('input_tensor', (224,224,3))
        cfg['feature_range'] = kwargs.pop('feature_range', None)
        cfg['data_filter'] = kwargs.pop('data_filter', [])
        return cfg        

    # classification
    def attr_cls (self, params, outHeader, objLabelHead='label'):
        cfg = self._get_cfg(**params)
        print ('Perform {} attribute classification'.format(outHeader))
        print (cfg)

        clsData = self._form_encoding_input_data(outHeader, cfg['data_filter'])
        modelf='{}-{}.h5'.format(outHeader, cfg['nn'])
        classifier = None
        if cfg['nn'] == 'mobileNet':
            from modelling import MobileNetClassifier
            classifier = MobileNetClassifier(clsData, outHeader=outHeader, objLabelHead=objLabelHead, input_tensor=cfg['input_tensor'])

        #* Developer can add own classifier here

        if classifier is None:
            raise ValueError('None type Classifier')

        if 'train' in clsData.keys() and 'test' in clsData.keys():
            classifier.build_model(augmentation=cfg['augmentation'])
            classifier.train_model(modelf=modelf, epochs=cfg['epochs'], batch_size=cfg['batch_size'])
        classifier.predict_data(modelf=modelf)
        self._print_decoded_output(outHeader)
        
    # regression
    def attr_reg (self, params, outHeader):
        cfg = self._get_cfg(**params)
        print ('Perform {} Attribute Regression'.format(outHeader))
        print (cfg)

        alphaData = self._form_regressor_input_data(outHeader, feature_range=cfg['feature_range'])
        modelf='{}-{}.h5'.format(outHeader, cfg['nn'])
        regressor = None
        if cfg['nn'] == 'mobileNet':
            from modelling import MobileNetRegressor
            regressor = MobileNetRegressor(alphaData, outHeader=outHeader, input_tensor=cfg['input_tensor'])

        #* Developer can add own regressor here

        if regressor is None:
            raise ValueError('None type Regressor')

        if 'train' in alphaData.keys() and 'test' in alphaData.keys():
            regressor.build_model(augmentation=cfg['augmentation'])
            regressor.train_model(modelf=modelf, epochs=cfg['epochs'], batch_size=cfg['batch_size'])
        regressor.predict_data(modelf=modelf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automatic Attribute Classification Module'
    )

    parser.add_argument('-lm', '--labelMe', type=str, help='Specify directory that contains LabelMe format files', default=None)
    parser.add_argument('-df', '--datafile', type=str, help='Specify input datafile', default=None)
    parser.add_argument('-sp', '--split', action='store_true', help='Specify whether or not data need to split train/test', default=False)
    parser.add_argument('-tr', '--trainfile', type=str, help='Specify training detafile (pickle format)', default=None)
    parser.add_argument('-te', '--testfile', type=str, help='Specify testing datafile (pickle format)', default=None)
    parser.add_argument('-ev', '--evalfile', type=str, help='Specify evaluation datafile (pickle format). This is to auto attribute classification the dataset in actual deployment', default=None)
    args = parser.parse_args(sys.argv[1:])
    
    # extract LabelMe contains
    if not args.labelMe is None:
        from utils import LabelMeExtractor
        lm = LabelMeExtractor(args.labelMe, isKitti=False)
        df = lm.extraction()
        args.datafile = lm.save_file(df)

    # split data
    if not args.datafile is None:
        from utils import TrainTestSplitter
        tts = TrainTestSplitter(args.datafile)
        args.trainfile, args.testfile = tts.train_test_split()

    # input parameters
    outDict = {
        'type' : {
            'matching': {'sedan': 'sedan', 'van': 'van', 'bus': 'bus', 'SUV': 'suv', 'lorry': 'lorry'},
            'augmentation': True,
            'data_filter': {'label': ['vehicle']},
        },
        'occlusion': {
            'matching': {'fully visible': 'no', 'partly occluded': 'small', 'largely occluded': 'high'},
            'augmentation': False,
            'data_filter': {'label': ['vehicle']}
        },
        'view': {
            'matching': {'back': 'back', 'front': 'front', 'side-45-degree': 'side45', 'side': 'side'},
            'augmentation': False,
            'data_filter': {'label': ['cyclist', 'biker']}
        }
    }
    '''
    outDict = {
        'occluded': {
            'matching': {'0.0': 'no', '1.0': 'small', '2.0': 'high'},
            'augmentation': True,
            'data_filter': {'label': ['Car', 'Pedestrian', 'Van']}
        }, 
        'rotation_y': {
            'ranging': {'default': 'side', 'back': [[-1.67, -1.33], [0.33, 0.67]], 'front': [[1.33, 1.67], [-0.67, -0.33]]},
        },
        'label': {
            'augmentation': True,
        },
        'alpha': {
            'method': 'regression',
            'feature_range': (0,1)
        },
        'truncated': {
            'method': 'regression'
        }
    }
    '''
    
    # attribute detector sample usage
    DEVELOPMENT = True
    if DEVELOPMENT:
        attrDet = AutoAttributeDetector(trainf=args.trainfile, testf=args.testfile)
    else:
        attrDet = AutoAttributeDetector(evalf=args.evalfile)

    #attrDet.data_preprocessing(grey_scale='white', save_img=True)
    attrDet.data_preprocessing()
    attrDet.get_data_output(outDict=outDict, objLabelHead='label')

    # loopping for attribute detection
    for key, value in outDict.items():
        method = value.get('method', 'classification')
        # for debugging purpose
        if key != 'occlusion': continue

        if method == 'classification':
            attrDet.attr_cls(value, outHeader=key)
        elif method == 'regression':
            attrDet.attr_reg(value, outHeader=key)
