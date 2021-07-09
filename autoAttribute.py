import pickle, os, argparse, sys
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from featureExtractor import FeatureExtractor

class AutoAttributeDetector (FeatureExtractor):
    def __init__(self, **kwargs):
        self.trainf = kwargs.pop('trainf', None)
        self.testf = kwargs.pop('testf', None)
        self.evalf = kwargs.pop('evalf', None)
        self.filter = kwargs.pop('filter', {})

        self.data = {
            'train': self._get_data(self.trainf, filter=self.filter)
            'test': self._get_data(self.testf, filter=self.filter),
            'eval': self._get_data(self.evalf, filter=self.filter)
        }
    
    def set_eval_data (self, evalf):
        self.data = {'eval': self._get_data(self.evalf, filter=self.filter)}

    # read data from pickle file
    def _get_data (self, dataf, filter={}):
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
        for k, v in filter.items():
            data = data[data[k].isin(v)]
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
    def get_data_output (self, outDict={}, objLabelHead='label', target_size=(224,224), outf={'train': 'trainData.pickle', 'test': 'testData.pickle', 'eval': 'evalData.pickle'}):
        self.dict = {}
        outHeader = outDict.keys()
        for key, data in self.data.items():
            if data is None: continue
            print ('Get Data & Output for {} data'.format(key))
            if not os.path.isfile(outf[key]):
                self.dict[key] = self.img2arr(data, outHeader=outHeader, outDict=outDict, objLabelHead=objLabelHead, target_size=target_size)
                with open(outf[key], 'wb') as handle: pickle.dump(self.dict[key], handle)
            else:
                with open(outf[key], 'rb') as handle:
                    self.dict[key] = pickle.load(handle)
   
   # form input dict {'data': [], 'output': []}
    def _form_encoding_input_data (self, key):
        _data = {}
        for k in self.dict.keys():
            _data[k] = self.get_output_encoder(self.dict[k], key)
        return _data
    
    def _form_regressor_input_data (self, key, feature_range):
        _data = {}
        for k in self.dict.keys():
            _data[k] = self.get_output_scaler(self.dict[k], key, feature_range)
        return _data
    
    # print output decoder
    def _print_decoded_output (self, key):
        for k in self.dict.keys():
            self.print_output_encoder(self.dict[k], key)

    def _get_cfg (self, **kwargs):
        cfg = {}
        cfg['augmentation'] = kwargs.pop('augmentation', False)
        cfg['nn'] = kwargs.pop('nn', 'mobileNet')
        cfg['epochs'] = kwargs.pop('epochs', 200)
        cfg['batch_size'] = kwargs.pop('batch_size', 32)
        cfg['input_tensor'] = kwargs.pop('input_tensor', (224,224,3))
        cfg['feature_range'] = kwargs.pop('feature_range', None)
        return cfg

    # occlusion classification
    def attr_cls (self, params, yLabel):
        cfg = self._get_cfg(**params)
        print ('Perform {} attribute classification'.format(yLabel))
        print (cfg)

        occData = self._form_encoding_input_data(yLabel)
        modelf='{}-{}.h5'.format(yLabel, cfg['nn'])
        if cfg['nn'] == 'mobileNet':
            from modelling import MobileNetClassifier
            mn = MobileNetClassifier(occData, yLabel=yLabel, input_tensor=cfg['input_tensor'])
            if 'train' in occData.keys() and 'test' in occData.keys():
                mn.build_model(augmentation=cfg['augmentation'])
                mn.train_model(modelf=modelf, epochs=cfg['epochs'], batch_size=cfg['batch_size'])
            mn.predict_data(modelf=modelf)
            self._print_decoded_output(yLabel)
        
    # alpha classification
    def attr_reg (self, params, yLabel):
        cfg = self._get_cfg(**params)
        print ('Perform {} Attribute Regression'.format(yLabel))
        print (cfg)

        alphaData = self._form_regressor_input_data(yLabel, feature_range=cfg['feature_range'])
        modelf='{}-{}.h5'.format(yLabel, cfg['nn'])
        if cfg['nn'] == 'mobileNet':
            from modelling import MobileNetRegressor
            mn = MobileNetRegressor(alphaData, yLabel=yLabel, input_tensor=cfg['input_tensor'])
            if 'train' in alphaData.keys() and 'test' in alphaData.keys():
                mn.build_model(augmentation=cfg['augmentation'])
                mn.train_model(modelf=modelf, epochs=cfg['epochs'], batch_size=cfg['batch_size'])
            mn.predict_data(modelf=modelf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automatic Attribute Classification Module'
    )

    parser.add_argument('-train', '--trainfile', type=str, help='Specify training detafile (pickle format)', default=None)
    parser.add_argument('-test', '--testfile', type=str, help='Specify testing datafile (pickle format)', default=None)
    parser.add_argument('-eval', '--evalfile', type=str, help='Specify evaluation datafile (pickle format). This is to auto attribute classification the dataset in actual deployment', default=None)
    args = parser.parse_args(sys.argv[1:])

    # input parameters
    dataFilter = {'label': ['Car', 'Pedestrian', 'Van']}
    outDict = {
        'occluded': {
            'matching': {'0.0': 'no', '1.0': 'small', '2.0': 'high'},
            'augmentation': True,
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
    
    # attribute detector sample usage
    DEVELOPMENT = True
    if DEVELOPMENT:
        attrDet = AutoAttributeDetector(trainf=args.trainfile, testf=args.testfile, filter=dataFilter)
    else:
        attrDet = AutoAttributeDetector(evalf=args.evalfile, filter=dataFilter)

    #attrDet.data_preprocessing(grey_scale='white', save_img=True)
    attrDet.data_preprocessing()
    attrDet.get_data_output(outDict=outDict, objLabelHead='label')

    for key, value in outDict.items():
        method = value.get('method', 'classification')
        # for debugging purpose
        if key != 'truncated': continue

        if method == 'classification':
            attrDet.attr_cls(value, yLabel=key)
        elif method == 'regression':
            attrDet.attr_reg(value, yLabel=key)
