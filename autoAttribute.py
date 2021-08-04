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
        self.isKitti = kwargs.pop('isKitti', False)
        self.baseDir = kwargs.pop('baseDir', None)
        if self.baseDir is None:
            self.baseDir = os.getcwd()

        self.data = {
            'train': self._get_data(self.trainf),
            'test': self._get_data(self.testf),
            'eval': self._get_data(self.evalf)
        }

        if self.isKitti and 'train' in self.data:
            self.data['train'] = self.data['train'].head(1000)
        
        FeatureExtractor.__init__(self)
          
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
    def data_preprocessing (self, imgPathHeader='imagePath', pad_rate=0.2, grey_scale='original', outPrefix='ds', save_img=False):
        for key, data in self.data.items():
            if data is None: continue
            print ('Data preprocessing: Process {} data'.format(key))
            df_file = '{}-{}df.pickle'.format(outPrefix, key)
            df_file = os.path.join(self.baseDir, df_file)
            if not os.path.isfile(df_file):
                data = self.img_padding(data, imgPathHeader, pad_rate=pad_rate, grey_scale=grey_scale, tag=key, save_img=save_img)
                with open(df_file, 'wb') as handle: pickle.dump(data, handle)
            else:
                with open(df_file, 'rb') as handle:
                    self.data[key] = pickle.load(handle)
    
    # get list of desire attribute
    def _get_outHeader (self, inputList):
        outHeader = []
        for l in inputList:
            attr = l.get('attribute', None)
            if attr is None: continue
            if not attr in outHeader:
                outHeader.append(attr)
        return outHeader

    # convert data into dictionary type: {'train': {'data': [], 'occluded': [], 'label': [] ...}}
    def get_data_output (self, inputParams={}, objLabelHead='label', indexHead='indexID', target_size=(224,224), outPrefix=''):
        self.objLabelHead = objLabelHead
        self.indexHead = indexHead
        self.fvData = {}
        outHeader = self._get_outHeader(inputParams.get('config', []))
        if outPrefix == '':
            outPrefix = '{}'.format('-'.join(oh for oh in sorted(outHeader)))
        for key, data in self.data.items():
            if data is None: continue
            print ('Get Data & Output for {} data'.format(key))
            outf = '{}_{}Data.pickle'.format(outPrefix, key)
            outf = os.path.join(self.baseDir, outf)
            if not os.path.isfile(outf):
                self.fvData[key] = self.img2arr(data, outHeader=outHeader, inputParams=inputParams, objLabelHead=objLabelHead, indexHead=indexHead, target_size=target_size)
                with open(outf, 'wb') as handle: pickle.dump(self.fvData[key], handle)
            else:
                with open(outf, 'rb') as handle:
                    self.fvData[key] = pickle.load(handle)
   
    # filter data
    def _data_filter (self, data, filter):
        df = copy.deepcopy(data)
        for k, v in filter.items():
            df = df[df[k].isin(v)]
        return df

   # form dataframe for classification
    def _form_encoding_input_data (self, outHeader, filter, second_class, prefix):
        _data = {}
        for k in self.fvData.keys():
            _data[k] = self._data_filter(self.fvData[k], filter)
            _data[k] = self.get_output_encoder(_data[k], outHeader, second_class, prefix, self.objLabelHead, self.indexHead)
        return _data
    
    # form dataframe for regre
    def _form_regressor_input_data (self, outHeader, feature_range, second_class, prefix):
        _data = {}
        for k in self.fvData.keys():
            _data[k] = self._data_filter(self.fvData[k], filter)
            _data[k] = self.get_output_scaler(data[k], outHeader, feature_range, second_class, prefix, self.objLabelHead, self.indexHead)
        return _data
    
    # print output decoder
    def _print_decoded_output (self, key, prefix):
        self.labelling = None
        for k in self.fvData.keys():
            labelling = self.print_output_encoder(self.fvData[k], key, prefix)
            if self.labelling is None:
                self.labelling = labelling

    # get default configuration dictionary
    def _get_cfg (self, outHeader, **kwargs):
        cfg = {}
        cfg['augmentation'] = kwargs.pop('augmentation', False)
        cfg['nn'] = kwargs.pop('nn', 'mobileNet')
        cfg['epochs'] = kwargs.pop('epochs', 1 if self.isKitti else 200)
        cfg['batch_size'] = kwargs.pop('batch_size', 32)
        cfg['input_tensor'] = kwargs.pop('input_tensor', (224,224,3))
        cfg['feature_range'] = kwargs.pop('feature_range', None)
        cfg['data_filter'] = kwargs.pop('data_filter', {})
        cfg['prefix'] = kwargs.pop('prefix', outHeader)
        cfg['second_class'] = kwargs.pop('second_class', None)
        return cfg        

    # classification
    def attr_cls (self, params, outHeader, objLabelHead='label'):
        cfg = self._get_cfg(outHeader, **params)
        print (cfg)

        clsData = self._form_encoding_input_data(outHeader, cfg['data_filter'], cfg['second_class'], cfg['prefix'])
        modelf='{}-{}.h5'.format(cfg['prefix'], cfg['nn'])
        modelf = os.path.join(self.baseDir, modelf)
        classifier = None
        if cfg['nn'] == 'mobileNet':
            from modelling import MobileNetClassifier
            classifier = MobileNetClassifier(clsData, outHeader=outHeader, objLabelHead=objLabelHead, input_tensor=cfg['input_tensor'], baseDir=self.baseDir)

        #* Developer can add own classifier here

        if classifier is None:
            raise ValueError('None type Classifier')

        if 'train' in clsData.keys() and 'test' in clsData.keys():
            classifier.build_model(augmentation=cfg['augmentation'])
            classifier.train_model(modelf=modelf, epochs=cfg['epochs'], batch_size=cfg['batch_size'])
        self._print_decoded_output(outHeader, cfg['prefix'])
        pred = classifier.predict_data(modelf=modelf, eClass=cfg['second_class'], labelling=self.labelling)        
        
    # regression
    def attr_reg (self, params, outHeader):
        cfg = self._get_cfg(**params)
        print (cfg)

        alphaData = self._form_regressor_input_data(outHeader, feature_range=cfg['feature_range'], second_class=cfg['second_class'], prefix=cfg['prefix'])
        modelf='{}-{}.h5'.format(cfg['prefix'], cfg['nn'])
        modelf = os.path.join(self.baseDir, modelf)
        regressor = None
        if cfg['nn'] == 'mobileNet':
            from modelling import MobileNetRegressor
            regressor = MobileNetRegressor(alphaData, outHeader=outHeader, input_tensor=cfg['input_tensor'], BaseDir=self.baseDir)

        #* Developer can add own regressor here

        if regressor is None:
            raise ValueError('None type Regressor')

        if 'train' in alphaData.keys() and 'test' in alphaData.keys():
            regressor.build_model(augmentation=cfg['augmentation'])
            regressor.train_model(modelf=modelf, epochs=cfg['epochs'], batch_size=cfg['batch_size'])
        pred = regressor.predict_data(modelf=modelf, eClass=cfg['second_class'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automatic Attribute Classification Module'
    )

    parser.add_argument('-lm', '--labelMe', nargs='+', help='Specify directory that contains LabelMe format files', default=None)
    parser.add_argument('-df', '--datafile', type=str, help='Specify input datafile', default=None)
    parser.add_argument('-sp', '--split', action='store_true', help='Specify whether or not data need to split train/test', default=False)
    parser.add_argument('-kt', '--kitti', action='store_true', help='Specify whether or not dataset is kitti dataset', default=False)
    parser.add_argument('-tr', '--trainfile', type=str, help='Specify training detafile (pickle format)', default=None)
    parser.add_argument('-te', '--testfile', type=str, help='Specify testing datafile (pickle format)', default=None)
    parser.add_argument('-ev', '--evalfile', type=str, help='Specify evaluation datafile (pickle format). This is to auto attribute classification the dataset in actual deployment', default=None)
    args = parser.parse_args(sys.argv[1:])
  
    # defined naming of label. 
    namingParams = {
        'baseDir': 'ld3-8',
        'labelMe': 'DS3-8',  # use timestamp as prefix if empty string
        'trainTest': '', # same as labelMe if '' else using this for train/test data output prefix
        'outPrefix': 'type-alpha-truncated'   # combination of unique sorted attribute list if '' else use this for attribute retrieval file
    }

    # input parameters
    if args.kitti:
        inputParams = {
            'convert': {
                'occluded': {'matching': {'0.0': 'no', '1.0': 'small', '2.0': 'high'}},
                'rotation_y': {'ranging': {'default': 'side', 'back': [[-1.67, -1.33], [0.33, 0.67]], 'front': [[1.33, 1.67], [-0.67, -0.33]]}}
            },
            'config': [
                {
                    'attribute': 'occluded',
                    'augmentation': False,
                    'data_filter': {'label': ['Car', 'Pedestrian', 'Van']}
                },
                {
                    'attribute': 'alpha',
                    'method': 'regression',
                    'feature_range': (0,1),
                    'ignore': True
                },
                {
                    'attribute': 'truncated',
                    'method': 'regression',
                    'ignore': True
                }
            ]
        }
    else:
        inputParams = {
            'convert': {
                'type': {
                    'matching': {'sedan': 'sedan', 'van': 'van', 'bus': 'bus', 'SUV': 'suv', 'lorry': 'lorry'},
                },
                'occlusion': {
                    'matching': {'fully visible': 'no', 'partly occluded': 'small', 'largely occluded': 'high', 'leg occluded': 'leg', 'both wheel occluded': 'wheel2', 'one wheel occluded': 'wheel1', 'body occluded': 'body', 'head occluded': 'head', 'cyclist occluded': 'cyclistOcc', 'bike occluded': 'bikeOcc'},
                },
                'view': {
                    'matching': {'back': 'back', 'front': 'front', 'side-45-degree': 'side45', 'side': 'side'},
                }
            },
            'config': [
                {
                    'attribute': 'type',
                    'augmentation': True,
                    'data_filter': {'label': ['vehicle']},
                    'prefix': 'type_vehicle',
                    'ignore': True
                },
                {
                    'attribute': 'occlusion',
                    'augmentation': False,
                    'data_filter': {'label': ['vehicle']},
                    'prefix': 'occlusion_vehicle',
                    'second_class': 'type',
                    'batch_size': 8,
                    'ignore': False
                },
                {
                    'attribute': 'view',
                    'augmentation': False,
                    'data_filter': {'label': ['cyclist', 'biker']},
                    'prefix': 'view_cyclist_biker',
                    'ignore': True
                },
                {
                    'attribute': 'occlusion',
                    'augmentation': False,
                    'data_filter': {'label': ['pedestrian']},
                    'prefix': 'occlusion_pedestrian',
                    'ignore': True
                },
                {
                    'attribute': 'occlusion',
                    'augmentation': False,
                    'data_filter': {'label': ['vehicle', 'pedestrian']},
                    'prefix': 'occlusion_vehicle_pedestrian',
                    'second_class': 'type',
                    'ignore': True
                }
            ]
        }
    
    if not namingParams.get('baseDir', None) is None:
        namingParams['baseDir'] = os.path.join(os.getcwd(), namingParams['baseDir'])
        if not os.path.isdir(namingParams['baseDir']):
            os.makedirs(namingParams['baseDir'])
    
    # extract LabelMe contains
    if not args.labelMe is None:
        from utils import LabelMeExtractor
        lm = LabelMeExtractor(args.labelMe, isKitti=args.kitti)
        args.datafile = lm.extraction(outPrefix=namingParams.get('labelMe', ''), baseDir=namingParams.get('baseDir', None))

    # split data
    if not args.datafile is None:
        from utils import TrainTestSplitter
        tts = TrainTestSplitter(args.datafile)
        args.trainfile, args.testfile = tts.train_test_split(outPrefix=namingParams.get('trainTest', ''), baseDir=namingParams.get('baseDir', None))

    # attribute detector sample usage
    DEVELOPMENT = True
    if DEVELOPMENT:
        attrDet = AutoAttributeDetector(trainf=args.trainfile, testf=args.testfile, isKitti=args.kitti, baseDir=namingParams.get('baseDir', None))
    else:
        attrDet = AutoAttributeDetector(evalf=args.evalfile, baseDir=namingParams.get('baseDir', None))

    # data preprocessing (image padding) and feature/attribute extraction
    dfPrefix = ''
    dic = {'train': args.trainfile, 'test': args.testfile, 'eval': args.evalfile}
    for k, v in dic.items():
        if not v is None:
            dfPrefix = os.path.basename(v).replace('-{}SetInfo.pickle'.format(k), '')
            break
    if dfPrefix == '':
        raise ValueError('Make sure input dataSetInfo is valid')
    attrDet.data_preprocessing(outPrefix=dfPrefix)
    attrDet.get_data_output(inputParams=inputParams, objLabelHead='label', outPrefix=namingParams.get('outPrefix', ''))

    # loopping for attribute detection
    for value in inputParams.get('config', []):
        method = value.get('method', 'classification')
        attr = value.get('attribute', None)
       
        if attr is None:
            print ('Attribute key cannot be None')
            continue

        if value.get('ignore', False):
            print ('Ignore {} {} as per configuration'.format(attr, value.get('prefix', '')))
            continue
        
        print ('Perform {} attribute {} for {}'.format(attr, method, value.get('prefix', 'Unknown Prefix')))

        if method == 'classification':
            attrDet.attr_cls(value, outHeader=attr)
        elif method == 'regression':
            attrDet.attr_reg(value, outHeader=attr)
