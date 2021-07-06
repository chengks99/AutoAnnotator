import os, math, pickle
from PIL import Image
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class FeatureExtractor(object):
    def __init__(self):
        print('Init Feature Extractor')
   
    # obtain object coordinate
    def __get_obj_coord (self, pts):
        return {'topX': pts[0][0] , 'topY': pts[0][1], 'btmX': pts[1][0] , 'btmY': pts[1][1]}

    # get maximum length from center to padding edge
    def __get_max_diff (self, imgCenter, pts):
        topDist = math.sqrt((imgCenter['x'] - pts['topX'])**2 + (imgCenter['y'] - pts['topY'])**2)
        btmDist = math.sqrt((imgCenter['x'] - pts['btmX'])**2 + (imgCenter['y'] - pts['btmY'])**2)
        return max([int(topDist), int(btmDist)])
    
    # get padding coordinard
    def __get_padding_coord (self, coord, rate):
        imgCenter = {
            'x': int((coord['topX'] + coord['btmX'])/ 2.), 
            'y': int((coord['topY'] + coord['btmY']) / 2.)
        }

        # get padding coordinate
        rCoord = {
            'topX': coord['topX'] - (abs(coord['topX'] - coord['btmX']) * rate),
            'topY': coord['topY'] - (abs(coord['topY'] - coord['btmY']) * rate),
            'btmX': coord['btmX'] + (abs(coord['topX'] - coord['btmX']) * rate),
            'btmY': coord['btmY'] + (abs(coord['topY'] - coord['btmY']) * rate)
        }

        # fix paddin to square to maintain aspect ratio
        diff = self.__get_max_diff(imgCenter, rCoord)
        rCoord = {
            'topX': imgCenter['x'] - diff,
            'topY': imgCenter['y'] - diff,
            'btmX': imgCenter['x'] + diff,
            'btmY': imgCenter['y'] + diff,
        }

        # make sure at least 10px for padding from bounding box
        minPad = 10
        calPad = []
        for k in rCoord.keys():
            calPad.append(abs(rCoord[k] - coord[k]))
        minCalPad = min(calPad)
        if minCalPad < minPad:
            dif = minPad - minCalPad
            for k in rCoord.keys():
                rCoord[k] += round(dif)
        return rCoord

    # sliding of cropped area to prevent shape overflow
    def __check_edge (self, coord, width, height):
        if coord['topX'] < 0:
            coord['btmX'] = coord['btmX'] + abs(coord['topX'])
            coord['topX'] = 0
        if coord['btmX'] > width:
            coord['topX'] = coord['topX'] - (coord['btmX'] - width)
            coord['btmX'] = width
        if coord['topY'] < 0:
            coord['btmY'] = coord['btmX'] + abs(coord['btmY'])
            coord['topY'] = 0
        if coord['btmY'] > height:
            coord['topY'] = coord['topY'] - (coord['btmY'] - height)
            coord['btmY'] = height
        return coord

    # fixing for image path. this is for development where developer moved input pickle file from to different workstation and would like to change the path name
    def _fix_for_diff_server (self, imgPath):
        baseName = os.path.basename(imgPath)
        if 'train' in imgPath:
            return os.path.join('/home/sdt-xs/annotator/train', baseName)
        else:
            return os.path.join('/home/sdt-xs/annotator/test', baseName)
    
    # get output image path
    def _get_img_path (self, row):
        imgPath = row['imagePath']
        # fix for sdt-xs server
        #imgPath = self._fix_for_diff_server(imgPath)
        imgPrefix = '{}_{}'.format(row['label'], row['group_id'])
        fname, fext = os.path.splitext(imgPath)
        cName = imgPath.replace(fext, '_{}{}'.format(imgPrefix, fext))

    # get image name and input image with original background
    def _crop_padding_obj (self, img, row, rate, save_img):
        cName = self._get_img_path(row)
        if not save_img: return cName

        objCoord = self.__get_obj_coord(row['points'])
        padCoord = self.__get_padding_coord(objCoord, rate)
        padCoord = self.__check_edge(padCoord, row['imageWidth'], row['imageHeight'])

        cropped = img.crop((padCoord['topX'], padCoord['topY'], padCoord['btmX'], padCoord['btmY']))
        cropped.save(cName)
        return cName

    # get image name and input image with white/black padding
    def _crop_greyscale_obj (self, img, row, rate, gs, save_img):
        cName = self._get_img_path(row)
        if not save_img: return cName

        # obtain object coord and its padding size
        objCoord = self.__get_obj_coord(row['points'])
        padCoord = self.__get_padding_coord(objCoord, rate)
        
        # cropped image array and size
        cropped = img.crop((objCoord['topX'], objCoord['topY'], objCoord['btmX'], objCoord['btmY']))
        crop_w, crop_h = cropped.size

        # base image array and size
        base_w = padCoord['btmX'] - padCoord['topX']
        base_h = padCoord['btmY'] - padCoord['topY']
        clr = (255, 255, 255) if gs == 'white' else (0, 0, 0)
        base_img = Image.new('RGB', (base_w, base_h), clr)

        # calculate offset to composite two image
        off_w = round((base_w - crop_w) / 2)
        off_h = round((base_h - crop_h) / 2)
        base_img.paste(cropped, (off_w, off_h))       
        base_img.save('tt22t.png')

    # output lbl conversion for range
    def _get_ranging_output (self, outDict, out):
        lbl = outDict.get('default', None)
        if lbl is None:
            raise ValueError('Default ranging output cannot be none')
        for k, v in outDict.items():
            if k == 'default': continue
            for _v in v:
                _min = min(_v)
                _max = max(_v)
                if out >= _min and out <= _max:
                    lbl = k
                    break
        return lbl

    # pad image to square to maintain aspect ratio
    #   rate: percentage of padding 
    #   Greyscale: padding area content. 
    #               'original' will pad with actual background, 
    #               'white' pad additional area with white background, 
    #               'black' pad additional area with black background 
    def img_padding (self, df, imgPathHeader, pad_rate=0.2, grey_scale='original', save_img=False):
        for ip in df[imgPathHeader].unique():
            _df = df[df[imgPathHeader] == ip]

            # fix for sdt-xs server
            imgPath = ip
            #imgPath = self._fix_for_diff_server(ip)
            
            img = Image.open(imgPath)
            for index, row in _df.iterrows():
                if grey_scale == 'original':
                    objImgPath = self._crop_padding_obj(img, row, pad_rate, save_img)
                else:
                    objImgPath = self._crop_greyscale_obj(img, row, pad_rate, grey_scale, save_img)
                df.loc[index,'objImagePath'] = objImgPath
        return df
    
    # get img array and output
    #   imgPathHeader: imgPathname
    #   outHeader: list of output header need to extract
    #   outDict: dictionary for output condition with output header as key. 
    #           outDict = {'occluded': {'0': 'no', '1': 'small', '2': 'high'}}
    #               will convert output 0 to no, 1 to small, 2 to high
    #   outf: output dataframe
    #   target_size: NN target input size
    def img2arr (self, df, imgPathHeader='objImagePath', outHeader=[], outDict={}, objLabelHead='label', target_size=(224, 224)):
        dic = {'data': [], 'obj': []}
        for lh in outHeader: dic[lh] = []
        for index, row in df.iterrows():
            img = load_img(row[imgPathHeader], target_size=target_size)
            img = img_to_array(img)
            img = preprocess_input(img)
            dic['data'].append(img)
            dic['obj'].append(row[objLabelHead])

            for lh in outHeader:
                lbl = row[lh]
                if not lh in dic: dic[lh] = []
                if lh in outDict.keys():
                    if 'ranging' in outDict[lh]:
                        lbl = self._get_ranging_output(outDict[lh]['ranging'], lbl)
                    else:
                        lbl = outDict[lh]['matching'].get(str(lbl), None)
                dic[lh].append(lbl)
        dic['data'] = np.array(dic['data'], dtype='float32')
        return dic
    
    # drop None data return dic with desired data & output
    def _get_data_dic (self, dic, outHeader):
        dic = {'data': dic['data'], 'output': dic[outHeader], 'obj': dic['obj']}
        
        # drop none
        x, y, o = [], [], []
        for _x, _y, _o in zip(list(dic['data']), dic['output'], dic['obj']):
            if not _y is None:
                x.append(_x)
                y.append(_y)
                o.append(_o)
        return {'data': np.array(x), 'output': np.array(y), 'obj': np.array(o)}
    
    # get output scaled
    def get_output_scaler (self, dic, outHeader, feature_range):
        dic = self._get_data_dic(dic, outHeader)
        if feature_range is None: return dic

        dic['output'] = dic['output'].reshape(-1,1)
        scaf = '{}-scaler.pickle'.format(outHeader)
        if not os.path.isfile(scaf):
            scaler = MinMaxScaler(feature_range=feature_range)
            scaler.fit(dic['output'])
            with open(scaf, 'wb') as handle: pickle.dump(scaler, handle)
        else:
            with open(scaf, 'rb') as handle: scaler = pickle.load(handle)
        dic['output'] = scaler.transform(dic['output'])
        return dic

    # get output encoded
    def get_output_encoder (self, dic, outHeader):
        dic = self._get_data_dic(dic, outHeader)

        encf = '{}-encoder.pickle'.format(outHeader)
        uniqueOutput = np.unique(dic['output'])
        if not os.path.isfile(encf):
            lb = LabelEncoder()
            lb.fit(dic['output'])
            with open(encf, 'wb') as handle: pickle.dump(lb, handle)
        else:
            with open(encf, 'rb') as handle: lb = pickle.load(handle)
        dic['output'] = lb.transform(dic['output'])
        dic['output'] = to_categorical(dic['output'], dtype='float32')

        for l in uniqueOutput:
            e = lb.transform(np.array([l]))
            print ('{} LabelEncoder {} transform to {}'.format(outHeader, l, e[0]))
        return dic
    
    # print output encoder
    def print_output_encoder (self, dic, outHeader):
        encf = '{}-encoder.pickle'.format(outHeader)
        if not os.path.isfile(encf):
            print ('Unable to locate output encoder file {}'.format(os.path.basename(encf)))
            return
        with open(encf, 'rb') as handle: lb = pickle.load(handle) 

        allOutput = np.array([x for x in dic[outHeader] if not x is None])
        for l in np.unique(allOutput):
            e = lb.transform(np.array([l]))
            print ('Output of {} transform to {}'.format(l, e[0]))



            



