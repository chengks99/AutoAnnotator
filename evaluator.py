import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from PIL import Image

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

class Evaluation(object):
    def __init__(self, modelf, outHeader, baseDir):
        print ('Inherit Evaluation Module')
        self.outPrefix = modelf.replace('.h5', '')
        self.outHeader = outHeader
        self.baseDir = baseDir
    
    # TPR and TNR rate calculation
    def _tpr_tnr (self, matrix):
        try:
            _tnr = float(matrix[0][0]) / (float(matrix[0][0]) + float(matrix[1][0]))
            _tpr = float(matrix[1][1]) / (float(matrix[0][1]) + float(matrix[1][1])) 
        except:
            return None, None
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
        outf = os.path.join(self.baseDir, outf)
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
        print (len(objType), len(actLabel), len(predLabel))
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
        import pickle
        jsonDic = {}
        jsonDic['data'] = data
        jsonDic['act'] = act
        jsonDic['pred'] = pred
        jsonDic['modelf'] = modelf
        jsonDic['eClass'] = eClass
        jsonDic['labelling'] = labelling

        with open('eval.pickle', 'wb') as handle:
            pickle.dump(jsonDic, handle)
        
        self.labelling = labelling
        self._basic_cf(act, pred)
        self._out_by_class(data, act, pred, eClass)

        self.xrai(data, modelf, labelling)

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
            
    # preparation for XRAI saliency
    def xrai (self, data, modelf, labelling):
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Input

        # classification model
        cModel = load_model(modelf)

        # model for saliency (currently only working with MobileNet)
        input_tensor = Input(shape=(224,224,3))
        m = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3), input_tensor=input_tensor)
        conv_layer = m.get_layer('block_16_project')
        model = tf.keras.models.Model([m.inputs], [conv_layer.output, m.output])

        outDir = os.path.join(self.baseDir, 'saliency-{}'.format(self.outHeader))
        if not os.path.isdir(outDir):
            os.makedirs(outDir)

        dfLen = data.shape[0]

        for index, row in data.iterrows():
            _input = np.array([row['data']])
            prediction = cModel.predict(_input)

            _actIndex = np.argmax(row[self.outHeader])
            _predIndex = np.argmax(prediction)
            
            _act = labelling[_actIndex]
            _pred = labelling[_predIndex]
            _actProb = prediction[0][_actIndex]
            _predProb = prediction[0][_predIndex]

            _res = 'W'
            if _act == _pred:
                _res = 'C'
            imgf = '{}_{}-{:.2f}-{}-{:.2f}_{}'.format(_res, _act, _actProb, _pred, _predProb, os.path.basename(row['objImagePath']))

            print ('********** {}/{} **********'.format(index, dfLen))
            print ('IndexID: {}'.format(row['indexID']))
            print ('ImgPath: {}'.format(row['objImagePath']))
            print ('Labelling: {} ({})'.format(row[self.outHeader], _act))
            print ('Outf: {}'.format(imgf))

            outf = os.path.join(outDir, '{}'.format(imgf))
            self.xrai_img(row['data'], model, _predIndex, outf)

    def xrai_img (self, imgData, model, pred, outf):
        import saliency.core as saliency
        from matplotlib import pylab as P

        class_idx_str = 'class_idx_str'
        def call_model_function(images, call_model_args=None, expected_keys=None):
            target_class_idx =  call_model_args[class_idx_str]
            images = tf.convert_to_tensor(images)
            with tf.GradientTape() as tape:
                if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
                    tape.watch(images)
                    _, output_layer = model(images)
                    output_layer = output_layer[:,target_class_idx]
                    gradients = np.array(tape.gradient(output_layer, images))
                    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
                else:
                    conv_layer, output_layer = model(images)
                    gradients = np.array(tape.gradient(output_layer, conv_layer))
                    return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                            saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}
        
        def ShowImage(im, title='', ax=None):
            if ax is None:
                P.figure()
            P.axis('off')
            P.imshow(im)
            P.title(title)

        def ShowHeatMap(im, title, ax=None):
            if ax is None:
                P.figure()
            P.axis('off')
            P.imshow(im, cmap='inferno')
            P.title(title)

        call_model_args = {class_idx_str: pred}
        xrai_obj = saliency.XRAI()
        xrai_attr = xrai_obj.GetMask(imgData, call_model_function, call_model_args, batch_size=20)
        
        ROWS = 1
        COLS = 3
        UPSCALE_FACTOR = 5
        #P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
        P.figure()

        # Show original image
        inImg = Image.fromarray((imgData * 255).astype(np.uint8))
        ShowImage(inImg, title='Original_Image', ax=P.subplot(ROWS, COLS, 1))

        # Show XRAI heatmap attributions
        ShowHeatMap(xrai_attr, title='XRAI_Heatmap', ax=P.subplot(ROWS, COLS, 2))

        # Show most salient 30% of the image
        mask = xrai_attr > np.percentile(xrai_attr, 70)
        im_mask = np.array(imgData)
        im_mask[~mask] = 0
        ShowImage(im_mask, title='Top_30%', ax=P.subplot(ROWS, COLS, 3))
        P.savefig(outf)