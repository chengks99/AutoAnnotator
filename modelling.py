import pickle, sys, os, math
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from evaluator import Evaluation

# ML model base
class ModellingBase (object):
    def __init__(self):
        print ('Inherit Modelling Base Module')
    
    # perform data augmentation generator
    def get_augmentation (self):
        return ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    # class weight for inbalance class 
    def get_class_weight (self, output):
        y_int = np.argmax(output, axis=1)
        class_weights = class_weight.compute_class_weight('balanced', np.array(y_int), y_int)
        d_class_weights = dict(enumerate(class_weights))
        return d_class_weights
    
    # convert dataframe into numpy array (input, label)
    def df_to_input (self, data):
        X, y = [], []
        for index, row in data.iterrows():
            X.append(row['data'])
            y.append(row[self.outHeader])
        return np.array(X), np.array(y)

    # shuffle input data
    def shuffle_input (self, data, val_split=0.2):
        X, y = self.df_to_input(data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, stratify=y)
        trainData = np.concatenate((X_train, X_val))
        trainOutput = np.concatenate((y_train, y_val))

        print ('InternalSpliting: TrainDataShape: {} TrainOutputShape: {}'.format(trainData.shape, trainOutput.shape))
        return trainData, trainOutput

    # optimizer
    def get_optimizer (self):
        return Adam(self.init_lrate)

    # callback list
    def get_callbacks (self, modelf):
        weight_path = modelf.replace('.h5', '_weights.best.hdf5')
        cp = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        el = EarlyStopping(monitor='val_loss', mode='min', patience=10)

        def step_decay(epoch):
            initial_lrate = self.init_lrate
            drop = 0.5
            epochs_drop = 10.
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
        lrate = LearningRateScheduler(step_decay)
        return [cp, el, lrate]
    
    '''
    # classification confusion matrix printout 
    def cls_res_output (self, act, pred):            
        conf_matrix = confusion_matrix(act.argmax(axis=1), pred.argmax(axis=1))
        print(conf_matrix)
        outList = np.array(range(len(act[0])))
        multiLabel_matrix = multilabel_confusion_matrix(act.argmax(axis=1), pred.argmax(axis=1), labels=outList)
        print(multiLabel_matrix)

    # classificaion result print out by class
    def cls_res_output_by_class (self, data, act, pred):
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
    
    # regression result output
    def reg_res_output (self, act, pred):
        resList = []
        for a, p in zip(list(act), list(pred)):
            print ('{:.2f} -> {:.2f}, diff: {:.2f}'.format(a, p[0], a-p[0]))
            resList.append(a - p[0])
        resList = np.array(resList)

        mean = np.mean(resList)
        std = np.std(resList)

        print ('Regression Result: Mean: {:.2f}%, Std: {:.2f}%'.format(mean, std))
        '''

# MobileNet Classifier
class MobileNetClassifier (ModellingBase, Evaluation):
    def __init__(self, data, outHeader, objLabelHead, input_tensor):
        self.data = data
        self.outHeader = outHeader
        self.objLabelHead = objLabelHead
        self.input_tensor = input_tensor
        self.init_lrate = 1e-4
        self.model = None

        ModellingBase.__init__(self)
        
    # classification layer
    def _get_output_layer (self, baseOut):
        clsModel = Flatten()(baseOut)
        clsModel = Dense(256, activation='relu')(clsModel)
        clsModel = Dropout(.25)(clsModel)
        return Dense(self.train[self.outHeader].to_numpy()[0].shape[0], activation='softmax')(clsModel)

    # build model
    def build_model (self, augmentation=False):
        self.train = self.data['train']
        self.test = self.data['test']

        from tensorflow.keras.applications import MobileNetV2
        self.aug = self.get_augmentation() if augmentation else None

        input_tensor = Input(shape=self.input_tensor)
        baseModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_tensor, input_tensor=input_tensor)
        for layer in baseModel.layers: layer.trainable = False
        output_tensor = self._get_output_layer(baseModel.output)

        self.model = Model(inputs=input_tensor, outputs=output_tensor)

        self.model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['categorical_accuracy'])
    
    # start training/loading model
    def train_model (self, modelf='cls-mobileNet.h5', epochs=200, batch_size=32):
        if not os.path.isfile(modelf):
            trainData, trainOutput = self.shuffle_input(self.train)
            testData, testOutput = self.df_to_input(self.test)
            if self.aug is None:
                history = self.model.fit(
                    trainData, trainOutput, batch_size=batch_size,
                    steps_per_epoch=len(trainData) // batch_size,
                    validation_data=(testData, testOutput),
                    validation_steps=len(testData) // batch_size,
                    epochs=epochs,
                    callbacks = self.get_callbacks(modelf),
                    class_weight=self.get_class_weight(trainOutput)
                )
            else:
                history = self.model.fit(
                    self.aug.flow(trainData, trainOutput, batch_size),
                    steps_per_epoch=trainData.shape[0] // batch_size,
                    validation_data=(testData, testOutput),
                    epochs=epochs,
                    callbacks=self.get_callbacks(modelf),
                    class_weight=self.get_class_weight(trainOutput)
                )
            with open('{}'.format(modelf.replace('.h5', '-history.pickle')), 'wb') as handle:
                pickle.dump(history.history, handle)
            self.model.save(modelf)
        else:
            self.model = load_model(modelf)

    # perform prediction
    def predict_data (self, modelf='cls-mobileNet.h5', data=None, eClass=None, labelling=[]):
        data = self.data['test'] if data is None else data
        testData, testOutput = self.df_to_input(data)
        if data is None:
            raise ValueError('Evaluation data is empty')
        if self.model is None and os.path.isfile(modelf):
            self.model = load_model(modelf)
        
        score, acc = self.model.evaluate(testData, testOutput, batch_size=32)
        print ('TestScore: {:.2f}, Accuracy: {:.2f}'.format(score, acc))
        pred = self.model.predict(testData)

        Evaluation.__init__(self, modelf)
        self.classification_result(data, testOutput, pred, eClass, modelf, labelling)

        #self.cls_res_output(testOutput, pred)
        #self.cls_res_output_by_class(data, testOutput, pred)
    
# MobileNet regressor
class MobileNetRegressor (ModellingBase, Evaluation):
    def __init__(self, data, outHeader, input_tensor):
        self.data = data
        self.outHeader = outHeader
        self.input_tensor = input_tensor
        self.init_lrate = 1e-4
        self.model = None

        ModellingBase.__init__(self)
    
    # regressor layer
    def _get_output_layer (self, baseOut):
        regModel = Flatten()(baseOut)
        regModel = Dense(128, activation='relu')(regModel)
        regModel = BatchNormalization()(regModel)
        regModel = Dropout(.5)(regModel)
        regModel = Dense(32, activation='relu')(regModel)
        return Dense(1, activation='linear')(regModel)

    # build model
    def build_model (self, augmentation=False):
        self.train = self.data['train']
        self.test = self.data['test']

        from tensorflow.keras.applications import MobileNetV2
        self.aug = self.get_augmentation() if augmentation else None

        input_tensor = Input(shape=self.input_tensor)
        baseModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_tensor, input_tensor=input_tensor)
        for layer in baseModel.layers: layer.trainable = False
        output_tensor = self._get_output_layer(baseModel.output)

        self.model = Model(inputs=input_tensor, outputs=output_tensor)

        self.model.compile(loss='mse', optimizer=self.get_optimizer(), metrics=['mean_absolute_error'])

    # start training/loading model
    def train_model (self, modelf='regressor-mobileNet.h5', epochs=200, batch_size=32):
        trainData, trainOutput = self.df_to_input(self.train)
        testData, testOutput = self.df_to_input(self.test)
        if not os.path.isfile(modelf):
            if self.aug is None:
                history = self.model.fit(
                    trainData, testData, batch_size=batch_size,
                    steps_per_epoch=len(trainData) // batch_size,
                    validation_data=(testData, testOutput),
                    validation_steps=len(testData) // batch_size,
                    epochs=epochs,
                    callbacks = self.get_callbacks(modelf)
                )
            else:
                history = self.model.fit(
                    self.aug.flow(trainData, trainOutput, batch_size),
                    steps_per_epoch=trainData.shape[0] // batch_size,
                    validation_data=(testData, testOutput),
                    epochs=epochs,
                    callbacks=self.get_callbacks(modelf)
                )
            with open('{}'.format(modelf.replace('.h5', '-history.pickle')), 'wb') as handle:
                pickle.dump(history.history, handle)
            self.model.save(modelf)
        else:
            self.model = load_model(modelf)
    
    # prediction
    def predict_data (self, modelf, data=None, eClass=None):
        data = self.data['test'] if data is None else data
        testData, testOutput = self.df_to_input(data)
        if data is None:
            raise ValueError('Evaluation data is empty')
        if self.model is None and os.path.isfile(modelf):
            self.model = load_model(modelf)
        
        pred = self.model.predict(testData)
        Evaluation.__init__(self, modelf)
        self.regression_result(testOutput, pred, eClass, modelf)
