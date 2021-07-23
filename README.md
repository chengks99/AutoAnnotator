# Auto Attribute Detector
This project working to provide auto attribute detection for detected object in image processing. Objects in image can have multiple attribute such as viewing angle, truncation and different label. Those attribute can help developer to find out which attribute actyally lead to false detection when perform image processing in ML

### SourceTree
|   **files/directories**    |   **Descriptions**   |
|----------------------------|---------------------|
|**autoAttribute.py**    |main code to handle data input and call repective library|
|**evaluator.py** | evaluate auto attribute model output |
|**featureExtractor.py**| module to extract feature from image and label encoding|
|**modelling.py**| module to perform classification or regression|
|**utils.py** | utilities file to perform labelme->dataframe conversion and train/test splitting  |
|**README.md**|this document|

### Command
```sh
$ python autoAttribute.py [-h] [-lm LABELME] [-df DATAFILE] [-sp] [-tr TRAINFILE] [-te TESTFILE] [-ev EVALFILE]
```

Automatic Attribute Classification Module

optional arguments:
  -h, --help            show this help message and exit
  -lm LABELME, --labelMe LABELME
                        Specify directory that contains LabelMe format files
  -df DATAFILE, --datafile DATAFILE
                        Specify input datafile
  -sp, --split          Specify whether or not data need to split train/test
  -tr TRAINFILE, --trainfile TRAINFILE
                        Specify training detafile (pickle format)
  -te TESTFILE, --testfile TESTFILE
                        Specify testing datafile (pickle format)
  -ev EVALFILE, --evalfile EVALFILE
                        Specify evaluation datafile (pickle format). This is to auto attribute classification the dataset in actual deployment

### inputParams

Dictionary to store input parameter. This input dictionary contains two keys:
| **key** | **Descriptions**  |
|---------|-------------------|
| **convert** | Use to convert original data into category based for classification. This field is in dictionary base with attribute name as key follow by configuration|
| **config**  | list of attribute configuration |

Explanation for inputParams['convert']:

| **convert** | **Descriptions**  |
|-------------|-------------------|
|**matching**|_CLASSIFICATION_ paramter. Convert integer/float value into string category.</br> For example **{'0.0': 'no', '1.0': 'small', '2.0': 'high'}** will convert data with _key_ into _value_.</br> This is useful for script to form one hot encoding labelling. **_matching_** KEY CANNOT WORK WITH **_ranging_**. USER SHOULD CHOOSE EITHER ONE BASED ON DATA INPUT FORMAT.|
|**ranging**|_CLASSIFICATION_ parameter. Convert a range of integer/float value into string category. User should also defined _default_ in this setting.</br> For example **{'default': 'side', 'back': [[-1.67, -1.33], [0.33, 0.67]], 'front': [[1.33, 1.67], [-0.67, -0.33]]}** will firstly define all data as _side_. System will consider data as _back_ if it value between -1.67 to -1.33 and 0.33 to 0.67.</br>  This is useful for script to form one hot encoding labelling. **_ranging_** KEY CANNOT WORK WITH **_matching_**. USER SHOULD CHOOSE EITHER ONE BASED ON DATA INPUT FORMAT.|

Explanation of parameter setting for inputParams['config']:

| **config**  | **Descriptions**  |
|-------------|-------------------|
|**attribute**| attribute name. This field is neccessary. |
|**method**|_classification_ or _regression_. Default is classification|
|**augmentation**|True if using _ImageDataGenerator_ augmentation method to expand training data, False otherwise. Default: False|
|**data_filter**|Dictionary of data to be select. Format should be {'column_header': [list_of_value]}. For example {'label': ['Car', 'Van']} will select only data with label _car_ and _van_. Default: {}|
|**prefix**|Output file prefix. Default: key of dictionary|
|**ignore**|Ignore this key setting is True. System will not process setting for this key. Default: False|
|**nn**|Define neural network model to use (ONLY SUPPORT MOBILENET). Default: mobileNet|
|**epochs**|Maximum epochs in model training. Default: 200|
|**batch_size**|Batch size use in ML. Default: 32|
|**input_tensor**|Data input shape. Default: (224, 224, 3)|
|**feature_range**|_REGRESSION_ parameter. None if not using scalling. A tuptle contain min & max scale and script will scale data input accordingly. Default: None|
