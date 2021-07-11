# Auto Attribute Detector
This project working to provide auto attribute detection for detected object in image processing. Objects in image can have multiple attribute such as viewing angle, truncation and different label. Those attribute can help developer to find out which attribute actyally lead to false detection when perform image processing in ML

### SourceTree
|   **files/directories**    |   **Descriptions**   |
|----------------------------|---------------------|
|**autoAttribute.py**    |main code to handle data input and call repective library|
|**featureExtractor.py**| module to extract feature from image and label encoding|
|**modelling**| module to perform classification or regression|
|**README.md**|this document|

### Command
```sh
$ python3 autoAttribute.py  [-h] [-train trainfile] [-test testfile] [-eval eavlfile]
```

- optional arguments</br>
  -h,       --help        show help message and exit</br>
  -train trainfile, --trainfile training pickle file path name. OPTIONAL</br>
  -test testfile, --testfile    testing pickle file path name. OPTIONAL</br>
  -eval evalfile, --evalfile    eval pickle file path name. OPTIONAL</br>

## dataFilter 

Dictionary to filter initial data with certain parameters. </br>For example **{'label': ['Car', 'Van']}** whill only process data with label is either car or van. User can insert multiple key into dictionary for filtering.

## outDict

Dictionary to store configuration for each attribute. Attribute name should be key for this dictionary. For each attribute, there have several parameters to set and all are optional:

|   **Parameters**    |   **Descriptions**   |
|----------------------------|---------------------|
|**matching**|Convert integer/float value into string category.</br> For example **{'0.0: 'no', '1.0': 'small', '2.0': 'high'}** will convert data with _key_ into _value_.</br> This is useful for script to form one hot encoding labelling. **_matching_** KEY CANNOT WORK WITH **_ranging_**. USER SHOULD CHOOSE EITHER ONE BASED ON DATA INPUT FORMAT.|
|**ranging**|Convert a range of integer/float value into string category. User should also defined _default_ in this setting.</br> For example **{'default': 'side', 'back': [[-1.67, -1.33], [0.33, 0.67]], 'front': [[1.33, 1.67], [-0.67, -0.33]]}** will firstly define all data as _side_. System will consider data as _back_ if it value between -1.67 to -1.33 and 0.33 to 0.67.</br>  This is useful for script to form one hot encoding labelling. **_ranging_** KEY CANNOT WORK WITH **_matching_**. USER SHOULD CHOOSE EITHER ONE BASED ON DATA INPUT FORMAT.|
|**method**|_classification_ or _regression_. Default is classification|
|**augmentation**|True if using _ImageDataGenerator_ augmentation method to expand training data, False otherwise. Default: False|
|**feature_range**|None if not using scalling. A tuptle contain min & max scale and script will scale data input accordingly. Default: None|
|**nn**|Define neural network model to use (ONLY SUPPORT MOBILENET). Default: mobileNet|
|**epochs**|Maximum epochs in model training. Default: 200|
|**batch_size**|Batch size use in ML. Default: 32|
|**input_tensor**|Data input shape. Default: (224, 224, 3)|
