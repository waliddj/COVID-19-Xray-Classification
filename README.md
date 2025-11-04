# COVID-19-Xray-Classification
A Convolutional Neuron Network (CNN) model for COVID-19 Xray Classification, with COVID-19 Pneumonia Detector.

Dataset: [COVID-19 Xray Dataset (Train & Test Sets)](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/code) .

Dataset structure:

\train

    |__\NORMAL

        |__74 files
    
    |__\PNEUMONIA

        |__74 files
\test

    |__\NORMAL
    
        |__20 files
    
    |__\PNEUMONIA

        |__20 files

# Code architecture:
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import pandas as pd
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import numpy as np
import pathlib
import kagglehub
import os
import matplotlib.pyplot as plt
```
## Dataset
### Download dataset from [Kaggle](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/data) using ```kagglehub.dataset_download()``` function:
```python
path = kagglehub.dataset_download("khoongweihao/covid19-xray-dataset-train-test-sets")
print("Path to dataset files:", path)
dataset_dir = 'C:/Users/walid/.cache/kagglehub/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/versions/1/xray_dataset_covid19/'
```
### Split train and test data
```python
train_data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/versions/1/xray_dataset_covid19/train'
test_data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/versions/1/xray_dataset_covid19/test'
```
### Get class names
```python
data_dir = pathlib.Path(test_data_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
```
## Data preprocessing
### Data normalization using ```ImageDataGenerator()``` TensorFlow's function:
```python
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)
test_data = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)
```
## Build the model
---
**Model architecture:**
- ```input``` layer with 10 ```filters```, a ```kernel_size```=3, ```input_shape```= ```(224,224,3)``` and a ```ReLU``` activation method.
- ```Conv2D``` layer with same parameters as the ```input``` layer but *without the ```input_shape```*.
- ```MaxPool2D```layer = ```(2,2)```  *(with defaul value)*.
- 2 ```Conv2D``` layers followed by a ```MaxPool2D``` layer.
- ```Flatten``` layer, followed by an ```output``` layer with an ```output_shape``` = ```1```, and a ```sigmoid``` activation methofd *(Because it's a Binary classification model)* .
---
Create the model
```python
model = Sequential([
    Conv2D(filters=10,kernel_size=3, input_shape=(224,224,3), activation='relu'),
    Conv2D(10,3,activation='relu'),
    MaxPool2D(),
    Conv2D(10,3,activation='relu'),
    Conv2D(10,3,activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation='sigmoid')
])
```
Compile the model
```python
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)
```
Fit the model to the data and track its history for evaluation
```python
history_model = model.fit(train_data,
                    epochs=6,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))

```

## Model evaluation
```python
pd.DataFrame(history_model.history).plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
```
|Metrics|Accuracy|Loss|
|-------|--------|----|
|Train| 93.10%|0.165 |
|Test| 100%| 0.032|

## Save the model
```python
model.save("C:/Users/walid/Desktop/Covid.keras")
```
# Plot random xRay images with their predication 
<img width="640" height="480" alt="nr" src="https://github.com/user-attachments/assets/aaaf144e-627b-4b77-920f-8f0482132310" />
<img width="640" height="480" alt="pn" src="https://github.com/user-attachments/assets/2b6dc869-5435-4817-bf5b-ce3b1873515a" />
<img width="640" height="480" alt="nr_2" src="https://github.com/user-attachments/assets/751aded0-9b26-4608-8002-1c9874d4d6c7" />
<img width="640" height="480" alt="pn_2" src="https://github.com/user-attachments/assets/2d74c09b-4349-45d3-be10-ac10cddd88ee" />
<img width="640" height="480" alt="pn_3" src="https://github.com/user-attachments/assets/6e15edab-075f-486b-8ebc-331119489a5d" />

# Appendix (Experiments)


## MODEL 1
## Data preprocessing:
- Data normalization
## Architecture:
- ```input``` layer with a ```ReLU``` activation method,```input_shape```= ```(224,224,3)```, ```filters```=3,```kernel_size```=3.
- 2 ``` Conv2D``` layers with same parameters as the ```input```layer (without the input shape).
- ```Flatten``` layer followed by an ```output``` layer with an a ```sigmoid``` activation method.
## Evaluation
|Metrics|Accuracy|Loss|
|-------|--------|----|
|Train| 93.01%| 0.224|
|Test| 97.50%|0.09 |

> **Note:** The testaccuracy increases from 43% (in the first epoch) to 97.50% in the 6th epoch. Moreover, its loss function drops from 0.4293  to 0.09.

> ***Conclusion:*** The model performance is acceptable.
***********************************************

## MODEL 2
## Data preprocessing:
- Data normalization
## Architecture modifications:
- Add a ```MaxPool2D``` layer just before the ```Flatten``` layer.
## Evaluation
|Metrics|Accuracy|Loss|
|-------|--------|----|
|Train| 91.35%|0.246 |
|Test| 100%| 0.042|

> **Note:** The Test accuracy increases from 50% in the frist epoch to 100% in the 5th epoch. Moreover, the loss fucntion drops from 1.4603 to 0.0442 in the 6th epoch.
Nevertheless, the experimental results showed that the model  does not reach a 100% accuracy.

> ***Conclusion:*** The addition of the ```MaxPool2D``` layer leaded to a significant improvement of the model.
****************************************************

## MODEL 3
## Data preprocessing:
- Data normalization
## Architecture modifications:
- Add a ```MaxPool2D``` layer.
- Add a ```Conv2D``` layer.
## Evaluation
|Metrics|Accuracy|Loss|
|-------|--------|----|
|Train| 93.10%|0.165 |
|Test| 100%| 0.032|

> **Note:** 
The new model architecture leaded to a diminution in the loss function value.

> ***Conclusion:*** 
Model 3's architecture is the most optimal for this classification problem.
