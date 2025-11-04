"""
*****************************************************************************************
This model was built by: Djaid Walid

__________________________________________________________________________________________________
                                   Contacts                                                      |
__________________________________________________________________________________________________
Github     | https://github.com/waliddj                                                          |
Linkedin   | www.linkedin.com/in/walid-djaid-375777229                                           |
Instagram  | https://www.instagram.com/d.w.science?igsh=MWlnMmNpOTM2OW0xaA%3D%3D&utm_source=qr   |
__________________________________________________________________________________________________

Dataset used to train this model is : COVID-19 Xray Dataset (Train & Test Sets).
Link to the dataset: https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/data
*****************************************************************************************
"""



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

# Download latest version of the dataset
path = kagglehub.dataset_download("khoongweihao/covid19-xray-dataset-train-test-sets")
print("Path to dataset files:", path)


dataset_dir = 'C:/Users/walid/.cache/kagglehub/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/versions/1/xray_dataset_covid19/'


for dirpath, dirnames, filenames in os.walk(dataset_dir):
    print(f"there are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")



train_data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/versions/1/xray_dataset_covid19/train'
test_data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/versions/1/xray_dataset_covid19/test'

data_dir = pathlib.Path(test_data_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

# ************** Preprocess the data ****************
# Normalize the data
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

# ******************** Build Model *********************

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

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)
history_model = model.fit(train_data,
                    epochs=6,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))

# ******************** Model Evaluation ***********************************
pd.DataFrame(history_model.history).plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

# ********************* Save the model ****************************
model.save("C:/Users/walid/Desktop/Covid.keras")


# ****************** Plot random xRay images with their predication ********************

md = tf.keras.models.load_model("C:/Users/walid/Desktop/Covid.keras")

import random
import pathlib

# Create a function for plotting a random image along with its prediction
def plot_random_prediction(model, data_dir, class_names, img_size=(224, 224)):
    """
    data_dir: Path or str to the test folder that contains one subfolder per class
    class_names: np.array or list of class names in the same order used by the model
    img_size: target size used to train the model
    """
    data_dir = pathlib.Path(data_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts and p.parent != data_dir]
    if not all_images:
        raise ValueError("No images found under data_dir. Check your path and folder structure.")

    img_path = random.choice(all_images)
    true_class = img_path.parent.name
    # Map true class name to index according to class_names
    class_names = np.array(class_names).astype(str)
    try:
        true_idx = np.where(class_names == true_class)[0][0]
    except IndexError:
        raise ValueError(f"True class '{true_class}' not found in class_names {class_names}.")

    # Load and preprocess image
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_arr = tf.keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(img_arr, axis=0)

    # Predict
    prob = float(model.predict(x, verbose=0).squeeze())  # sigmoid output in [0,1]
    pred_idx = int(prob >= 0.5)
    pred_class = class_names[pred_idx]

    # Confidence for the predicted class
    confidence = prob if pred_idx == 1 else (1.0 - prob)

    # Plot
    plt.figure()
    plt.imshow(img_arr)
    plt.axis("off")
    correct = (pred_idx == true_idx)
    title_color = "green" if correct else "red"
    plt.title(f"Pred: {pred_class} ({confidence*100:.1f}%) | True: {true_class}", color=title_color)
    plt.show()


plot_random_prediction(md, test_data_dir, class_names, img_size=(224, 224))



# ********************* Experiments ***************************
"""
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

> ***Conclusion:*** 
"""