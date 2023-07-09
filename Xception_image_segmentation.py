import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from keras.utils import plot_model
from keras.utils import image_dataset_from_directory
import zipfile

import os, shutil, pathlib
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


"""
[Download and extract zip files] 
"""

file_name = 'dogs-vs-cats.zip'
if os.path.exists(file_name):
    pass
else:
    import gdown
    gdown.download(id='18uC7WTuEXKJDDxbj-Jq6EjzpFrgE7IAd', output=file_name)

    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall('extracted_files/')
        with zipfile.ZipFile('extracted_files/train.zip', 'r') as zip_2nd:
            zip_2nd.extractall('extracted_files/')


"""
[Train/validation/test] 
"""

original_dir = pathlib.Path("extracted_files/train")
new_base_dir = pathlib.Path("cats_vs_dogs_small")

def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)

path_name = "cats_vs_dogs_small"
if os.path.exists(path_name):
    pass
else:
    make_subset("train", start_index=0, end_index=1000)
    make_subset("validation", start_index=1000, end_index=1500)
    make_subset("test", start_index=1500, end_index=2500)


"""
[make dataset] 
"""

train_ds = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32)
val_ds = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32)
test_ds = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


"""
[data visualization] 
"""

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()


"""
[improvement performance] 
"""

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

## 필터 개수가 변경되는 잔차 블록
def get_model1():

    inputs = keras.Input(shape=(180, 180, 3))

    ## 잔차를 따로 저장
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    residual = x

    ## 출력 필터 32 --> 64 증가 / same padding으로 다운샘플링 방지
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    ## 잔차가 32개 필터이기에 동일하게 하기 위해 잔차 층 수정.
    residual = layers.Conv2D(64, 1)(residual)

    x = layers.add([x, residual])

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


model = get_model1()
model.summary()
# keras.utils.plot_model(model, show_shapes=True)


epochs = 50

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks = [keras.callbacks.ModelCheckpoint(
    filepath='like_Xception.keras',
    save_best_only=True,
    monitor='val_loss')
    ]

history = model.fit(train_ds,
                    epochs=epochs,
                    callbacks=callbacks,
                    batch_size=32,
                    validation_data=val_ds)