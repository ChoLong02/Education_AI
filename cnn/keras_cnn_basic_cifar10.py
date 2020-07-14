######################################
## KERAS를 사용한 CNN 모델 사용하기 ##
##  - CIFAR-10 데이터셋 사용        ##
######################################
## 주의: CPU 사용자는 GPU 코드라인 주석후 사용하세요.

import tensorflow as tf
import keras
from keras import utils
from keras import layers
from keras import datasets
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils.multi_gpu_utils import multi_gpu_model # Multi GPU 사용하기 위한 라이브러리

import numpy as np
import matplotlib.pyplot as plt

###########################
## 1.데이터 로드 및 탐색 ##
###########################

# CIFAR-10은 60000개의 32x32 컬러 이미지로 10개 클래스로 구성 → class_names[]
# Train 50000개(5000 x 10(클래스))
# Test 10000개(1000 x 10(클래스))
cifar_mnist = datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar_mnist.load_data()
# print(train_images.shape) (50000, 32, 32, 3) -> (이미지 개수, 32 x 32 픽셀 이미지, 색채널 1:흑백 / 3:RGB)

# 미리 정해진 사물들을 알기쉽게 텍스트로
class_names = [
    'Airplane',
    'Car',
    'Birds',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

#####################
## 2.데이터 전처리 ##
#####################

batch_size = 64
num_classes = 10
epochs = 35

# float32 타입으로 정규화 후 255로 나눔
# → Keras 데이터를 0에서 1사이의 값으로 구동시 최적의 성능을 보임
# → 현재 0~255 사이의 값들을 0~1사이의 값으로 변경(각 픽셀의 강도를 최대 강도 값인 255로 나눔)
train_images = train_images.astype('float32')
train_images = train_images / 255

test_images = test_images.astype('float32')
test_images = test_images / 255

# utils.to_categorical() → 원핫인코딩로 변환
# → num_classes가 10개기 때문에 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]과 같은 방식으로 변환
train_labels = utils.to_categorical(train_labels, num_classes)
test_labels = utils.to_categorical(test_labels, num_classes)

#####################
## 3.CNN 모델 구성 ##
#####################

model = keras.Sequential([
    # Conv2D
    #  1번 매개변수(32) → 컨볼루션 필터의 수
    #  kernel_size=(3, 3 → 컨벌루션 커널의(행, 열)
    #  padding → 경계 처리 방법을 정의
    #             - valid: 유효한 영역만 출력, 출력이미지 사이즈가 입력사이즈보다 작음
    #             - same: 출력이미지 사이즈가 입력 이미지 사이즈와 동일
    # input_shape → 샘플 수를 제외한 입력 형태를 정의(모델에서 첫 레이어에서만 정의하면 됨 → (행, 열, 채널수(1 or 3)))
    Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=train_images.shape[1:], activation=tf.nn.relu),
    MaxPooling2D(pool_size=(2, 2)), # (2, 2) → 출력영상크기는 입력영상크기의 반으로 줄어듬
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(), # Flatten() → 영상을 일차원으로 변경, 컨볼루션과 맥스풀링 레이어를 반복적으로 거치면 주요 특징만 추출됨,
               #             컨볼루션과 맥스풀링은 주로 2차원 데이터를 다루지만, 전결합층에 전달하기 위해서 1차원 자료로 변경
    Dense(64, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(num_classes, activation=tf.nn.softmax)
])

model.summary()

# MULTI GPU 사용하기, CPU 사용시에는 주석
model = multi_gpu_model(model, gpus=2)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#####################
## 4.CNN 모델 훈련 ##
#####################

# 과대적합을 막기위해서 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    train_images, train_labels,
    epochs=epochs,
    validation_data=(test_images, test_labels),
    shuffle=True,
    callbacks=[early_stopping]
)

#####################
## 5.CNN 모델 평가 ##
#####################

loss, acc = model.evaluate(test_images, test_labels)
print('\nLoss: {}, Acc: {}'.format(loss, acc))


def plt_show_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)


def plt_show_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)


plt_show_loss(history)
plt.show()

plt_show_acc(history)
plt.show()


################
## 6.예측하기 ##
################

# 예측
predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == np.argmax(true_label):
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[np.argmax(true_label)]),
                                         color=color)

def plot_value_array(i, prediction_array, true_label):
    prediction_array, true_label = prediction_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)

    thisplot[predicted_label].set_color('red')
    thisplot[np.argmax(true_label)].set_color('blue')


num_rows = 5
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
