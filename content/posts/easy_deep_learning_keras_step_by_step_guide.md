---
title: "Keras를 이용한 딥러닝 모델 구축: 손쉬운 단계별 가이드"
keywords: ["Keras 튜토리얼", "딥러닝 모델 구축", "Python 딥러닝", "Keras 기초"]
date: "2024-07-22T21:45:17+12:00"
cover:
  image: "/images/ai4.webp"
  alt: "AI Deep Learning Model"
---

이번 포스트에서는 Keras를 사용하여 딥러닝 모델을 구축하는 방법을 소개해 보도록 할게요. 초보자분들도 쉽게 따라하실 수 있는 단계별 가이드를 통해서 딥러닝의 기초를 학습해 보시길 바래요.

## Keras 소개
Keras는 Python에서 사용할 수 있는 고수준 신경망 API로, TensorFlow의 위에서 동작합니다. 사용이 간편하고 직관적이기 떄문에 딥러닝 모델을 신속하게 프로토타이핑하는 데 많이 사용된답니다.

## 환경 설정
먼저, Keras와 필요한 라이브러리를 설치해야 합니다. 터미널에 아래 명령어를 입력해 주세요.

```
pip install tensorflow numpy matplotlib
```

## 데이터 준비
Keras에서 제공하는 MNIST 데이터셋을 사용하여 숫자 이미지를 분류하는 모델을 만들어 보겠습니다. MNIST 데이터셋은 0부터 9까지의 손글씨 숫자 이미지로 구성되어 있습니다.

```
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 데이터 로드
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 정규화
X_train, X_test = X_train / 255.0, X_test / 255.0

# 데이터 형상 변환
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

## 모델 생성
이제 CNN(Convolutional Neural Network)을 사용하여 모델을 생성해 보겠습니다.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

## 모델 학습
이제 모델을 학습시켜 보겠습니다. 학습 데이터로 모델을 훈련시키고, 테스트 데이터로 성능을 평가합니다.

```
# 모델 학습
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## 모델 평가
학습된 모델의 성능을 평가하고 정확도를 확인합니다.

```
# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'테스트 정확도: {test_acc:.2f}')
```

## 예측 결과 시각화
학습된 모델을 사용하여 예측 결과를 시각화해 봅니다.

```
import numpy as np
import matplotlib.pyplot as plt

# 예측 수행
predictions = model.predict(X_test)

# 예측 결과 시각화
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel(f"{predicted_label} ({true_label})", color=color)

# 예제 이미지와 예측 결과 출력
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plot_image(i, predictions, y_test, X_test)
plt.show()
```

## 결론
이 튜토리얼에서는 Keras를 사용하여 간단한 딥러닝 모델을 만드는 과정을 다뤘습니다. 데이터 준비, 모델 생성, 학습, 평가, 시각화까지의 전 과정을 통해 딥러닝의 기초를 이해하고 실습할 수 있으셨기를 바라면서 다음 포스트에서는 OpenCV를 사용하여 얼굴을 인식하는 간단한 프로그램을 만들어 보도록 하겠습니다.