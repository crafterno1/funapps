---
title: "Python으로 손글씨 인식 프로그램 만들기: 딥러닝 예제"
keywords: ["손글씨 인식", "딥러닝", "Python", "TensorFlow", "MNIST"]
date: "2024-08-07T22:57:46+12:00"
cover:
  image: "/images/ai18.webp"
  alt: "AI Deep Learning Model"
---

이번 글에서는 딥러닝을 사용하여 손글씨 인식 프로그램을 만드는 방법을 소개하도록 하겠습니다. 손글씨 인식은 텍스트를 이미지로부터 인식하여 디지털 텍스트로 변환하는 기술이며 OCR 등으로도 불립니다. 쉽게 따라할 수 있도록 구성하였으니 천천히 같이 구현해 보시면서 공부해 보시기 바랍니다.

## 준비 작업

### Python과 필요한 라이브러리 설치하기
손글씨 인식을 구현하기 위해선 우선 Python과 몇 가지 라이브러리를 설치해야 합니다. 다음 명령어를 사용하여 필요한 패키지를 설치할 수 있습니다:

```bash
pip install numpy tensorflow matplotlib
```
**NumPy: 과학 계산을 위한 라이브러리
**TensorFlow: 딥러닝 모델을 위한 라이브러리
**Matplotlib: 데이터 시각화를 위한 라이브러리

## 데이터 준비
손글씨 인식을 위해 유명한 MNIST 데이터셋을 사용하겠습니다. MNIST 데이터셋은 0부터 9까지의 손글씨 숫자 이미지로 구성되어 있습니다.

```python
import tensorflow as tf

# MNIST 데이터셋 불러오기
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0
```
**정규화 (Normalization): 데이터 값을 0과 1 사이로 조정하는 과정

## 딥러닝 모델 생성
손글씨 인식을 위한 딥러닝 모델을 생성합니다:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 입력층
    tf.keras.layers.Dense(128, activation='relu'),  # 은닉층
    tf.keras.layers.Dense(10, activation='softmax')  # 출력층
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
**Flatten: 2D 배열을 1D 배열로 변환하는 층
**Dense: 완전 연결 신경망 층
**relu (Rectified Linear Unit): 활성화 함수 중 하나로, 비선형성을 모델에 추가
**softmax: 다중 클래스 분류를 위한 활성화 함수

## 모델 훈련
모델을 훈련시킵니다:

```python
model.fit(train_images, train_labels, epochs=5)
```
**epochs: 데이터셋을 훈련에 사용하는 전체 반복 횟수

## 모델 평가
훈련된 모델을 사용하여 손글씨 인식 성능을 평가합니다:

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")
```

## 전체 코드 예제
아래는 위의 모든 단계를 포함한 전체 코드 예제입니다:

```python
import tensorflow as tf

# MNIST 데이터셋 불러오기
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

# 딥러닝 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(train_images, train_labels, epochs=5)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")
```

실제 애플리케이션에서는 다양한 데이터 전처리 기법과 모델을 적용하여 성능을 높일 수 있습니다.

## 마무리
이번 포스트에서는 손글씨를 인식하는 기본적인 구조의 프로그램을 다뤄보았습니다. 손글씨 인식은 다양한 텍스트 인식 애플리케이션에서 매우 유용하게 사용될 수 있으니 꼭 이해하고 넘어가시길 바랍니다.