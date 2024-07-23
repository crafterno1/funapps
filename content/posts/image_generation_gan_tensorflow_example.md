---
title: "GAN을 활용한 이미지 생성: TensorFlow 예제"
keywords: ["GAN", "이미지 생성", "TensorFlow", "딥러닝", "생성적 적대 신경망", "GAN 예제"]
date: "2024-07-24T10:02:51+12:00"
cover:
  image: "/images/ai8.webp"
  alt: "AI Deep Learning Model"
---

GAN(Generative Adversarial Network, 생성적 적대 신경망)은 두 개의 신경망을 활용하여 서로 경쟁하면서 데이터를 생성하는 딥러닝 모델입니다. 이번 글에서는 TensorFlow를 사용하여 GAN을 구현하고, 이를 통해 이미지를 생성하는 과정을 살펴보겠습니다.

## GAN의 기본 개념

GAN은 생성자(Generator)와 판별자(Discriminator)라는 두 개의 신경망으로 구성됩니다. 생성자는 무작위 노이즈를 입력받아 가짜 이미지를 생성하고, 판별자는 이 이미지가 진짜인지 가짜인지 판별합니다. 두 신경망은 서로 경쟁하면서 성능이 향상됩니다.

## TensorFlow 설치하기

먼저 TensorFlow를 설치해야 합니다. 다음 명령어를 사용해 TensorFlow를 설치할 수 있습니다:

```bash
pip install tensorflow
```

설치가 완료되면 TensorFlow를 사용해 GAN을 구현할 수 있습니다.

## 데이터셋 준비
GAN을 훈련시키기 위해 데이터셋이 필요합니다. 이번 예제에서는 MNIST 데이터셋을 사용하겠습니다:

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# MNIST 데이터셋 불러오기
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # -1과 1 사이로 정규화
```

## 생성자 모델 만들기
생성자 모델은 무작위 노이즈를 입력받아 이미지를 생성합니다. 간단한 CNN 구조로 생성자를 구현할 수 있습니다:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, ReLU

def build_generator():
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    return model
```

## 판별자 모델 만들기
판별자 모델은 이미지를 입력받아 진짜인지 가짜인지 판별합니다. 간단한 CNN 구조로 판별자를 구현할 수 있습니다:

```python
from tensorflow.keras.layers import Flatten, Conv2D, LeakyReLU, Dropout

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1))
    
    return model
```

## GAN 훈련하기
생성자와 판별자를 결합하여 GAN을 훈련합니다. GAN 훈련 과정은 다음과 같습니다:

```python
import numpy as np
import matplotlib.pyplot as plt

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 판별자 손실 함수
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# 생성자 손실 함수
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        
        # 생성된 이미지 시각화
        noise = tf.random.normal([16, 100])
        generated_images = generator(noise, training=False)
        fig = plt.figure(figsize=(4, 4))
        
        for i in range(generated_images.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        
        plt.show()

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, epochs=50)
```

위 코드를 실행하면 GAN 모델이 훈련되며, 훈련 과정 중에 생성된 이미지들을 시각화할 수 있습니다.

## 마무리
이번 글에서는 TensorFlow를 이용해 GAN을 구현하고 이미지를 생성하는 방법을 살펴보았습니다. GAN은 생성자와 판별자가 경쟁하면서 성능이 향상되는 딥러닝 모델로, 다양한 응용 분야에서 활용될 수 있습니다. 다음글에서는 조금 더 재밌는 내용으로 Python을 사용해서 간단한 텍스트 요약 프로그램을 만들어 보도록 하겠습니다. 다음 포스트도 기대해 주세요~.