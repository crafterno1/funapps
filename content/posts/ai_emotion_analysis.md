---
title: "딥러닝을 활용한 간단한 감정 분석 프로그램 만들기"
keywords: ["딥러닝", "감정 분석", "자연어 처리", "Python", "AI"]
date: "2024-08-02T14:28:26+12:00"
cover:
  image: "/images/ai17.webp"
  alt: "AI Deep Learning Model"
---

이번 글에서는 딥러닝을 사용하여 간단한 감정 분석 프로그램을 만드는 방법을 소개하게 하도록 하겠습니다. 감정 분석(Sentiment Analysis)은 텍스트 데이터에서 긍정, 부정, 중립과 같은 감정을 분석하는 기술이며 앞으로 다가올 휴머노이드 시대에 꼭 필요한 기반 기술입니다. 이 가이드는 초보자도 쉽게 따라할 수 있도록 구성되어 있으니 천천히 따라해 보시면서 익히시길 바랍니다.

## 준비 작업

### Python과 필요한 라이브러리 설치하기

감정 분석을 구현하기 위해서는 Python과 몇 가지 라이브러리를 설치해야 합니다. 다음 명령어를 사용하여 필요한 패키지를 설치하세요 :

```bash
pip install numpy pandas tensorflow sklearn
```

- **NumPy**: 과학 계산을 위한 라이브러리
- **Pandas**: 데이터 분석을 위한 라이브러리
- **TensorFlow**: 딥러닝 모델을 위한 라이브러리
- **scikit-learn (sklearn)**: 기계 학습을 위한 라이브러리

## 데이터 준비

감정 분석을 위한 훈련 데이터를 준비합니다. 여기서는 간단한 예제로 IMDb 영화 리뷰 데이터셋을 사용하겠습니다:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 불러오기
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
df = pd.read_csv(url, compression='gzip', error_bad_lines=False)

# 데이터셋 나누기
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
```

## 텍스트 전처리

텍스트 데이터를 딥러닝 모델에 적합한 형태로 전처리합니다:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 토크나이저 설정
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['review'])

# 텍스트 시퀀스 변환
train_sequences = tokenizer.texts_to_sequences(train_data['review'])
train_padded = pad_sequences(train_sequences, maxlen=200)

test_sequences = tokenizer.texts_to_sequences(test_data['review'])
test_padded = pad_sequences(test_sequences, maxlen=200)
```

- **Tokenizer**: 텍스트를 숫자 시퀀스로 변환하는 도구
- **pad_sequences**: 시퀀스의 길이를 동일하게 맞추기 위해 패딩을 추가하는 함수

## 딥러닝 모델 생성

감정 분석을 위한 딥러닝 모델을 생성합니다:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=200),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- **Embedding**: 단어를 고차원 공간에 매핑하는 층
- **GlobalAveragePooling1D**: 시퀀스의 평균을 구하는 층
- **Dense**: 완전 연결 신경망 층
- **binary_crossentropy**: 이진 분류를 위한 손실 함수
- **sigmoid**: 출력값을 0과 1 사이로 변환하는 활성화 함수

## 모델 훈련

모델을 훈련시킵니다:

```python
history = model.fit(train_padded, train_data['sentiment'], epochs=10, validation_data=(test_padded, test_data['sentiment']), verbose=2)
```

## 모델 평가

훈련된 모델을 사용하여 감정 분석을 평가합니다:

```python
loss, accuracy = model.evaluate(test_padded, test_data['sentiment'])
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

## 전체 코드 예제

아래는 위의 모든 단계를 포함한 전체 코드 예제입니다:

```python
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터 불러오기
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
df = pd.read_csv(url, compression='gzip', error_bad_lines=False)

# 데이터셋 나누기
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# 텍스트 전처리
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['review'])

train_sequences = tokenizer.texts_to_sequences(train_data['review'])
train_padded = pad_sequences(train_sequences, maxlen=200)

test_sequences = tokenizer.texts_to_sequences(test_data['review'])
test_padded = pad_sequences(test_sequences, maxlen=200)

# 딥러닝 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=200),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
history = model.fit(train_padded, train_data['sentiment'], epochs=10, validation_data=(test_padded, test_data['sentiment']), verbose=2)

# 모델 평가
loss, accuracy = model.evaluate(test_padded, test_data['sentiment'])
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

이 코드는 간단한 감정 분석의 기본적인 구조를 제공합니다. 실제 애플리케이션에서는 다양한 데이터 전처리 기법과 모델을 적용하여 성능을 높일 수 있습니다.

## 마무리

이번 글에서는 딥러닝을 사용하여 간단한 감정 분석 프로그램을 만드는 방법을 소개하였습니다. 감정 분석은 다양한 텍스트 데이터 분석 애플리케이션에서 매우 유용하게 사용될 수 있습니다. 다음 포스트에서는 조금 더 재밌는 내용인 Python으로 손글씨 인식 프로그램을 만들어 보도록 하겠습니다.
