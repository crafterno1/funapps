---
title: "텍스트 자동 완성 기능 구현: 자연어 처리 실습"
keywords: ["텍스트 자동 완성", "자연어 처리", "Python", "기계 학습", "NLP"]
date: "2024-07-27T20:16:38+12:00"
cover:
  image: "/images/ai14.webp"
  alt: "AI Deep Learning Model"
---

텍스트 자동 완성 기능은 사용자가 입력하는 단어나 문장을 예측하여 자동으로 완성하는 기능입니다. 이번 글에서는 Python과 자연어 처리(NLP) 기술을 사용하여 간단한 텍스트 자동 완성 기능을 구현하는 방법을 소개하겠습니다. 이 가이드는 초보자도 쉽게 따라할 수 있도록 구성되어 있습니다.

## 준비 작업

### Python과 필요한 라이브러리 설치하기

우선 Python과 몇 가지 주요 라이브러리를 설치해야 합니다. 다음 명령어를 사용하여 필요한 패키지를 설치할 수 있습니다:

```bash
pip install numpy pandas tensorflow keras nltk
```

## 데이터 준비
텍스트 자동 완성 모델을 학습시키기 위해서는 대량의 텍스트 데이터가 필요합니다. 이번 예제에서는 NLTK 라이브러리에서 제공하는 셰익스피어의 작품을 사용하겠습니다:

```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

texts = gutenberg.raw('shakespeare-hamlet.txt')
```
## 데이터 전처리
데이터를 전처리하여 모델 학습에 적합한 형태로 변환합니다. 여기에는 텍스트 토큰화와 시퀀스 생성이 포함됩니다:

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 텍스트 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts([texts])
total_words = len(tokenizer.word_index) + 1

# 시퀀스 생성
input_sequences = []
for line in texts.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 시퀀스 패딩
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# 특징과 레이블 분리
import numpy as np
input_sequences = np.array(input_sequences)
X, y = input_sequences[:,:-1], input_sequences[:,-1]

# 레이블 원핫 인코딩
y = tf.keras.utils.to_categorical(y, num_classes=total_words)
```

## 모델 구성
간단한 신경망 모델을 사용하여 텍스트 자동 완성 모델을 구성합니다:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## 모델 훈련
모델을 훈련시킵니다:

```python
history = model.fit(X, y, epochs=100, verbose=1)
```

## 텍스트 자동 완성 함수
훈련된 모델을 사용하여 텍스트 자동 완성 기능을 구현합니다:

```python
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    predicted_word = tokenizer.index_word[predicted_word_index[0]]
    return predicted_word

seed_text = "To be or not to be"
next_word = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
print(f"Seed text: '{seed_text}', Next word: '{next_word}'")
```

## 전체 코드 예제
아래는 위의 모든 단계를 포함한 전체 코드 예제입니다:

```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 데이터 준비
texts = gutenberg.raw('shakespeare-hamlet.txt')

# 텍스트 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts([texts])
total_words = len(tokenizer.word_index) + 1

# 시퀀스 생성
input_sequences = []
for line in texts.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 시퀀스 패딩
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# 특징과 레이블 분리
input_sequences = np.array(input_sequences)
X, y = input_sequences[:,:-1], input_sequences[:,-1]

# 레이블 원핫 인코딩
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 신경망 모델 구성
model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(X, y, epochs=100, verbose=1)

# 텍스트 자동 완성 함수
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    predicted_word = tokenizer.index_word[predicted_word_index[0]]
    return predicted_word

seed_text = "To be or not to be"
next_word = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
print(f"Seed text: '{seed_text}', Next word: '{next_word}'")
```

이 코드는 간단한 텍스트 자동 완성 기능의 기본적인 구조를 제공합니다. 실제 기능의 성능을 높이기 위해서는 다양한 데이터 전처리 기법과 모델을 적용할 수 있습니다.

## 마무리
이번 글에서는 Python과 자연어 처리를 사용하여 간단한 텍스트 자동 완성 기능을 구현하는 방법을 소개했습니다. 텍스트 자동 완성 기능은 다양한 애플리케이션에서 유용하게 사용될 수 있습니다. 다음 포스트에서는 조금 더 심화된 내용인 'Python과 OpenCV를 이용한 실시간 객체 추적'을 다뤄보도록 하겠습니다.