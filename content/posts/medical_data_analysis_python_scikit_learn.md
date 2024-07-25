---
title: "Python과 Scikit-learn으로 간단한 의료 데이터 분석"
keywords: ["Python", "Scikit-learn", "의료 데이터 분석", "기계 학습", "데이터 과학"]
date: "2024-07-26T09:53:11+12:00"
cover:
  image: "/images/ai12.webp"
  alt: "AI Doctor Model"
---

의료 데이터 분석은 환자 기록, 의료 진단, 치료 결과 등을 분석하여 유의미한 인사이트를 도출하는 과정입니다. 이번 글에서는 Python과 Scikit-learn을 사용하여 간단한 의료 데이터 분석을 수행하는 방법을 소개하겠습니다. 이 가이드는 초보자도 쉽게 따라할 수 있도록 구성되어 있으니 차근차근 따라해 보시길 바래요.

## 준비 작업

### Python과 필요한 라이브러리 설치하기

우선 Python과 Scikit-learn, Pandas 라이브러리를 설치해야 합니다. 다음 명령어를 사용하여 필요한 패키지를 설치할 수 있습니다:

```bash
pip install pandas scikit-learn
```

## 데이터 준비

의료 데이터 분석을 위해서는 의료 데이터가 필요합니다. 이번 예제에서는 간단한 환자 데이터를 사용하겠습니다. 다음과 같은 형식의 데이터가 있다고 가정합니다:

```plaintext
patient_id,age,gender,bp,cholesterol,target
1,63,M,high,high,1
2,67,F,low,high,0
3,67,F,low,high,1
4,37,F,low,normal,0
5,41,M,normal,normal,0
```

이 데이터를 Pandas를 사용하여 불러옵니다:

```python
import pandas as pd

data = pd.read_csv('medical_data.csv')
print(data.head())
```

## 데이터 전처리

분석에 앞서 데이터를 전처리해야 합니다. 여기에는 결측값 처리, 범주형 데이터 인코딩 등이 포함됩니다:

```python
from sklearn.preprocessing import LabelEncoder

# 결측값 처리 (예: 결측값을 평균으로 대체)
data.fillna(data.mean(), inplace=True)

# 범주형 데이터 인코딩
labelencoder = LabelEncoder()
data['gender'] = labelencoder.fit_transform(data['gender'])
data['bp'] = labelencoder.fit_transform(data['bp'])
data['cholesterol'] = labelencoder.fit_transform(data['cholesterol'])
print(data.head())
```

## 데이터 분할

이제 데이터를 훈련 세트와 테스트 세트로 분할합니다:

```python
from sklearn.model_selection import train_test_split

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 모델 훈련

다음으로, 로지스틱 회귀 모델을 사용하여 데이터를 학습합니다:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

## 모델 평가

훈련된 모델을 사용하여 테스트 데이터에 대한 예측을 수행하고, 정확도를 평가합니다:

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.2f}")
```

## 전체 코드 예제

아래는 위의 모든 단계를 포함한 전체 코드 예제입니다:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 불러오기
data = pd.read_csv('medical_data.csv')

# 결측값 처리 (예: 결측값을 평균으로 대체)
data.fillna(data.mean(), inplace=True)

# 범주형 데이터 인코딩
labelencoder = LabelEncoder()
data['gender'] = labelencoder.fit_transform(data['gender'])
data['bp'] = labelencoder.fit_transform(data['bp'])
data['cholesterol'] = labelencoder.fit_transform(data['cholesterol'])

# 데이터 분할
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = LogisticRegression()
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.2f}")
```

이 코드는 간단한 의료 데이터 분석의 기본적인 구조를 제공합니다. 실제 분석의 정확도를 높이기 위해서는 다양한 데이터 전처리 기법과 모델을 적용할 수 있습니다.

## 마무리

이번 글에서는 Python과 Scikit-learn을 사용하여 간단한 의료 데이터 분석을 수행하는 방법을 소개했습니다. 의료 데이터 분석은 환자의 건강 상태를 예측하고 진단하는 데 중요한 역할을 합니다. 다음 포스트에서는 더 심화된 내용으로 AI 기반의 자율주행 시뮬레이션을 만들어 보도록 하겠습니다.