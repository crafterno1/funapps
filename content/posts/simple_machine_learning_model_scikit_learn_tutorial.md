---
title: "초보자를 위한 간단한 머신러닝 모델 만들기: Scikit-learn 튜토리얼"
keywords: ["머신러닝", "Scikit-learn 튜토리얼", "Python 머신러닝", "초보자 머신러닝"]
date: "2024-07-21"
cover:
  image: "/images/ai1.webp"
  alt: "AI Machine Learning"
---

머신러닝은 데이터 분석과 인공지능의 핵심 기술 중 하나입니다. 이번 포스트에서는 초보자도 쉽게 따라할 수 있는 Scikit-learn을 이용한 간단한 머신러닝 모델 만들기를 소개해 보려고 합니다. 이 튜토리얼을 통해 머신러닝의 기본 개념에 대해서 간접적으로나마 경험해 보실수 있기를 바래요.

## Scikit-learn 소개
Scikit-learn은 Python에서 가장 널리 사용되는 머신러닝 라이브러리 중 하나로, 간단한 인터페이스와 다양한 알고리즘을 제공합니다. 데이터 전처리, 모델 학습, 평가 등을 쉽게 할 수 있어 초보자에게 적합합니다.

## 환경 설정
먼저, Scikit-learn과 필요한 라이브러리를 설치해야 합니다. 터미널에 아래 명령어를 입력해 주세요.

```
pip install numpy pandas scikit-learn matplotlib
```

## 데이터 준비
우리는 Iris 데이터셋을 사용할 것입니다. 이 데이터셋은 꽃잎과 꽃받침의 길이와 너비를 기준으로 세 종류의 붓꽃을 분류하는 데 사용됩니다.

```
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
```

## 데이터 전처리
데이터를 전처리하고 학습 데이터와 테스트 데이터로 나눕니다.

```
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 모델 학습
이번 튜토리얼에서는 K-최근접 이웃(K-Nearest Neighbors, KNN) 알고리즘을 사용하여 모델을 학습시킵니다.

```
from sklearn.neighbors import KNeighborsClassifier

# 모델 생성
knn = KNeighborsClassifier(n_neighbors=3)

# 모델 학습
knn.fit(X_train, y_train)
```

## 모델 평가
학습한 모델을 평가하고 정확도를 확인합니다.

```
from sklearn.metrics import accuracy_score

# 예측
y_pred = knn.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'모델 정확도: {accuracy:.2f}')
```

## 시각화
마지막으로, 학습된 모델의 결과를 시각화해 봅니다.

```
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 데이터 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 시각화
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1], label=iris.target_names[i])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.title('Iris 데이터셋의 PCA 시각화')
plt.show()
```

## 결론
이 튜토리얼에서는 Scikit-learn을 사용하여 간단한 머신러닝 모델을 만드는 과정을 다뤘습니다. 데이터 준비, 전처리, 모델 학습, 평가, 시각화까지의 전 과정을 담아보았는데요, 이 과정을 통해 머신러닝의 기본 개념을 이해하고 실습할 수 있으셨기를 바랍니다.

다음에는 조금 더 복잡하지만 재밌는 내용을 소개해 드리도록 할게요~!