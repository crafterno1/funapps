---
title: "Python으로 영화 추천 시스템 만들기: 기초 추천 알고리즘"
keywords: ["Python", "영화 추천 시스템", "추천 알고리즘", "기계 학습", "데이터 과학"]
date: "2024-07-25T09:34:15+12:00"
cover:
  image: "/images/ai10.webp"
  alt: "AI Python Learning"
---

# Python으로 영화 추천 시스템 만들기: 기초 추천 알고리즘

추천 시스템은 사용자의 과거 행동과 선호도를 기반으로 개인화된 추천을 제공하는 시스템입니다. 이번 글에서는 Python을 사용하여 간단한 영화 추천 시스템을 만드는 방법을 소개하겠습니다. 이 시스템은 기본적인 추천 알고리즘을 활용하여 사용자가 좋아할 만한 영화를 추천합니다.

## 준비 작업

### Python과 필요한 라이브러리 설치하기

우선 Python과 필요한 라이브러리를 설치해야 합니다. Pandas와 예전 포스트에서 한번 사용해 보았던 Scikit-learn 라이브러리를 사용합니다. 다음 명령어를 사용하여 필요한 패키지를 설치할 수 있습니다:

```bash
pip install pandas scikit-learn
```

### 데이터 준비
영화 추천 시스템을 만들기 위해서는 영화 데이터가 필요합니다. 이번 예제에서는 간단한 영화 평점 데이터를 사용하겠습니다. 다음과 같은 형식의 데이터가 있다고 가정합니다:

```plaintext
user_id,movie_id,rating
1,1,4
1,2,5
2,1,3
2,3,4
```

이 데이터를 Pandas를 사용하여 불러옵니다:

```python
import pandas as pd

data = pd.read_csv('ratings.csv')
print(data.head())
```

## 유사도 계산
추천 시스템의 핵심은 유사도를 계산하는 것입니다. 여기서는 코사인 유사도를 사용하여 영화 간의 유사도를 계산합니다:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

movie_user_matrix = data.pivot_table(index='movie_id', columns='user_id', values='rating').fillna(0)
movie_similarity = cosine_similarity(movie_user_matrix)
print(movie_similarity)
```

## 영화 추천
유사도를 기반으로 사용자가 아직 보지 않은 영화를 추천합니다. 다음 코드는 특정 사용자가 보지 않은 영화를 추천하는 예제입니다:

```python
def recommend_movies(user_id, movie_user_matrix, movie_similarity, n=5):
    user_ratings = movie_user_matrix[user_id]
    similar_scores = movie_similarity.dot(user_ratings)
    scores = [(movie_id, score) for movie_id, score in enumerate(similar_scores)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = [movie_id for movie_id, score in scores if movie_id not in user_ratings.index][:n]
    return recommendations
```

## 전체 코드 예제
아래는 위의 모든 단계를 포함한 전체 코드 예제입니다:

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 데이터 불러오기
data = pd.read_csv('ratings.csv')

# 유저-영화 매트릭스 생성
movie_user_matrix = data.pivot_table(index='movie_id', columns='user_id', values='rating').fillna(0)

# 코사인 유사도 계산
movie_similarity = cosine_similarity(movie_user_matrix)

# 영화 추천 함수
def recommend_movies(user_id, movie_user_matrix, movie_similarity, n=5):
    user_ratings = movie_user_matrix[user_id]
    similar_scores = movie_similarity.dot(user_ratings)
    scores = [(movie_id, score) for movie_id, score in enumerate(similar_scores)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = [movie_id for movie_id, score in scores if movie_id not in user_ratings.index][:n]
    return recommendations

# 예제 사용자에 대한 영화 추천
user_id = 1
recommendations = recommend_movies(user_id, movie_user_matrix, movie_similarity)
print(f"User {user_id} 추천 영화: {recommendations}")
```

이 코드는 간단한 영화 추천 시스템의 기본적인 구조를 제공합니다. 실제 추천 시스템의 정확도를 높이기 위해서는 추가적인 알고리즘과 모델을 적용할 수 있습니다.

## 마무리
이번 글에서는 Python을 사용하여 간단한 영화 추천 시스템을 만드는 방법을 소개했습니다. 추천 시스템은 사용자의 취향을 반영하여 개인화된 추천을 제공하는 중요한 기술입니다. 다음 포스트에서는 Magenta를 이용하여 AI 음악을 생성해 보도록 하겠습니다. 다음 글도 기대해주세요~.