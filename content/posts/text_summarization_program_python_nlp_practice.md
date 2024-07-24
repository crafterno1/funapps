---
title: "Python으로 텍스트 요약 프로그램 만들기: 자연어 처리 실습"
keywords: ["Python", "텍스트 요약", "자연어 처리", "NLTK", "기계 학습", "텍스트 분석"]
date: "2024-07-24T22:49:41+12:00"
cover:
  image: "/images/ai9.webp"
  alt: "AI Deep Learning Model"
---

자연어 처리(NLP, Natural Language Processing)는 텍스트 데이터를 이해하고 처리하는 데 중점을 둔 인공지능의 한 분야입니다. 이번 글에서는 Python과 NLTK를 사용하여 텍스트 요약 프로그램을 만드는 방법을 소개해 드리도록 하겠습니다. 이 프로그램은 긴 텍스트를 요약하여 핵심 내용을 추출하는 데 유용하니 조금 어렵더라도 천천히 따라해 보시면서 꼭 이해해 하시길 바래요.

## 준비 작업

### Python과 NLTK 설치하기

우선 Python과 NLTK(Natural Language Toolkit)를 설치해야 합니다. NLTK는 파이썬을 위한 강력한 자연어 처리 라이브러리입니다. 다음 명령어를 사용하여 필요한 패키지를 설치할 수 있습니다:

```bash
pip install nltk
```

설치가 완료되면, 필요한 NLTK 데이터를 다운로드합니다:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 텍스트 전처리

텍스트 요약의 첫 단계는 텍스트 데이터를 전처리하는 것입니다. 여기에는 문장 분할, 불용어 제거, 단어 토큰화 등이 포함됩니다. 다음 코드를 사용하여 텍스트를 전처리할 수 있습니다:

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words
```

## 중요 문장 추출

이제 텍스트에서 중요한 문장을 추출하는 단계를 진행합니다. 이를 위해 각 문장의 점수를 계산하고, 높은 점수를 받은 문장을 요약에 포함시킵니다:

```python
def sentence_score(sentences, words):
    scores = {}
    for sentence in sentences:
        for word in words:
            if word in sentence.lower():
                if sentence not in scores:
                    scores[sentence] = 1
                else:
                    scores[sentence] += 1
    return scores
```

## 텍스트 요약 생성

마지막으로, 점수가 높은 문장들을 결합하여 최종 요약을 생성합니다:

```python
def summarize_text(text, n):
    sentences = sent_tokenize(text)
    words = preprocess_text(text)
    scores = sentence_score(sentences, words)
    ranked_sentences = sorted(scores, key=scores.get, reverse=True)
    summary = " ".join(ranked_sentences[:n])
    return summary
```

## 전체 코드 예제

아래는 위의 모든 단계를 포함한 전체 코드 예제입니다:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words

def sentence_score(sentences, words):
    scores = {}
    for sentence in sentences:
        for word in words:
            if word in sentence.lower():
                if sentence not in scores:
                    scores[sentence] = 1
                else:
                    scores[sentence] += 1
    return scores

def summarize_text(text, n):
    sentences = sent_tokenize(text)
    words = preprocess_text(text)
    scores = sentence_score(sentences, words)
    ranked_sentences = sorted(scores, key=scores.get, reverse=True)
    summary = " ".join(ranked_sentences[:n])
    return summary

# 예제 텍스트
text = """
Your long text goes here.
"""

# 요약 문장 수
summary = summarize_text(text, 3)
print(summary)
```

이 코드는 간단한 텍스트 요약 프로그램의 기본적인 구조를 제공합니다. 실제 텍스트 요약의 정확도를 높이기 위해서는 추가적인 알고리즘과 모델을 적용할 수 있습니다.

## 마무리

이번 글에서는 Python과 NLTK를 사용하여 텍스트 요약 프로그램을 만드는 방법을 소개했습니다. 자연어 처리 기술을 활용하여 텍스트 데이터를 효율적으로 요약하고 분석할 수 있습니다. 다음 포스트에서는 조금 더 재밌는 내용인 Python으로 영화 추천 시스템을 만들어 보도록 하겠습니다.