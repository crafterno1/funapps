---
title: "Python과 OpenCV를 이용한 실시간 객체 추적"
keywords: ["Python", "OpenCV", "객체 추적", "컴퓨터 비전", "실시간"]
date: "2024-07-29T12:59:01+12:00"
cover:
  image: "/images/ai15.webp"
  alt: "AI Deep Learning Model"
---

# Python과 OpenCV를 이용한 실시간 객체 추적

객체 추적은 컴퓨터 비전 분야에서 매우 중요한 기술 중 하나입니다. 이번 글에서는 Python과 OpenCV를 사용하여 실시간 객체 추적을 구현하는 방법을 소개하겠습니다. 이번 포스트는 초보자분들도 쉽게 따라하실 수 있도록 구성하였으니 천천히 따라해 보세요~

## 준비 작업

### Python과 OpenCV 설치하기

우선 Python과 OpenCV 라이브러리를 설치해야 합니다. 다음 명령어를 사용하여 필요한 패키지를 설치할 수 있습니다:

```bash
pip install numpy opencv-python
```

## 객체 추적을 위한 동영상 스트림 설정

객체 추적을 위해 웹캠을 사용하여 실시간 동영상 스트림을 설정합니다:

```python
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
```

## 객체 추적기 초기화

OpenCV에서는 다양한 객체 추적 알고리즘을 제공합니다. 여기서는 CSRT 추적기를 사용하겠습니다:

```python
tracker = cv2.TrackerCSRT_create()
```

## 객체 선택 및 추적 시작

사용자가 추적할 객체를 선택하고 추적을 시작합니다:

```python
ret, frame = cap.read()
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)
```

## 실시간 객체 추적

실시간으로 동영상을 읽어와서 객체를 추적합니다:

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    success, bbox = tracker.update(frame)
    
    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 전체 코드 예제

아래는 위의 모든 단계를 포함한 전체 코드 예제입니다:

```python
import cv2

# 웹캠 동영상 스트림 설정
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# 객체 추적기 초기화
tracker = cv2.TrackerCSRT_create()

# 첫 프레임 읽기
ret, frame = cap.read()
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

# 실시간 객체 추적
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    success, bbox = tracker.update(frame)
    
    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

이 코드는 간단한 실시간 객체 추적의 기본적인 구조를 제공합니다. 실제 기능의 성능을 높이기 위해서는 다양한 데이터 전처리 기법과 모델을 적용할 수 있습니다.

## 마무리

이번 글에서는 Python과 OpenCV를 사용하여 간단한 실시간 객체 추적을 구현하는 방법을 소개했습니다. 객체 추적은 다양한 애플리케이션에서 유용하게 사용될 수 있습니다. 다음 포스트에서는 강화학습으로 게임 캐릭터의 행동 패턴을 만들어 보도록 하겠습니다.