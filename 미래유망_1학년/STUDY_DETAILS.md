# 미래유망 1학년 - 학습 상세 기록

> 2024년 9월 ~ 11월, 초급 AI 교육 과정의 상세 학습 내용입니다.

## 목차
- [NumPy 모듈](#numpy-모듈)
- [Pandas 모듈](#pandas-모듈)
- [Matplotlib 모듈](#matplotlib-모듈)
- [OpenCV 모듈](#opencv-모듈)
- [딥러닝 모듈](#딥러닝-모듈)

---

## NumPy 모듈

### NumPy_01.ipynb - 기본 배열 연산

**학습 내용:**
- 배열 생성: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`
- 배열 속성: shape, dtype, ndim, size
- 인덱싱 및 슬라이싱
- 기본 산술 연산

**주요 코드 예시:**
```python
# 배열 생성
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
range_arr = np.arange(0, 10, 2)
linspace_arr = np.linspace(0, 1, 11)

# 배열 인덱싱
arr[0]          # 첫 번째 요소
arr[1:3]        # 1번부터 2번까지
arr[-1]         # 마지막 요소

# 기본 연산
arr + 1         # 모든 요소에 1 추가 (브로드캐스팅)
arr * 2         # 모든 요소에 2 곱하기
```

**학습 포인트:**
- NumPy 배열은 Python 리스트보다 훨씬 빠르고 효율적
- 브로드캐스팅 개념 이해
- 배열의 기본 속성 파악

---

### NumPy_02.ipynb - 고급 연산

**학습 내용:**
- 다차원 배열 다루기
- 배열 재형성 (reshape)
- 배열 결합 (concatenate, stack)
- 통계 함수 (mean, std, sum, max, min)
- 선형대수 (행렬 연산, 전치, 역행렬)
- 난수 생성

**주요 코드 예시:**
```python
# 다차원 배열 생성
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])

# 배열 재형성
reshaped = matrix.reshape(9)      # 1D로 변환
reshaped2 = matrix.reshape(1, 9)  # 1x9로 변환

# 통계 함수
np.mean(matrix)                    # 평균: 5.0
np.std(matrix)                     # 표준편차
np.sum(matrix)                     # 합계: 45
np.max(matrix)                     # 최대값: 9
np.min(matrix)                     # 최소값: 1

# 선형대수
np.dot(matrix, matrix)             # 행렬 곱셈
matrix.T                           # 전치
np.linalg.inv(matrix)              # 역행렬

# 난수 생성
random_arr = np.random.rand(3, 3)  # [0, 1) 균등 분포
normal_arr = np.random.randn(5)    # 표준 정규 분포
random_int = np.random.randint(0, 10, 5)  # 0-9 정수
```

**학습 포인트:**
- 다차원 데이터 조작의 중요성
- 통계 함수를 통한 데이터 분석
- 선형대수 연산의 필요성 (특히 딥러닝에서)
- 난수 생성을 통한 데이터 시뮬레이션

---

### numpy_정리.ipynb - NumPy 요약 및 정리

**학습 내용:**
- NumPy의 핵심 개념 복습
- 자주 사용되는 함수 정리
- 실무에서의 활용 팁

**주요 내용:**
- 배열 생성 방법 비교
- 연산 성능 (NumPy vs Python List)
- 유용한 함수 목록
- 일반적인 실수 및 디버깅

---

## Pandas 모듈

### pandas.ipynb - 기본 데이터프레임

**학습 내용:**
- Series 개념 및 생성
- DataFrame 개념 및 생성
- 데이터프레임 인덱싱
- 열(column) 추가/제거
- 기본 통계

**주요 코드 예시:**
```python
import pandas as pd

# Series 생성
s = pd.Series([1, 3, 5, np.nan, 6, 8])  # 인덱스가 자동으로 0부터 생성
s_named = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# DataFrame 생성
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Incheon']
})

# 기본 정보
df.head()          # 처음 5개 행 보기
df.tail()          # 마지막 5개 행 보기
df.info()          # 데이터프레임 정보
df.describe()      # 통계 요약

# 열 접근
df['name']         # 열 선택
df[['name', 'age']]  # 여러 열 선택

# 행 선택
df.loc[0]          # 라벨 기반 선택
df.iloc[0]         # 위치 기반 선택
```

**학습 포인트:**
- Series는 1차원 배열, DataFrame은 2차원 테이블
- 라벨(label) 기반 인덱싱의 편의성
- DataFrame의 기본 구조와 메타데이터

---

### pandas2.ipynb - 데이터 조작

**학습 내용:**
- 데이터프레임 병합 (merge, concat)
- 데이터 정렬 (sort_values)
- 데이터 필터링 (조건식)
- 열 연산 및 생성
- 결측치 처리 기초 (dropna, fillna)

**주요 코드 예시:**
```python
# 데이터 필터링
df[df['age'] > 25]           # 나이가 25 이상인 행
df[df['city'] == 'Seoul']    # Seoul에 사는 행
df[(df['age'] > 25) & (df['city'] == 'Seoul')]  # 조건 결합

# 데이터 정렬
df.sort_values('age')        # 나이로 오름차순 정렬
df.sort_values('age', ascending=False)  # 내림차순

# 열 연산
df['age_group'] = df['age'] // 10 * 10  # 새 열 생성

# 병합
df1 = pd.DataFrame({'key': ['A', 'B'], 'val1': [1, 2]})
df2 = pd.DataFrame({'key': ['A', 'B'], 'val2': [3, 4]})
merged = pd.merge(df1, df2, on='key')

# 결측치 처리
df.dropna()                  # 결측치가 있는 행 제거
df.fillna(0)                 # 결측치를 0으로 채우기
```

**학습 포인트:**
- Boolean 인덱싱을 통한 조건부 필터링
- 데이터 정렬과 검색의 중요성
- 데이터 결합의 여러 방식
- 결측치 처리의 기초

---

### pandas_3.ipynb - 심화 내용

**학습 내용:**
- 그룹화 및 집계 (groupby)
- 피벗 테이블 (pivot_table)
- 데이터 변환 함수 (apply, map)
- 시간 데이터 처리
- 데이터 저장 및 로드

**주요 코드 예시:**
```python
# 그룹화 및 집계
grouped = df.groupby('city')
grouped.mean()               # 도시별 평균
grouped['age'].sum()         # 도시별 나이 합계

# Apply 함수
df['age'].apply(lambda x: 'Young' if x < 30 else 'Old')

# 데이터 저장 및 로드
df.to_csv('data.csv', index=False)
df_loaded = pd.read_csv('data.csv')
df.to_excel('data.xlsx', index=False)
df_loaded = pd.read_excel('data.xlsx')
```

**학습 포인트:**
- 데이터 집계를 통한 인사이트 도출
- Lambda 함수를 통한 데이터 변환
- 다양한 파일 형식 지원
- 대용량 데이터 처리 기초

---

### Pandas_정리.ipynb - Pandas 요약

**학습 내용:**
- Pandas의 핵심 개념 복습
- 자주 사용되는 메서드 정리
- 데이터 분석 워크플로우

---

## Matplotlib 모듈

### Matplotlib_01.ipynb - 기본 플롯

**학습 내용:**
- Figure와 Axes 객체
- 라인 플롯 (line plot)
- 산점도 (scatter plot)
- 히스토그램 (histogram)
- 기본 스타일링

**주요 코드 예시:**
```python
import matplotlib.pyplot as plt

# Figure와 Axes 생성
fig, ax = plt.subplots(figsize=(10, 6))

# 라인 플롯
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
ax.plot(x, y, label='Line 1', color='blue', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Simple Line Plot')
ax.legend()
plt.show()

# 산점도
ax.scatter(x, y, s=100, color='red', alpha=0.6)

# 히스토그램
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5]
ax.hist(data, bins=5, color='green', edgecolor='black')
```

**학습 포인트:**
- OOP 방식의 Matplotlib 사용 (fig, ax)
- 기본 플롯 타입별 특징
- 레이블과 범례의 중요성

---

### Matplotlib_02.ipynb - 고급 시각화

**학습 내용:**
- 서브플롯 생성 (subplot)
- 박스플롯 (boxplot)
- 히트맵 (heatmap)
- 색상 맵 (colormap)
- 스타일 지정

**주요 코드 예시:**
```python
# 서브플롯
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].hist(data)
axes[1, 1].bar(x, y)

# 박스플롯
data_list = [np.random.randn(100) for _ in range(4)]
ax.boxplot(data_list, labels=['A', 'B', 'C', 'D'])

# 히트맵 (Seaborn 사용)
import seaborn as sns
sns.heatmap(data_matrix, cmap='viridis', annot=True)
```

**학습 포인트:**
- 복잡한 시각화를 위한 다중 플롯
- 데이터 분포 시각화
- 색상을 통한 정보 표현

---

### Matplotlib_정리.ipynb - Matplotlib 요약

**학습 내용:**
- Matplotlib의 핵심 개념
- 차트 선택 가이드
- 효과적인 시각화 팁

---

## OpenCV 모듈

### openCV_01_grayscale.ipynb - 흑백 이미지 처리

**학습 내용:**
- 이미지 읽기 및 표시
- 흑백 변환 (Grayscale)
- 이미지 속성 (크기, 채널)
- 이미지 저장

**주요 코드 예시:**
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 읽기
img = cv2.imread('image.jpg')  # BGR 형식

# 흑백 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지 속성
print(img.shape)   # (height, width, channels)
print(gray.shape)  # (height, width)

# 이미지 표시
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
plt.show()

# 이미지 저장
cv2.imwrite('gray_image.jpg', gray)
```

**학습 포인트:**
- OpenCV의 BGR 색상 순서 (RGB와 다름)
- 채널별 이미지 표현
- 이미지 파일 입출력
- matplotlib과의 연동

---

### openCV_2_color.ipynb - 색상 이미지 처리

**학습 내용:**
- RGB/BGR 색공간
- HSV 색공간
- 색상 범위 필터링
- 채널 분리 및 결합

**주요 코드 예시:**
```python
# BGR to RGB 변환 (matplotlib에서 표시용)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR to HSV 변환
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV를 이용한 색상 범위 필터링
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(img_hsv, lower_red, upper_red)
result = cv2.bitwise_and(img, img, mask=mask)

# 채널 분리
b, g, r = cv2.split(img)

# 채널 결합
merged = cv2.merge([b, g, r])
```

**학습 포인트:**
- 다양한 색공간의 특징과 용도
- HSV를 이용한 효과적인 색상 추출
- 마스킹을 통한 이미지 처리
- 채널 조작의 중요성

---

## 딥러닝 모듈

### Deep_Learning_01_손글씨 분류.ipynb

**문제 정의:**
- MNIST 데이터셋 사용 (28×28 흑백 이미지, 10개 클래스)
- 손글씨 숫자 0~9 분류

**모델 구조:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# 데이터 로드
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 정규화 (0-255 → 0-1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),      # 784차원 벡터로 변환
    Dense(128, activation='relu'),      # 숨겨진 층 1
    Dense(64, activation='relu'),       # 숨겨진 층 2
    Dense(10, activation='softmax')     # 출력층 (10개 클래스)
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

**결과:**
- Test Accuracy: ~97%
- 학습 시간: ~1분 (GPU 환경)

**학습 포인트:**
- 신경망의 기본 구조 이해
- Flatten 레이어의 역할
- Softmax 활성화 함수와 다중 분류
- 모델 컴파일 및 학습

---

### Deep_Learning_02_시험 점수 예측.ipynb

**문제 정의:**
- 회귀 문제: 공부 시간으로부터 시험 점수 예측
- 선형 관계 학습

**모델 구조:**
```python
# 간단한 회귀 데이터 생성
X_train = np.random.rand(100, 1) * 10  # 공부 시간 (0-10시간)
y_train = X_train * 10 + 5 + np.random.randn(100, 1) * 5  # 점수

# 모델 구성 (회귀용)
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(1)  # 출력층 (활성화 함수 없음 - 연속값)
])

# 회귀용 컴파일
model.compile(optimizer='adam', loss='mse')  # MSE: Mean Squared Error

# 학습
model.fit(X_train, y_train, epochs=100)

# 예측
prediction = model.predict([[5]])  # 5시간 공부할 때 점수
```

**학습 포인트:**
- 회귀 vs 분류
- 연속값 예측
- MSE 손실 함수
- 활성화 함수 없는 출력층

---

### Deep_Learning_03_이진분류.ipynb

**문제 정의:**
- 이진 분류 입문
- 두 개 클래스로 구분하는 문제

**모델 구조:**
```python
# 이진 분류 데이터
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 모델 구성
model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')  # Sigmoid: 0~1 사이의 확률
])

# 이진 분류용 컴파일
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # 이진 분류 손실
    metrics=['accuracy']
)

model.fit(X, y, epochs=50)
```

**학습 포인트:**
- 이진 분류의 특징
- Sigmoid 활성화 함수
- Binary Crossentropy 손실 함수
- 확률 기반 예측

---

### Deeplearning_손글씨_CNN.ipynb

**내용:**
- CNN(합성곱 신경망) 기초 소개
- Conv2D 레이어 개념
- MNIST에 CNN 적용

**모델 구조:**
```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

# CNN 모델
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=10)
```

**학습 포인트:**
- Conv2D의 필터와 특징맵
- MaxPooling의 역할
- CNN이 이미지 처리에 적합한 이유

---

### numpy_random_Deep_Laerning.ipynb

**내용:**
- 난수 생성과 딥러닝의 관계
- 가중치 초기화
- 데이터 증강과 난수

**학습 포인트:**
- 신경망 가중치의 난수 초기화
- 배치 정렬화를 위한 난수
- 데이터 증강에서의 난수 사용

---

### pooling_01.ipynb

**내용:**
- Pooling 연산의 개념과 종류
- MaxPooling과 AveragePooling
- Pooling의 효과

**학습 포인트:**
- 특징맵 축소
- 계산 효율성 향상
- 노이즈 감소

---

### 과제1.ipynb, 용어_EX.ipynb, 실습1.ipynb

**내용:**
- 학습 내용의 종합 복습
- AI 관련 용어 정리
- 실전 실습 문제

---

## 학습 시간 관리

| 모듈 | 학습 기간 | 중점 사항 |
|------|---------|---------|
| NumPy | 1주 | 배열 연산 기초 |
| Pandas | 1주 | 데이터 조작 및 분석 |
| Matplotlib | 4일 | 기본 시각화 |
| OpenCV | 3일 | 이미지 처리 입문 |
| 딥러닝 | 2주 | 신경망 구조 및 학습 |

---

## 주요 성과

✅ Python 데이터 분석 라이브러리 기초 습득  
✅ NumPy와 Pandas를 활용한 데이터 처리  
✅ Matplotlib을 통한 데이터 시각화  
✅ OpenCV를 통한 기본 이미지 처리  
✅ 신경망 기초 이해 및 MNIST 분류 (97% 정확도)

---

**Last Updated**: 2024년 11월  
**총 학습 시간**: 약 40-50시간  
**주요 실습 프로젝트**: 5개
