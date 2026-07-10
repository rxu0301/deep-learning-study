# 빅데이터 분석 - 학습 상세 기록

> 2025년 3월 ~ 6월, 머신러닝 알고리즘과 데이터 분석의 상세 학습 내용입니다.

## 목차
- [기초 학습 (3월)](#기초-학습-3월)
- [머신러닝 알고리즘 (4월-5월)](#머신러닝-알고리즘-4월-5월)
- [실전 프로젝트 (5월-6월)](#실전-프로젝트-5월-6월)
- [고급 분석 기법](#고급-분석-기법)

---

## 기초 학습 (3월)

### 20250312numpy류현정.ipynb - NumPy 심화

**학습 내용:**
- 행렬 연산 심화
- 선형대수 함수
- 통계 함수 심화
- 성능 최적화

**주요 코드 예시:**
```python
import numpy as np

# 행렬 연산
A = np.random.randn(3, 3)
B = np.random.randn(3, 3)
C = np.dot(A, B)           # 행렬 곱셈
D = np.linalg.inv(A)       # 역행렬
det = np.linalg.det(A)     # 행렬식
eig_vals, eig_vecs = np.linalg.eig(A)  # 고유값, 고유벡터

# 통계 함수
mean = np.mean(A)
std = np.std(A)
percentile = np.percentile(A, 75)

# 조건부 연산
result = np.where(A > 0, 'positive', 'negative')
```

**학습 포인트:**
- 행렬식과 고유값의 의미
- 통계 함수를 통한 데이터 분석
- NumPy의 성능 최적화

---

### 20250319-pandas 2308류현정.ipynb - Pandas 심화

**학습 내용:**
- 고급 인덱싱 (MultiIndex)
- 데이터 변환 함수
- 결측치 처리 전략
- 그룹화 및 집계

**주요 코드 예시:**
```python
import pandas as pd

# 데이터 로드 (예: CSV)
df = pd.read_csv('data.csv')

# 기본 탐색
df.head()
df.info()
df.describe()

# 결측치 처리
df.isnull().sum()           # 결측치 개수
df = df.dropna()            # 결측치 행 제거
df = df.fillna(df.mean())   # 평균으로 채우기

# 그룹화 및 집계
grouped = df.groupby('category')
grouped['value'].agg(['mean', 'sum', 'count'])

# 데이터 변환
df['new_col'] = df['col1'] + df['col2']
df['category'] = pd.cut(df['value'], bins=3)
```

**학습 포인트:**
- 대용량 데이터 처리
- 효율적인 그룹화 및 집계
- 결측치 처리 전략

---

### 20250409_데이터 인덱싱.ipynb - 인덱싱 심화

**학습 내용:**
- loc, iloc 구분
- Boolean 인덱싱 심화
- 인덱싱 성능 최적화

**주요 코드 예시:**
```python
# Label-based 인덱싱
df.loc[df['age'] > 25]
df.loc[0:5, ['name', 'age']]

# Position-based 인덱싱
df.iloc[0:5]
df.iloc[0:5, 0:2]

# Boolean 인덱싱 (복합 조건)
mask = (df['age'] > 25) & (df['city'] == 'Seoul')
df[mask]

# at, iat를 이용한 빠른 접근
value = df.at[0, 'name']
value = df.iat[0, 0]
```

**학습 포인트:**
- loc vs iloc 구분
- Boolean 마스킹의 효율성
- 대용량 데이터의 빠른 접근

---

### 20250501 필터링 결측치 2308류현정.ipynb

**학습 내용:**
- 고급 필터링 기법
- 결측치 처리 전략
- 이상치 탐지

**주요 내용:**
```python
# 고급 필터링
# 다중 조건 필터링
filtered = df[(df['age'] > 20) & (df['salary'] > 30000)]

# isin을 이용한 범주 필터링
cities = ['Seoul', 'Busan', 'Incheon']
df_filtered = df[df['city'].isin(cities)]

# 결측치 패턴 분석
missing_patterns = df.isnull().sum(axis=0)  # 열별 결측치
missing_patterns = df.isnull().sum(axis=1)  # 행별 결측치

# 이상치 탐지
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['value'] < Q1 - 1.5*IQR) | (df['value'] > Q3 + 1.5*IQR)]
```

**학습 포인트:**
- 복잡한 조건 필터링
- 결측치 분석 및 처리
- 이상치 탐지 방법

---

## 머신러닝 알고리즘 (4월-5월)

### 선형 회귀 (Linear Regression)

**개념:**
- 연속값 예측 문제
- y = mx + b 선형 관계 학습
- 손실 함수: Mean Squared Error (MSE)

**구현 예시:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 계수 확인
print(f"Coefficient: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

**적용 사례:**
- 집값 예측
- 판매량 예측
- 기온 예측

**성능 평가:**
- R² Score: 0 ~ 1 (높을수록 좋음)
- RMSE: 낮을수록 좋음

---

### 로지스틱 회귀 (Logistic Regression)

**개념:**
- 이진/다중 분류 문제
- Sigmoid 함수를 이용한 확률 변환
- 손실 함수: Binary/Categorical Crossentropy

**구현 예시:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 이진 분류
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # 확률값

# 평가
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

**주요 파라미터:**
- C: 정규화 강도 (기본값: 1.0, 작을수록 강한 정규화)
- solver: 최적화 알고리즘 ('lbfgs', 'liblinear', 'saga')
- max_iter: 최대 반복 횟수

**적용 사례:**
- 이메일 스팸 분류
- 질병 진단
- 고객 이탈 예측

---

### K-Nearest Neighbors (KNN)

**개념:**
- 가장 가까운 k개의 이웃을 보고 예측
- 거리 기반 알고리즘 (Euclidean, Manhattan)
- 인스턴스 기반 학습 (게으른 학습)

**구현 예시:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 데이터 정규화 (필수!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 생성 및 학습
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# 예측
y_pred = knn.predict(X_test_scaled)
accuracy = knn.score(X_test_scaled, y_test)
```

**주요 파라미터:**
- n_neighbors: k값 (기본값: 5)
- weights: 거리 가중치 ('uniform' 또는 'distance')
- metric: 거리 척도 ('euclidean', 'manhattan' 등)

**장점:**
- 구현이 간단
- 해석이 용이

**단점:**
- 거리 계산 비용 높음
- 차원의 저주 (고차원 데이터에서 성능 저하)

---

### 나이브 베이즈 (Naive Bayes)

**개념:**
- 베이즈 정리 기반 확률 분류기
- 특성 간 독립성 가정 (Naive)
- 빠른 학습 및 예측

**구현 예시:**
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score

# 가우시안 나이브 베이즈 (연속형 데이터)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 다항식 나이브 베이즈 (범주형 데이터)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

**주요 파라미터:**
- GaussianNB: 연속형 데이터용
- MultinomialNB: 문서 분류, 텍스트 분석용
- BernoulliNB: 이진 데이터용

**적용 사례:**
- 스팸 필터
- 텍스트 분류
- 감정 분석

---

### 서포트 벡터 머신 (Support Vector Machine, SVM)

**개념:**
- 마진을 최대화하는 초평면으로 분류
- 비선형 분류를 위한 커널 트릭
- 이진 분류 및 다중 분류 지원

**구현 예시:**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 데이터 정규화 (중요!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 선형 SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)

# RBF 커널 SVM
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_rbf.fit(X_train_scaled, y_train)

# Polynomial 커널 SVM
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = svm_linear.predict(X_test_scaled)
accuracy = svm_linear.score(X_test_scaled, y_test)
```

**주요 파라미터:**
- kernel: 'linear', 'rbf', 'poly' 등
- C: 정규화 파라미터 (작을수록 강한 정규화)
- gamma: RBF 커널 계수
- degree: Polynomial 커널 차수

**장점:**
- 고차원 데이터에 강함
- 작은 샘플 크기에서도 좋은 성능

**단점:**
- 대용량 데이터에서는 느림
- 파라미터 튜닝 필요

---

### 랜덤 포레스트 (Random Forest)

**개념:**
- 여러 결정 트리의 앙상블
- 배깅과 특성 부분 샘플링 활용
- 회귀 및 분류 모두 지원

**구현 예시:**
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, feature_importances_

# 분류
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# 회귀
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

# 특성 중요도
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# 평가
accuracy = accuracy_score(y_test, y_pred)
```

**주요 파라미터:**
- n_estimators: 트리 개수 (기본값: 100)
- max_depth: 트리 최대 깊이
- min_samples_split: 노드 분할 최소 샘플 수
- min_samples_leaf: 리프 노드 최소 샘플 수
- max_features: 각 분할에 고려할 특성 수

**장점:**
- 과적합 방지
- 특성 중요도 제공
- 높은 정확도

**단점:**
- 모델 해석 어려움
- 학습 시간 오래 걸림

---

### K-Means 군집화

**개념:**
- 비지도 학습 (레이블 없음)
- k개의 클러스터로 데이터 분할
- 반복적 알고리즘

**구현 예시:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 최적 k값 찾기 (Elbow Method)
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# 최적 k로 모델 학습
optimal_k = 3  # 시각화로 결정
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# 결과 분석
print(f"Inertia: {kmeans.inertia_}")
print(f"Silhouette Score: {silhouette_score(X, cluster_labels)}")
```

**주요 파라미터:**
- n_clusters: 클러스터 개수
- init: 초기 중심점 선택 ('k-means++' 추천)
- n_init: 서로 다른 초기화로 실행 횟수
- max_iter: 최대 반복 횟수

**최적 k 결정:**
- Elbow Method: 관성(inertia)이 급격히 감소하는 지점
- Silhouette Score: -1 ~ 1 범위, 1에 가까울수록 좋음

---

### 주성분 분석 (PCA)

**개념:**
- 고차원 데이터를 저차원으로 축소
- 분산을 최대화하는 방향 찾기
- 비지도 학습 (특성 학습)

**구현 예시:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 데이터 정규화 (필수!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 모델 (전체 컴포넌트)
pca_full = PCA()
pca_full.fit(X_scaled)

# 누적 설명 분산 시각화
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
print(f"95% 분산 설명에 필요한 컴포넌트: {np.argmax(cumsum >= 0.95) + 1}")

# 최적 컴포넌트 개수로 PCA
pca = PCA(n_components=0.95)  # 95% 분산 설명
X_pca = pca.fit_transform(X_scaled)

# 결과 분석
print(f"Original shape: {X_scaled.shape}")
print(f"Reduced shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Explained variance: {pca.explained_variance_}")
```

**주요 파라미터:**
- n_components: 축소할 차원 수 (정수 또는 분산 비율)
- svd_solver: 계산 방식 ('auto', 'full', 'arpack', 'randomized')

**활용:**
- 데이터 시각화 (2D, 3D)
- 차원 축소를 통한 계산 효율성 향상
- 노이즈 제거

---

## 실전 프로젝트 (5월-6월)

### Titanic 생존자 예측 프로젝트

**데이터:**
- 훈련 데이터: 891개 샘플
- 테스트 데이터: 418개 샘플
- 특성: 11개 (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked 등)

**데이터 전처리:**
```python
import pandas as pd
import numpy as np

# 데이터 로드
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')

# 결측치 확인
print(train_df.isnull().sum())

# Age 결측치 처리 (평균 또는 중앙값)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# Embarked 결측치 처리 (최빈값)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# 범주형 변수 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])  # male:1, female:0
train_df['Embarked'] = le.fit_transform(train_df['Embarked'])

# 불필요한 열 제거
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_df['Survived']
```

**모델 학습 및 비교:**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 1: 로지스틱 회귀
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")

# 모델 2: 랜덤 포레스트
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # 정규화 불필요
rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

# 모델 3: SVM
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}")

# 상세 평가
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, rf_pred)}")
print(classification_report(y_test, rf_pred))
```

**결과:**
- 로지스틱 회귀: ~78% 정확도
- 랜덤 포레스트: ~82% 정확도
- SVM: ~80% 정확도

**특성 중요도 분석:**
```python
importances = rf.feature_importances_
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

for name, importance in sorted(zip(feature_names, importances), 
                               key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")
    
# 결과:
# Sex: 0.3345
# Fare: 0.2891
# Age: 0.2345
# Pclass: 0.1234
# ...
```

---

### 서울시 CCTV 대비 범죄율 분석

**데이터:**
- 서울시 25개 구
- CCTV 설치 수
- 인구수
- 범죄율

**분석:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 데이터 로드
df = pd.read_csv('seoul_data.csv')

# 기본 통계
print(df.describe())

# 상관계수 계산
correlation = df['CCTV수'].corr(df['범죄율'])
r_value, p_value = pearsonr(df['CCTV수'], df['범죄율'])
print(f"Pearson Correlation: {r_value:.4f} (p-value: {p_value:.4f})")

# 산점도 + 회귀선
plt.figure(figsize=(10, 6))
sns.regplot(x='CCTV수', y='범죄율', data=df, scatter_kws={'s': 100}, 
            line_kws={'color': 'red'})
plt.title('CCTV 수 vs 범죄율')
plt.xlabel('CCTV 설치 수')
plt.ylabel('범죄율')
plt.show()

# 상관 행렬 히트맵
correlation_matrix = df[['CCTV수', '인구수', '범죄율']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()
```

**주요 발견:**
- CCTV 수와 범죄율 간 상관계수 분석
- 지역별 특성 파악
- 정책 시사점 도출

---

### Kakao 치킨점 분석

**분석 내용:**
- 지역별 가게 밀도 분석
- K-Means 클러스터링을 통한 지역 그룹핑
- 특성별 지역 분류

**코드 예시:**
```python
# K-Means 클러스터링
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 데이터 준비
X = df[['인구수', '평균_가게_수']].values

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# 결과 시각화
plt.scatter(df['인구수'], df['평균_가게_수'], c=cluster_labels, cmap='viridis', s=100)
plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],
            scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
            marker='X', s=300, c='red', edgecolors='black', linewidths=2)
plt.xlabel('인구수')
plt.ylabel('평균 가게 수')
plt.title('K-Means 클러스터링 결과')
plt.show()
```

---

## 고급 분석 기법

### 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

# GridSearchCV를 이용한 체계적 탐색
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
```

### 교차 검증

```python
from sklearn.model_selection import cross_val_score

# k-fold 교차 검증
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

**Last Updated**: 2025년 6월  
**총 학습 시간**: 약 120시간  
**주요 머신러닝 알고리즘**: 8개  
**실전 프로젝트**: 3개
