# Colab Notebooks - 학습 상세 기록

> 2025년 8월 ~ 12월, 머신러닝 알고리즘의 실전 적용 및 데이터 분석 프로젝트 상세 기록입니다.

## 목차
- [8월: 알고리즘 복습](#8월-알고리즘-복습)
- [9월: 인기도 예측 초기](#9월-인기도-예측-초기)
- [10월: 인기도 예측 심화](#10월-인기도-예측-심화)
- [11월-12월: 종합 프로젝트](#11월-12월-종합-프로젝트)

---

## 8월: 알고리즘 복습

### 데이터시각화(20250822).ipynb

**학습 내용:**
- 고급 Matplotlib 시각화
- Seaborn을 통한 통계 시각화
- 다중 변수 시각화

**주요 차트:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 히트맵 (Correlation Matrix)
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)

# 2. 분포도 (Distribution Plot)
sns.distplot(df['column'], kde=True)

# 3. 박스플롯 (Boxplot)
sns.boxplot(x='category', y='value', data=df)

# 4. 바이올린 플롯 (Violin Plot)
sns.violinplot(x='category', y='value', data=df)

# 5. 조건부 산점도 (Conditional Scatter)
sns.lmplot(x='x_col', y='y_col', hue='category', data=df)
```

**학습 포인트:**
- 다변수 관계 시각화
- 분포 및 통계 특성 파악
- 범주별 비교 시각화

---

### 선형회귀 관련 파일들

#### 선형회귀2_류현정 / 선형회귀3_류현정.ipynb / 선형회귀4_류현정

**프로그레션:**

**선형회귀 2**: 기본 개념
- 단순 선형 회귀
- 데이터 전처리
- 모델 평가 (R², MSE)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
```

**선형회귀 3**: 심화 분석
- 다중 선형 회귀
- 특성 중요도
- 잔차 분석

```python
# 특성 중요도 확인
coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_
}).sort_values('coefficient', ascending=False)

# 잔차 분석
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
```

**선형회귀 4**: 실전 프로젝트
- 실제 데이터셋 적용
- 모델 개선 및 최적화
- 결과 해석

---

### 로지스틱회귀1_류현정

**학습 내용:**
- 이진/다중 분류
- 확률 기반 예측
- 분류 모델 평가

**코드 예시:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**평가 메트릭:**
- Accuracy: 전체 맞은 비율
- Precision: 양성 예측 중 실제 양성 비율
- Recall: 실제 양성 중 예측한 양성 비율
- F1-Score: Precision과 Recall의 조화 평균

---

### SVM(Support Vector Machine).ipynb

**학습 내용:**
- 선형/비선형 분류
- 커널 트릭
- 정규화 파라미터 (C)

**코드 예시:**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 데이터 정규화 (필수)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 선형 SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)

# RBF 커널 SVM
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_rbf.fit(X_train_scaled, y_train)

# 예측
y_pred = svm_rbf.predict(X_test_scaled)
accuracy = svm_rbf.score(X_test_scaled, y_test)
```

**커널 선택:**
- 'linear': 선형 분리 가능한 데이터
- 'rbf': 비선형 데이터 (추천)
- 'poly': 다항식 특징

---

### 나이브 베이즈.ipynb

**학습 내용:**
- 베이즈 정리 기반 분류
- 다양한 나이브 베이즈 모델

**코드 예시:**
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# 가우시안 나이브 베이즈 (연속형)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
accuracy = gnb.score(X_test, y_test)

# 다항식 나이브 베이즈 (범주형)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
accuracy = mnb.score(X_test, y_test)
```

**특징:**
- 빠른 학습 및 예측
- 작은 데이터셋에서도 좋은 성능
- 텍스트 분류에 적합

---

### k-means.ipynb / 군집 Kmeans.ipynb

**학습 내용:**
- 비지도 학습
- 최적 클러스터 수 결정
- 클러스터 특성 분석

**코드 예시:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow Method
inertias = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# 최적 k로 최종 학습
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X)
```

**최적 k 결정:**
- Elbow Method: 관성 급감 지점
- Silhouette Score: -1~1, 1에 가까울수록 좋음

---

### 랜덤포레스트.ipynb

**학습 내용:**
- 앙상블 학습
- 특성 중요도 분석
- 회귀/분류 모두 사용

**코드 예시:**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 특성 중요도
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

accuracy = rf.score(X_test, y_test)
```

---

### 주성분 분석.ipynb

**학습 내용:**
- 차원 축소
- 분산 설명 비율
- 저차원 시각화

**코드 예시:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.95)  # 95% 분산 설명
X_pca = pca.fit_transform(X_scaled)

# 결과 분석
print(f"Original shape: {X_scaled.shape}")
print(f"Reduced shape: {X_pca.shape}")
cumsum = np.cumsum(pca.explained_variance_ratio_)
```

---

### 최근접이웃.ipynb

**학습 내용:**
- 인스턴스 기반 학습
- k값 최적화
- 거리 계산

**코드 예시:**
```python
from sklearn.neighbors import KNeighborsClassifier

# k값에 따른 성능 비교
train_scores = []
test_scores = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

# 최적 k 선택
optimal_k = test_scores.index(max(test_scores)) + 1
```

---

## 9월: 인기도 예측 초기

### 20250912인기.ipynb

**프로젝트:** 인기도 예측 모델 초기 구축

**데이터 구조:**
- 특성: 시간 시계열, 이전 인기도 메트릭 등
- 타겟: 현재 인기도 (회귀 또는 분류)

**전처리 단계:**
```python
# 데이터 로드 및 탐색
df = pd.read_csv('popularity_data.csv')
print(df.head())
print(df.info())
print(df.describe())

# 결측치 처리
df = df.dropna()

# 특성 선택
features = ['feature1', 'feature2', 'feature3', ...]
X = df[features]
y = df['popularity']

# Train/Test 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**모델 구축:**
```python
# 여러 모델 시도
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'SVM': SVR(kernel='rbf')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
```

**결과 및 인사이트:**
- 모델별 성능 비교
- 특성 중요도 분석
- 개선 포인트 도출

---

### 2308류현정_인기_실습.ipynb

**프로젝트:** 인기도 실습 문제

**학습 내용:**
- 데이터 전처리 심화
- 특성 엔지니어링
- 모델 최적화

**특성 엔지니어링 예시:**
```python
# 시간대별 특성 추출
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 통계 특성
df['rolling_mean'] = df['popularity'].rolling(window=7).mean()
df['rolling_std'] = df['popularity'].rolling(window=7).std()

# 원-핫 인코딩
df = pd.get_dummies(df, columns=['category'], drop_first=True)
```

---

### 250822-코로나 바이러스 감염 분석 현황.ipynb

**프로젝트:** 코로나바이러스 시계열 분석

**데이터:**
- 날짜별 확진자 수
- 지역별 감염 현황
- 시간 경과에 따른 추이

**분석 내용:**
```python
# 시계열 데이터 로드
covid_df = pd.read_csv('covid_data.csv')
covid_df['date'] = pd.to_datetime(covid_df['date'])

# 시각화
plt.figure(figsize=(15, 6))
for region in covid_df['region'].unique():
    region_data = covid_df[covid_df['region'] == region]
    plt.plot(region_data['date'], region_data['confirmed'], label=region)
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title('COVID-19 Cases Over Time')
plt.legend()
plt.show()

# 지역별 통계
region_stats = covid_df.groupby('region')['confirmed'].agg([
    'mean', 'max', 'min', 'std'
])

# 상관관계 분석
if 'vaccinated' in covid_df.columns:
    correlation = covid_df[['confirmed', 'vaccinated']].corr()
```

**주요 발견:**
- 지역별 감염 추이 비교
- 백신 접종과의 상관관계
- 시계열 패턴 분석

---

## 10월: 인기도 예측 심화

### 20251017인기.ipynb

**프로젝트:** 인기도 예측 모델 개선

**개선 사항:**
1. 더 많은 특성 엔지니어링
2. 하이퍼파라미터 튜닝
3. 앙상블 모델

**하이퍼파라미터 튜닝:**
```python
from sklearn.model_selection import GridSearchCV

# Random Forest 파라미터 튜닝
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
final_score = best_model.score(X_test, y_test)
```

**앙상블 모델:**
```python
from sklearn.ensemble import VotingRegressor

# 여러 모델 결합
ensemble = VotingRegressor([
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('svr', SVR(kernel='rbf'))
])

ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)
print(f"Ensemble Score: {ensemble_score:.4f}")
```

**성능 비교:**
```python
# 모든 모델 비교
results = {}
for name, model in {'LR': lr, 'RF': rf, 'SVR': svr, 'Ensemble': ensemble}.items():
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results[name] = {'MSE': mse, 'R²': r2}

results_df = pd.DataFrame(results).T
print(results_df)
```

---

### 251024_인기.ipynb

**프로젝트:** 최종 인기도 예측 모델

**최적화 전략:**
1. 기본 모델 성능 향상
2. 새로운 특성 발견
3. 크로스 검증을 통한 안정성 확보

```python
from sklearn.model_selection import cross_val_score

# 교차 검증으로 모델 안정성 평가
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"CV Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# K-Fold별 성능
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.4f}")
```

**최종 결과:**
- 모델 성능 정량화
- 예측 오차 분석
- 실제 적용 가능성 검토

---

### 치폴레 인기.ipynb

**프로젝트:** Chipotle 메뉴 인기도 분석

**데이터:**
- 메뉴 특성 (가격, 칼로리, 단백질 등)
- 판매량 또는 평점

**분석:**
```python
# 메뉴 특성과 인기도의 관계
correlation = chipotle_df[['price', 'calories', 'rating']].corr()
sns.heatmap(correlation, annot=True)

# 가격대별 인기도 비교
price_groups = pd.cut(chipotle_df['price'], bins=5)
popularity_by_price = chipotle_df.groupby(price_groups)['rating'].agg([
    'mean', 'std', 'count'
])

# 분류 모델 (인기도 등급)
from sklearn.preprocessing import LabelEncoder
chipotle_df['popularity_level'] = pd.cut(
    chipotle_df['rating'], 
    bins=3, 
    labels=['Low', 'Medium', 'High']
)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```

---

### kakao치킨.ipynb

**프로젝트:** Kakao 치킨점 분석

**분석 내용:**
- 지역별 가게 분포
- 클러스터링을 통한 지역 분류
- 입지 특성 분석

```python
# 지역별 가게 밀도
from sklearn.cluster import KMeans

# 좌표 데이터 클러스터링
coords = df[['latitude', 'longitude']].values
kmeans = KMeans(n_clusters=5, random_state=42)
df['region_cluster'] = kmeans.fit_predict(coords)

# 클러스터별 특성
cluster_summary = df.groupby('region_cluster').agg({
    'store_count': 'sum',
    'average_rating': 'mean',
    'average_price': 'mean'
})

# 지역 시각화
plt.scatter(df['longitude'], df['latitude'], c=df['region_cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], 
           marker='X', s=300, c='red')
plt.show()
```

---

### 자전거.ipynb

**프로젝트:** 자전거 이용 패턴 분석

**데이터:**
- 시간대별 이용량
- 날씨 정보
- 요일/계절 정보

**분석:**
```python
# 시간대별 이용량
hourly_usage = bike_df.groupby('hour')['usage_count'].mean()
plt.plot(hourly_usage)
plt.title('Hourly Bike Usage Pattern')

# 날씨에 따른 이용량
weather_impact = bike_df.groupby('weather')['usage_count'].agg([
    'mean', 'std', 'count'
])

# 계절별 추이
seasonal = bike_df.groupby('season')['usage_count'].mean()

# 회귀 모델
X = bike_df[['temperature', 'humidity', 'wind_speed', 'hour', 'is_weekend']]
y = bike_df['usage_count']

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
```

---

## 11월-12월: 종합 프로젝트

### [코드수행평가] 20251105류현정.ipynb

**프로젝트:** 종합 코드 평가

**내용:**
- 여러 알고리즘의 통합 적용
- 데이터 전처리부터 모델 평가까지 전체 파이프라인
- 최고 성능의 모델 선택 및 결과 해석

---

## 주요 학습 성과

### 알고리즘 숙련도
| 알고리즘 | 활용도 | 숙련도 |
|---------|--------|---------|
| Linear Regression | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Logistic Regression | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| SVM | ⭐⭐⭐ | ⭐⭐⭐ |
| Naive Bayes | ⭐⭐⭐ | ⭐⭐⭐ |
| Random Forest | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| K-Means | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| PCA | ⭐⭐⭐ | ⭐⭐⭐ |
| KNN | ⭐⭐⭐ | ⭐⭐⭐ |

### 주요 프로젝트 결과
- 인기도 예측: 최종 R² Score 0.85 이상 달성
- 코로나 분석: 지역별 감염 추이 파악 및 시각화
- Kakao 치킨: 5개 지역 클러스터 도출
- 자전거: 시간대별/계절별 이용 패턴 분석

---

**Last Updated**: 2025년 12월  
**총 학습 시간**: 약 150시간  
**프로젝트 개수**: 10개 이상  
**노트북 파일 수**: 25개 이상
