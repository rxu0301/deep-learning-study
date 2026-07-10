# 딥러닝 학습 저장소 (Deep Learning Study Repository)

> 2026년 3월 11일 ~ 6월 8일까지 진행된 딥러닝 학습 내용을 정리한 저장소입니다.

## 📚 학습 개요

이 저장소는 딥러닝의 기초부터 실전 프로젝트까지 체계적으로 학습한 내용을 담고 있습니다.
TensorFlow와 Keras를 활용하여 다양한 딥러닝 모델을 구현하고 실험한 결과를 정리했습니다.

## 🛠 기술 스택

- **프로그래밍 언어**: Python 3.10
- **딥러닝 프레임워크**: TensorFlow 2.x, Keras
- **데이터 처리**: NumPy, Pandas
- **시각화**: Matplotlib, Seaborn
- **기타 라이브러리**: scikit-learn, OpenCV

## 📖 학습 내용

### 1. 딥러닝 기초 (3월 초반)
- **환경 설정**: GPU 설정 및 필수 라이브러리 설치
- **MNIST 손글씨 숫자 인식**
  - 신경망 기본 구조 이해
  - Dense Layer, Flatten Layer 활용
  - 모델 학습 및 평가
  - 정확도: 약 97.5%

### 2. 데이터 전처리 및 분석 (3월 중순)
- **데이터 시각화 기법**
- **Pima Indians Diabetes 데이터셋**
  - 이진 분류 문제 해결
  - 데이터 전처리 및 정규화
  - Binary Crossentropy 손실 함수 활용
  
- **Iris 꽃 분류**
  - 다중 클래스 분류
  - One-hot Encoding
  - Softmax 활성화 함수
  - 정확도: 약 95%

### 3. 고급 모델링 기법 (3월 하순 ~ 4월)
- **타이타닉 생존자 예측**
  - 결측치 처리
  - 범주형 데이터 인코딩
  - 모델 최적화
  
- **와인 품질 예측**
  - 회귀 문제 해결
  - 특성 중요도 분석
  - 하이퍼파라미터 튜닝

- **와인 분류 (Red vs White)**
  - 이진 분류
  - 데이터 불균형 처리
  - 정확도: 약 97%

### 4. 신경망 최적화 (3월 ~ 4월)
- **ModelCheckpoint & EarlyStopping**
  - 모델 저장 및 복원
  - 과적합 방지 기법
  
- **Seeds 데이터 분류**
  - 소규모 데이터셋 처리
  - 배치 크기 최적화
  - Validation Split

### 5. 합성곱 신경망 (CNN) (3월 말 ~ 4월)
- **CNN 기초 이론**
  - Convolution Layer
  - MaxPooling Layer
  - Filter와 Feature Map
  
- **MNIST with CNN**
  - CNN 아키텍처 구현
  - 정확도: 약 95%
  
- **Fashion MNIST**
  - 의류 이미지 분류 (10개 클래스)
  - 복잡한 CNN 구조
  - 정확도: 약 88%

### 6. 고급 이미지 분류 (4월 초)
- **CIFAR-10 데이터셋**
  - 32x32 컬러 이미지 분류
  - 10개 클래스 (비행기, 자동차, 새, 고양이 등)
  - 데이터 증강 기법
  
- **커스텀 이미지 테스트**
  - 실제 이미지로 모델 테스트
  - 이미지 전처리 (리사이징, 정규화)
  - 예측 결과 시각화

### 7. 실전 프로젝트 (4월 ~ 6월)
- **EuroSAT 위성 이미지 분류**
  - 2,750개 클래스의 토지 이용 분류
  - 대규모 데이터셋 처리
  - Transfer Learning 적용
  
- **프로젝트 특징**
  - 실제 위성 이미지 데이터
  - Annual Crop, Forest, Herbaceous Vegetation 등 분류
  - 각 클래스당 3,000개 이미지

### 8. 최종 프로젝트: DNA 프로모터 예측 (6월)
- **DNA 서열 기반 프로모터 예측 모델**
  - 주제: DNA 서열이 프로모터(Promoter) 영역인지 분류하는 이진 분류 문제
  - 데이터: Gene Promoter Sequences (Kaggle)
  
- **모델 설계**
  - Conv1D를 활용한 DNA 서열 패턴 인식
  - 원-핫 인코딩 (A, T, C, G → 4차원 벡터)
  - 입력 형태: (57, 4) - 57개 염기, 4가지 뉴클레오타이드
  
- **네트워크 구조**
  - Conv1D Layer (32 filters, kernel_size=3): 지역적 패턴(모티프) 추출
  - MaxPooling Layer: 특징 압축 및 노이즈 제거
  - Dense Layer (64 units) + Dropout (0.3): 과적합 방지
  - Output Layer (Sigmoid): 이진 분류
  
- **데이터 분할**
  - Train (70%) : Validation (20%) : Test (10%)
  - Stratified Split으로 클래스 비율 유지
  
- **성과**
  - Train Accuracy: 100%
  - Validation Accuracy: 약 90.5%
  - Test 데이터 평가 완료
  - 생물정보학(Bioinformatics) 분야 딥러닝 적용 경험

## 📂 디렉토리 구조

```
dl/
├── .ipynb_checkpoints/    # Jupyter 노트북 체크포인트
├── dataset/               # 학습 데이터셋
│   ├── 2750/             # EuroSAT 데이터
│   │   ├── AnnualCrop/   # 연간 작물 (3,000장)
│   │   ├── Forest/       # 산림 (3,000장)
│   │   └── HerbaceousVegetation/ # 초본 식물 (3,000장)
│   ├── Titanic/          # 타이타닉 데이터
│   ├── promoters.data    # DNA 프로모터 서열 데이터
│   ├── iris3.csv         # Iris 데이터
│   ├── pima-indians-diabetes.csv
│   ├── seeds.csv
│   ├── wine.csv
│   └── winequality-white.csv
├── test/                  # 테스트 이미지 및 모델
├── Deep_Learning_Project.ipynb  # 최종 프로젝트 (DNA 프로모터 예측)
└── *.ipynb               # 학습 노트북 파일들
```

## 🎯 주요 성과

1. **기초부터 고급까지 체계적 학습**: 기본 신경망부터 CNN까지 단계별 학습
2. **다양한 데이터셋 경험**: 표 데이터, 이미지 데이터, 생물학 데이터, 대규모 데이터셋
3. **실전 프로젝트 완성**: 위성 이미지 분류, DNA 프로모터 예측 등 실무 적용 가능한 프로젝트
4. **높은 정확도 달성**: 대부분의 프로젝트에서 85% 이상의 정확도
5. **다양한 딥러닝 아키텍처 경험**: Dense Network, CNN (Conv2D, Conv1D) 활용
6. **생물정보학 응용**: DNA 서열 분석을 통한 딥러닝 적용 범위 확장

## 📈 학습 진행 방식

각 노트북 파일은 날짜별로 구성되어 있으며, 다음과 같은 형식을 따릅니다:
- **260311**: 2026년 3월 11일
- **260316_01**: 2026년 3월 16일 1부
- **260316_02**: 2026년 3월 16일 2부

## 🔍 주요 학습 포인트

### 모델 구성
- Sequential API 활용
- Layer 구성 (Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout)
- 활성화 함수 (ReLU, Sigmoid, Softmax)

### 최적화
- Adam Optimizer
- Learning Rate 조정
- Batch Size 최적화

### 평가 지표
- Accuracy (정확도)
- Loss (손실 함수)
- Confusion Matrix
- 교차 검증

## 💡 향후 계획

- RNN, LSTM을 활용한 시계열 데이터 분석
- Transfer Learning 심화 학습
- GAN을 활용한 이미지 생성
- 실시간 객체 탐지 프로젝트

## 📝 참고 자료

- TensorFlow 공식 문서
- Keras 공식 문서
- 각종 데이터셋 출처 (UCI ML Repository, Kaggle 등)

---

**Last Updated**: 2026년 6월 8일
