# 미래유망 1학년 - AI 기초 학습 저장소 (2024년 후반기)

> 2024년 9월 ~ 11월, 1학기 AI 교육 과정에서 학습한 기초 내용을 정리한 저장소입니다.

## 📚 학습 개요

이 저장소는 Python 데이터 분석 라이브러리 기초부터 기본 딥러닝 개념까지 초급 수준의 학습 내용을 담고 있습니다.
NumPy, Pandas, Matplotlib, OpenCV, 그리고 TensorFlow/Keras를 활용하여 데이터 처리, 시각화, 그리고 신경망 기초를 학습했습니다.

## 🎯 학습 내용 요약

### 1. 기초 라이브러리 (NumPy, Pandas)
- **NumPy**: 배열 연산, 선형대수, 행렬 조작
- **Pandas**: 데이터프레임 생성 및 조작, 데이터 전처리 기초

### 2. 데이터 시각화 (Matplotlib, Seaborn)
- 기본 차트: 라인 플롯, 산점도, 히스토그램, 박스플롯
- 고급 시각화: 다중 플롯, 색상 매핑, 스타일 지정

### 3. 이미지 처리 (OpenCV)
- 이미지 읽기/쓰기
- 흑백 변환 (Grayscale)
- 색상 공간 변환 (RGB, HSV)

### 4. 기본 딥러닝 (TensorFlow/Keras)
- Dense 신경망 구조
- MNIST 손글씨 인식
- 회귀 문제 (시험 점수 예측)
- 이진 분류 (Binary Classification) 입문
- Pooling 연산 이해

## 📂 디렉토리 구조

```
미래유망_1학년/
├── numpy/                        # NumPy 학습 자료
│   ├── NumPy_01.ipynb           # 기본 배열 연산
│   ├── NumPy_02.ipynb           # 고급 연산
│   └── numpy_정리.ipynb          # NumPy 요약
├── pandas/                        # Pandas 학습 자료
│   ├── pandas.ipynb              # 기본 데이터프레임
│   ├── pandas2.ipynb             # 데이터 조작
│   ├── pandas_3.ipynb            # 심화 내용
│   └── Pandas_정리.ipynb         # Pandas 요약
├── Matplotlib/                    # Matplotlib 학습 자료
│   ├── Matplotlib_01.ipynb       # 기본 플롯
│   ├── Matplotlib_02.ipynb       # 고급 시각화
│   └── Matplotlib_정리.ipynb     # Matplotlib 요약
├── openCV_01_grayscale.ipynb     # 흑백 이미지 처리
├── openCV_2_color.ipynb          # 색상 이미지 처리
├── Deep_Learning_01_손글씨 분류.ipynb    # 신경망 기초
├── Deep_Learning_02_시험 점수 예측.ipynb # 회귀 문제
├── Deep_Learning_03_이진분류.ipynb      # 분류 문제
├── Deeplearning_손글씨_CNN.ipynb        # CNN 입문
├── numpy_random_Deep_Laerning.ipynb      # 난수와 딥러닝
├── pooling_01.ipynb              # Pooling 연산
├── 과제1.ipynb                   # 과제
├── 용어_EX.ipynb                 # 용어 정리
├── 실습1.ipynb                   # 실습
└── 실습/                         # 실습 폴더
    ├── one.ipynb
    ├── two.ipynb
    └── titanic.zip
```

## 🛠 기술 스택

- **프로그래밍 언어**: Python 3.7+
- **데이터 처리**: NumPy, Pandas
- **시각화**: Matplotlib, Seaborn
- **이미지 처리**: OpenCV 4.x
- **딥러닝**: TensorFlow 2.x, Keras
- **환경**: Google Colab, Jupyter Notebook

## 📖 학습 모듈별 설명

### NumPy 모듈
- 배열 생성 및 초기화
- 배열 연산 (기본, 브로드캐스팅)
- 선형대수 함수
- 난수 생성 및 통계 함수

### Pandas 모듈
- Series와 DataFrame 이해
- 데이터 인덱싱 및 선택
- 결측치 처리 기초
- 그룹화 및 병합

### Matplotlib 모듈
- Figure와 Axes 객체
- 다양한 플롯 유형
- 서브플롯 생성
- 스타일 및 색상 커스터마이징

### OpenCV 모듈
- 이미지 읽기/쓰기/표시
- 색공간 변환
- 이미지 처리 기초

### 딥러닝 모듈
- Sequential 모델 구축
- Dense 레이어 이해
- 활성화 함수 (ReLU, Sigmoid, Softmax)
- 손실 함수 및 최적화기
- 모델 학습 및 평가

## 🎓 주요 학습 포인트

1. **배열 기반 연산**: NumPy를 통한 효율적인 수치 계산
2. **데이터 조작**: Pandas를 통한 실제 데이터 처리
3. **데이터 시각화**: 차트를 통한 패턴 인식
4. **이미지 처리**: OpenCV 기초 조작
5. **신경망 기초**: 간단한 Dense 네트워크 구현
6. **분류 및 회귀**: 두 가지 기본 문제 유형 이해

## 📊 학습 난이도

| 모듈 | 난이도 | 소요 시간 |
|------|--------|---------|
| NumPy | ⭐⭐☆☆☆ | 1-2주 |
| Pandas | ⭐⭐⭐☆☆ | 2-3주 |
| Matplotlib | ⭐⭐☆☆☆ | 1-2주 |
| OpenCV | ⭐⭐⭐☆☆ | 1-2주 |
| 딥러닝 기초 | ⭐⭐⭐⭐☆ | 2-3주 |

## 💾 학습 커리큘럼

### Week 1-2: NumPy 기초
- 배열 생성 및 속성
- 기본 연산 및 브로드캐스팅
- 통계 함수

### Week 3-4: Pandas 기초
- Series와 DataFrame
- 데이터 인덱싱
- 데이터 결합

### Week 5-6: Matplotlib 기초
- 기본 플롯 생성
- 다중 플롯
- 스타일 지정

### Week 7: OpenCV 기초
- 이미지 읽기/쓰기
- 색공간 변환
- 기본 필터

### Week 8-10: 딥러닝 기초
- 신경망 구조 이해
- MNIST 손글씨 인식
- 회귀 및 분류 문제
- CNN 기초

## 🔍 주요 실습 프로젝트

### 1. 손글씨 숫자 인식 (MNIST)
- 28×28 흑백 이미지 분류
- 10개 클래스 (0-9)
- Dense 신경망 사용
- 예상 정확도: ~95%

### 2. 시험 점수 예측
- 회귀 문제
- 선형 관계 학습
- Dense 네트워크 활용

### 3. 이진 분류 문제
- 두 개 클래스 분류 입문
- Sigmoid 활성화 함수
- Binary Crossentropy 손실 함수

### 4. 이미지 처리
- 흑백 변환
- 색상 공간 변환
- 기본 필터 적용

## 🎯 성취 목표

✅ NumPy와 Pandas를 통한 데이터 처리 능력  
✅ Matplotlib을 통한 데이터 시각화 기술  
✅ OpenCV를 통한 이미지 처리 기초  
✅ Keras를 통한 간단한 신경망 구현  
✅ 머신러닝 문제의 기본 개념 이해

## 📚 상세 학습 기록

각 파일별 상세한 학습 내용은 [STUDY_DETAILS.md](./STUDY_DETAILS.md)를 참고하세요.

## 💡 다음 단계

이 학습 과정을 마친 후의 추천 진행 경로:

1. 머신러닝 알고리즘 학습 (scikit-learn)
2. 더 복잡한 데이터셋 분석
3. CNN을 통한 고급 이미지 분류
4. 데이터 전처리 심화
5. 실제 프로젝트 구현

---

**Last Updated**: 2024년 11월  
**학습 기간**: 2024년 9월 ~ 11월 (약 3개월)  
**총 노트북 파일**: 20개 이상
