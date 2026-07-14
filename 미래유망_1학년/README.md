# 미래유망_1학년 전체 설정 가이드

## 📋 프로젝트 개요

이 프로젝트는 pandas, matplotlib, numpy를 학습하는 학생들을 위해 모든 출력 파일(그래프, CSV, Excel 등)이 자동으로 정리된 폴더에 저장되도록 설정되어 있습니다.

## 🎯 주요 설정 사항

### 1. 한국어 폰트 자동 설정
모든 matplotlib을 사용하는 파일에 자동으로 한국어 폰트가 설정됩니다:
- **Windows**: Malgun Gothic (맑은 고딕)
- **macOS**: AppleGothic
- **Linux**: Noto Sans CJK JP

### 2. 결과 폴더 구조
```
미래유망_1학년/
├── Matplotlib/
│   ├── results/          ← 모든 그래프가 여기에 저장됨
│   └── *.ipynb
├── pandas/
│   ├── results/          ← 모든 CSV, Excel, 그래프 저장
│   └── *.ipynb
├── numpy/
│   ├── results/          ← 모든 NPY, CSV, 그래프 저장
│   └── *.ipynb
└── README.md
```

### 3. 그래프 자동 저장
모든 `plt.show()` 전에 자동으로 `plt.savefig()`가 실행됩니다:
```python
# 자동 추가됨
plt.savefig('results/filename.png', dpi=300, bbox_inches='tight')
plt.show()
```

**저장 설정:**
- 형식: PNG (무손실)
- 해상도: DPI 300 (인쇄용 고품질)
- 여백: 자동 최적화

## 📁 각 폴더 설명

### Matplotlib 폴더
- **학습 내용**: 기본 그래프 작성, 스타일, 커스터마이징
- **저장되는 파일**:
  - 선 그래프 이미지
  - 산점도 이미지
  - 히스토그램 이미지
  - 기타 matplotlib 시각화
- **파일 수**: ~50개 그래프

### pandas 폴더
- **학습 내용**: 데이터프레임 생성, 조작, 시각화
- **저장되는 파일**:
  - CSV 파일 (데이터)
  - Excel 파일 (데이터)
  - JSON 파일 (데이터)
  - PNG 이미지 (그래프)
- **파일 수**: 다양한 데이터 + ~10개 그래프

### numpy 폴더
- **학습 내용**: 배열 생성, 연산, 데이터 저장
- **저장되는 파일**:
  - NPY 파일 (numpy 배열)
  - CSV 파일 (배열 데이터)
  - TXT 파일 (배열 데이터)
  - PNG 이미지 (시각화)
- **파일 수**: 다양한 배열 + ~10개 그래프

## 🚀 사용 방법

### Jupyter Notebook 실행
1. 각 폴더의 `.ipynb` 파일 열기
2. 노트북 실행 (Shift + Enter)
3. 모든 출력은 자동으로 `results/` 폴더에 저장됨

### 결과 확인
```
Matplotlib/results/
├── Matplotlib_1_2.png
├── Matplotlib_2_2.png
└── ... (총 50개+)
```

### 저장된 파일 초기화
```bash
# results 폴더 내용 삭제
rm -r Matplotlib/results/*
rm -r pandas/results/*
rm -r numpy/results/*

# 또는 수동으로 폴더 열어서 파일 삭제
```

## 📊 파일 생성 현황

| 폴더 | 파일 | 그래프 수 | 데이터 파일 |
|------|------|---------|-----------|
| **Matplotlib** | 3개 | ~50개 | - |
| **pandas** | 2개 | ~10개 | CSV, Excel, JSON |
| **numpy** | 2개 | ~10개 | NPY, CSV, TXT |

## ⚙️ 기술 세부사항

### 한국어 폰트 설정 코드
```python
import matplotlib.pyplot as plt
import platform

system = platform.system()
if system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
```

### 그래프 저장 코드
```python
plt.savefig('results/filename.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 🐛 문제 해결

### 그래프가 결과 폴더에 저장되지 않는 경우
1. `results/` 폴더가 존재하는지 확인
2. 경로가 맞는지 확인 (`results/filename.png`)
3. 쓰기 권한이 있는지 확인

### 한국어가 깨지는 경우
1. 해당 폰트가 설치되어 있는지 확인
2. 캐시 초기화: 
   ```python
   import matplotlib
   matplotlib.font_manager.rebuild()
   ```

### results 폴더 생성 오류
```python
import os
os.makedirs('results', exist_ok=True)
```

## 📝 노트북별 상세 설명

### Matplotlib 폴더
- `Matplotlib_01.ipynb`: 기본 플롯 (14개 그래프 저장)
- `Matplotlib_02.ipynb`: 스타일 및 포맷 (14개 그래프 저장)
- `Matplotlib_정리.ipynb`: 종합 가이드 (22개 그래프 저장)

### pandas 폴더
- `pandas.ipynb`: 기본 개념
- `pandas2.ipynb`: 데이터 조작 + CSV 저장
- `pandas_3.ipynb`: 고급 기능 + CSV 저장
- `Pandas_정리.ipynb`: 종합 가이드 (CSV, Excel, JSON 저장)
- `pandas_응용.ipynb`: 실습 프로젝트 (그래프 저장)
- `pandas_프로젝트.ipynb`: 데이터 분석 프로젝트 (그래프 저장)

### numpy 폴더
- `NumPy_01.ipynb`: 기본 배열
- `NumPy_02.ipynb`: 배열 연산
- `numpy_정리.ipynb`: 종합 정리 (NPY, CSV 저장)
- `numpy_응용.ipynb`: 실습 (그래프 저장)
- `numpy_프로젝트.ipynb`: 프로젝트 (그래프 저장)

## 🔗 관련 문서
- `SETUP_SUMMARY.md`: 전체 설정 요약
- `GRAPH_SAVE_REPORT.md`: 그래프 저장 상세 보고

## ✨ 주요 특징

✅ **자동화**: 한번 설정하면 모든 출력이 자동으로 정리됨
✅ **일관성**: 모든 파일이 같은 폴더에 저장됨
✅ **한국어 지원**: 한글이 깨지지 않음
✅ **고품질**: 그래프가 300 DPI로 저장됨
✅ **다양한 포맷**: PNG, CSV, Excel, JSON, NPY 등 지원

## 📞 문의 및 피드백

문제가 발생하면:
1. `results/` 폴더 확인
2. 파일 경로 확인
3. 쓰기 권한 확인

---

**마지막 업데이트**: 2026-07-14
**상태**: ✅ 모든 설정 완료
**총 수정 파일**: 19개 (한국어 폰트) + 12개 (출력 경로)
