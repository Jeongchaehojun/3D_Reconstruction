# 3D Reconstruction: Structure from Motion (SfM)

<p align="center">
  <img src="docs/images/sfm_concept.png" alt="SfM Concept" width="600"/>
</p>

> 여러 2D 이미지로부터 3D 포인트 클라우드를 복원하는 Structure from Motion 파이프라인

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 목차

- [개요](#-개요)
- [알고리즘 원리](#-알고리즘-원리)
- [설치](#-설치)
- [사용법](#-사용법)
- [프로젝트 구조](#-프로젝트-구조)
- [테스트](#-테스트)
- [참고 자료](#-참고-자료)

---

## 🎯 개요

이 프로젝트는 **Structure from Motion (SfM)** 알고리즘을 순수 Python으로 구현합니다. 여러 각도에서 촬영한 2D 이미지로부터 장면의 3D 구조와 카메라 위치를 동시에 복원합니다.

### 주요 기능

- 🔍 **특징점 검출**: SIFT, ORB 알고리즘 지원
- 🔗 **특징점 매칭**: BFMatcher, FLANN 기반 매칭
- 📐 **기하학적 추정**: Fundamental/Essential Matrix 계산
- 📷 **카메라 포즈 추정**: 상대적 R, t 복원
- 🔺 **삼각측량**: DLT 기반 3D 점 복원
- 🎯 **번들 조정**: 비선형 최적화로 정밀도 향상
- 👁️ **3D 시각화**: Open3D/Matplotlib 지원

---

## 📐 알고리즘 원리

### 1. 에피폴라 기하학 (Epipolar Geometry)

두 이미지 사이의 기하학적 관계는 **기초 행렬(Fundamental Matrix)** 식에 의해 정의됩니다:

$$x'^T F x = 0$$

여기서:
- $x$, $x'$는 각각 첫 번째와 두 번째 이미지의 동차 좌표 대응점
- $F$는 3×3 기초 행렬 (rank 2)

### 2. 에센셜 행렬 (Essential Matrix)

카메라 내부 파라미터 $K$가 알려진 경우, 에센셜 행렬 $E$는 다음과 같이 계산됩니다:

$$E = K'^T F K$$

에센셜 행렬은 카메라 간의 **회전 행렬 $R$**과 **평행이동 벡터 $t$**로 분해됩니다:

$$E = [t]_\times R$$

### 3. 삼각측량 (Triangulation)

두 카메라의 투영 행렬 $P_1$, $P_2$와 대응점 $x_1$, $x_2$가 주어지면, 3D 점 $X$는 다음 관계를 통해 복원됩니다:

$$x_1 = P_1 X, \quad x_2 = P_2 X$$

**DLT(Direct Linear Transform)** 방식으로 $AX = 0$ 형태의 선형 시스템을 풀어 3D 좌표를 계산합니다.

### 4. 번들 조정 (Bundle Adjustment)

모든 재투영 에러의 합을 최소화하는 비선형 최적화:

$$\min_{C_i, X_j} \sum_i \sum_j \| x_{ij} - \pi(C_i, X_j) \|^2$$

여기서:
- $x_{ij}$: $j$번째 3D 점의 $i$번째 카메라에서의 2D 관측
- $\pi$: 투영 함수
- $C_i$: $i$번째 카메라 파라미터

### 5. 성능 지표

알고리즘의 정확도는 **정답 매칭 개수**와 **전체 매칭 시도 횟수** 사이의 비율로 계산됩니다:

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

---

## 🚀 설치

### 요구사항

- Python 3.8 이상
- OpenCV 4.5 이상 (opencv-contrib-python 권장)

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/3D_reconstruction.git
cd 3D_reconstruction

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

---

## 💻 사용법

### 기본 사용

```python
from src.sfm_pipeline import SfMPipeline, create_default_camera_matrix
import cv2

# 이미지 크기에 맞는 카메라 행렬 생성
K = create_default_camera_matrix(width=1920, height=1080, fov_degrees=60)

# SfM 파이프라인 생성
pipeline = SfMPipeline(K, feature_algorithm="sift")

# 이미지 목록으로 3D 재구성 실행
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
result = pipeline.run(image_paths)

if result is not None:
    print(f"복원된 3D 점: {result.num_points}개")
    print(f"재투영 에러: {result.reprojection_error:.4f} 픽셀")
    
    # 포인트 클라우드 저장
    pipeline.save_point_cloud(result, "output.ply")
```

### 개별 모듈 사용

```python
# 특징점 검출
from src.feature_detection import FeatureDetector

detector = FeatureDetector(algorithm="sift")
result = detector.detect(image)
print(f"검출된 특징점: {result.num_features}개")

# 특징점 매칭
from src.feature_matching import FeatureMatcher

matcher = FeatureMatcher(matcher_type="bf")
match_result = matcher.match(descriptors1, descriptors2)
print(f"좋은 매칭: {match_result.num_good}개")

# 기초 행렬 계산
from src.fundamental_matrix import FundamentalMatrixEstimator

estimator = FundamentalMatrixEstimator(method="ransac")
F_result = estimator.estimate(pts1, pts2)
print(f"인라이어 비율: {F_result.inlier_ratio:.2%}")
```

### 커맨드라인 실행

```bash
# 데모 실행
python examples/demo.py

# 이미지로 SfM 실행
python -m src.sfm_pipeline image1.jpg image2.jpg image3.jpg

# 테스트 실행
python -m pytest tests/ -v
```

### 3D 시각화

```python
from visualization.point_cloud_viewer import PointCloudVisualizer, load_ply

# PLY 파일 로드 및 시각화
points, colors = load_ply("output.ply")

viz = PointCloudVisualizer()
viz.visualize(points, colors, title="My Point Cloud")
```

---

## 📁 프로젝트 구조

```
3D_reconstruction/
├── src/                          # 핵심 라이브러리
│   ├── __init__.py
│   ├── feature_detection.py      # SIFT/ORB 특징점 검출
│   ├── feature_matching.py       # 특징점 매칭
│   ├── fundamental_matrix.py     # 기초 행렬 계산
│   ├── camera_pose.py            # 카메라 포즈 추정
│   ├── triangulation.py          # 삼각측량
│   ├── bundle_adjustment.py      # 번들 조정
│   └── sfm_pipeline.py           # 전체 파이프라인
│
├── visualization/                # 시각화 도구
│   └── point_cloud_viewer.py
│
├── tests/                        # 테스트
│   ├── test_images/
│   └── test_sfm.py
│
├── examples/                     # 예제
│   └── demo.py
│
├── docs/                         # 문서
│   └── images/
│
├── requirements.txt
└── README.md
```

---

## ✅ 테스트

### 전체 테스트 실행

```bash
python -m pytest tests/ -v
```

### 특정 모듈 테스트

```bash
# 특징점 검출 테스트
python -m pytest tests/test_sfm.py::TestFeatureDetection -v

# 기초 행렬 테스트
python -m pytest tests/test_sfm.py::TestFundamentalMatrix -v
```

### 테스트 이미지

다양한 특성의 테스트 이미지:

| 유형 | 설명 | 예상 결과 |
|------|------|----------|
| 텍스처 풍부 | 벽돌, 나뭇잎 등 | 1000+ 특징점 |
| 텍스처 부족 | 단색 벽면 | < 50 특징점 (실패 케이스) |
| 반복 패턴 | 타일, 창문 | 매칭 모호성 발생 가능 |

---

## 📚 참고 자료

### 논문

- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*
- Lowe, D. G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*
- Triggs, B., et al. (2000). *Bundle Adjustment - A Modern Synthesis*

### 라이브러리

- [OpenCV](https://opencv.org/) - 컴퓨터 비전 라이브러리
- [Open3D](http://www.open3d.org/) - 3D 데이터 처리
- [SciPy](https://scipy.org/) - 과학 계산

---

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

