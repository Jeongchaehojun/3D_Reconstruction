"""
Fundamental Matrix Module

두 이미지 사이의 기하학적 관계를 정의하는 기초 행렬(Fundamental Matrix)을 계산합니다.

두 이미지 사이의 기하학적 관계는 x'^T F x = 0 와 같은 기초 행렬 식에 의해 정의됩니다.
여기서 x와 x'는 각각 첫 번째와 두 번째 이미지의 대응점입니다.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FundamentalMatrixResult:
    """기초 행렬 계산 결과"""
    F: np.ndarray           # 3x3 기초 행렬
    mask: np.ndarray        # 인라이어 마스크
    num_inliers: int        # 인라이어 수
    inlier_ratio: float     # 인라이어 비율


class FundamentalMatrixEstimator:
    """
    기초 행렬 추정 클래스
    
    8-point 알고리즘과 RANSAC을 사용하여 robust하게 기초 행렬을 계산합니다.
    
    에피폴라 제약식:
        두 이미지 사이의 기하학적 관계는 x'^T F x = 0 와 같은 
        기초 행렬(Fundamental Matrix) 식에 의해 정의됩니다.
    
    Attributes:
        method: 계산 방법 (RANSAC, LMEDS, 8POINT)
        ransac_threshold: RANSAC 임계값 (픽셀 단위)
        confidence: 계산 신뢰도
    """
    
    # 지원하는 방법
    METHODS = {
        "ransac": cv2.FM_RANSAC,
        "lmeds": cv2.FM_LMEDS,
        "8point": cv2.FM_8POINT
    }
    
    def __init__(self, method: str = "ransac", 
                 ransac_threshold: float = 3.0,
                 confidence: float = 0.99):
        """
        기초 행렬 추정기 초기화
        
        Args:
            method: "ransac", "lmeds", 또는 "8point"
            ransac_threshold: RANSAC 에러 임계값 (픽셀)
            confidence: 결과 신뢰도 (0~1)
        """
        if method.lower() not in self.METHODS:
            raise ValueError(f"지원하지 않는 방법: {method}")
        
        self.method = self.METHODS[method.lower()]
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
    
    def estimate(self, pts1: np.ndarray, 
                 pts2: np.ndarray) -> Optional[FundamentalMatrixResult]:
        """
        대응점으로부터 기초 행렬을 계산합니다.
        
        Args:
            pts1: 첫 번째 이미지의 점들 (Nx2)
            pts2: 두 번째 이미지의 점들 (Nx2)
        
        Returns:
            FundamentalMatrixResult: 계산 결과, 실패 시 None
        """
        if len(pts1) < 8 or len(pts2) < 8:
            print(f"경고: 기초 행렬 계산에는 최소 8개의 대응점이 필요합니다. "
                  f"현재: {len(pts1)}개")
            return None
        
        # 기초 행렬 계산
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=self.method,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence
        )
        
        if F is None or F.shape != (3, 3):
            return None
        
        # 인라이어 통계
        mask = mask.ravel()
        num_inliers = np.sum(mask)
        inlier_ratio = num_inliers / len(mask)
        
        return FundamentalMatrixResult(
            F=F,
            mask=mask,
            num_inliers=int(num_inliers),
            inlier_ratio=inlier_ratio
        )
    
    def compute_epipolar_lines(self, points: np.ndarray, 
                               F: np.ndarray,
                               which_image: int = 1) -> np.ndarray:
        """
        한 이미지의 점에 대응하는 다른 이미지의 에피폴라 라인을 계산합니다.
        
        에피폴라 라인은 l = F^T x (또는 l' = Fx) 로 계산됩니다.
        
        Args:
            points: 점들의 좌표 (Nx2)
            F: 기초 행렬
            which_image: 점이 속한 이미지 (1 또는 2)
        
        Returns:
            np.ndarray: 에피폴라 라인 (Nx3, [a, b, c] for ax + by + c = 0)
        """
        points = np.float32(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        lines = cv2.computeCorrespondEpilines(points, which_image, F)
        return lines.reshape(-1, 3)
    
    def draw_epipolar_lines(self, image1: np.ndarray, image2: np.ndarray,
                            pts1: np.ndarray, pts2: np.ndarray,
                            F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        두 이미지에 에피폴라 라인을 그립니다.
        
        Args:
            image1, image2: 입력 이미지
            pts1, pts2: 대응점
            F: 기초 행렬
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 에피폴라 라인이 그려진 이미지
        """
        img1_color = image1.copy() if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        img2_color = image2.copy() if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        h, w = img1_color.shape[:2]
        
        # 이미지 2의 에피폴라 라인 (이미지 1의 점에 대응)
        lines1 = self.compute_epipolar_lines(pts1, F, which_image=1)
        
        # 이미지 1의 에피폴라 라인 (이미지 2의 점에 대응)
        lines2 = self.compute_epipolar_lines(pts2, F, which_image=2)
        
        np.random.seed(42)  # 재현성을 위한 시드
        
        for i, (line, pt1, pt2) in enumerate(zip(lines1, pts1, pts2)):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # 에피폴라 라인의 끝점 계산
            a, b, c = line
            x0, y0 = 0, int(-c / b) if b != 0 else 0
            x1, y1 = w, int(-(c + a * w) / b) if b != 0 else 0
            
            # 이미지 2에 에피폴라 라인 그리기
            cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)
            
            # 이미지 1에 점 그리기
            cv2.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)
        
        return img1_color, img2_color
    
    def verify_epipolar_constraint(self, pts1: np.ndarray, 
                                   pts2: np.ndarray,
                                   F: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        에피폴라 제약식 x'^T F x = 0 을 검증합니다.
        
        Args:
            pts1, pts2: 대응점
            F: 기초 행렬
        
        Returns:
            Tuple[float, np.ndarray]: 평균 에러와 각 점의 에러
        """
        errors = []
        
        for pt1, pt2 in zip(pts1, pts2):
            # 동차 좌표로 변환
            p1 = np.array([pt1[0], pt1[1], 1.0])
            p2 = np.array([pt2[0], pt2[1], 1.0])
            
            # x'^T F x 계산
            error = np.abs(p2 @ F @ p1)
            errors.append(error)
        
        errors = np.array(errors)
        return float(np.mean(errors)), errors


def compute_essential_matrix(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    기초 행렬로부터 에센셜 행렬을 계산합니다.
    
    E = K'^T F K
    
    Args:
        F: 기초 행렬 (3x3)
        K: 카메라 내부 파라미터 행렬 (3x3)
    
    Returns:
        np.ndarray: 에센셜 행렬 (3x3)
    """
    E = K.T @ F @ K
    return E


if __name__ == "__main__":
    print("=== 기초 행렬 모듈 테스트 ===\n")
    
    # 테스트용 가상 데이터
    np.random.seed(42)
    
    # 3D 점 생성 (실제에서는 알 수 없음)
    num_points = 50
    X = np.random.rand(num_points, 3) * 10
    X[:, 2] += 5  # Z 방향으로 이동
    
    # 가상 카메라 설정
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 첫 번째 카메라: 원점
    R1 = np.eye(3)
    t1 = np.zeros(3)
    
    # 두 번째 카메라: 우측으로 이동
    R2 = np.eye(3)
    t2 = np.array([1.0, 0, 0])
    
    # 투영 (간단한 예시)
    pts1 = (K @ X.T).T
    pts1 = pts1[:, :2] / pts1[:, 2:3]
    
    X_transformed = (R2 @ X.T).T + t2
    pts2 = (K @ X_transformed.T).T
    pts2 = pts2[:, :2] / pts2[:, 2:3]
    
    # 기초 행렬 계산
    estimator = FundamentalMatrixEstimator(method="ransac")
    result = estimator.estimate(pts1, pts2)
    
    if result is not None:
        print(f"기초 행렬 F:\n{result.F}\n")
        print(f"인라이어: {result.num_inliers}/{len(pts1)} ({result.inlier_ratio:.2%})")
        
        # 에피폴라 제약식 검증
        mean_error, _ = estimator.verify_epipolar_constraint(pts1, pts2, result.F)
        print(f"평균 에피폴라 에러: {mean_error:.6f}")
    else:
        print("기초 행렬 계산 실패")
