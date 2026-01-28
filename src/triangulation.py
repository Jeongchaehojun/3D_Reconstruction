"""
Triangulation Module

두 이미지의 대응점과 카메라 행렬로부터 3D 점을 복원합니다.

삼각측량(Triangulation)은 두 시점에서 관찰된 2D 점을 역투영하여
교차점을 계산하는 방식으로 3D 좌표를 추정합니다.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TriangulationResult:
    """삼각측량 결과"""
    points_3d: np.ndarray       # Nx3 3D 점
    reprojection_error: float   # 평균 재투영 에러
    valid_mask: np.ndarray      # 유효한 점 마스크


class Triangulator:
    """
    삼각측량 클래스
    
    두 카메라의 투영 행렬과 대응점을 사용하여 3D 점을 복원합니다.
    
    DLT(Direct Linear Transform) 방식을 사용하며, OpenCV의 
    triangulatePoints 함수로 구현됩니다.
    """
    
    def __init__(self, min_depth: float = 0.0, max_depth: float = 1000.0):
        """
        삼각측량기 초기화
        
        Args:
            min_depth: 유효한 깊이의 최소값
            max_depth: 유효한 깊이의 최대값
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def triangulate(self, P1: np.ndarray, P2: np.ndarray,
                    pts1: np.ndarray, pts2: np.ndarray) -> TriangulationResult:
        """
        두 이미지의 대응점으로부터 3D 점을 복원합니다.
        
        삼각측량 원리:
            x = P X (2D 점 = 투영행렬 × 3D 점)
        
        DLT를 통해 역으로 X를 계산합니다.
        
        Args:
            P1, P2: 3x4 투영 행렬
            pts1, pts2: Nx2 대응점
        
        Returns:
            TriangulationResult: 복원된 3D 점과 통계
        """
        # 점 형태 변환
        pts1 = np.float64(pts1)
        pts2 = np.float64(pts2)
        
        if pts1.ndim == 2:
            pts1 = pts1.T  # 2xN
        if pts2.ndim == 2:
            pts2 = pts2.T  # 2xN
        
        # 삼각 측량 (결과: 4xN 동차 좌표)
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        
        # 동차 좌표를 3D 좌표로 변환
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T  # Nx3
        
        # 유효성 검사 (카메라 앞에 있고 적절한 깊이)
        # 첫 번째 카메라 기준 깊이
        depths = points_3d[:, 2]
        valid_mask = (depths > self.min_depth) & (depths < self.max_depth)
        
        # 재투영 에러 계산
        reprojection_error = self._compute_reprojection_error(
            P1, P2, pts1.T, pts2.T, points_3d
        )
        
        return TriangulationResult(
            points_3d=points_3d,
            reprojection_error=reprojection_error,
            valid_mask=valid_mask
        )
    
    def _compute_reprojection_error(self, P1: np.ndarray, P2: np.ndarray,
                                    pts1: np.ndarray, pts2: np.ndarray,
                                    points_3d: np.ndarray) -> float:
        """
        재투영 에러를 계산합니다.
        
        재투영 에러 = 원래 2D 점과 3D 점을 다시 투영한 점 사이의 거리
        
        Args:
            P1, P2: 투영 행렬
            pts1, pts2: 원래 2D 점 (Nx2)
            points_3d: 복원된 3D 점 (Nx3)
        
        Returns:
            float: 평균 재투영 에러 (픽셀 단위)
        """
        # 동차 좌표로 변환
        points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        
        # 재투영
        reproj1 = (P1 @ points_h.T).T
        reproj1 = reproj1[:, :2] / reproj1[:, 2:3]
        
        reproj2 = (P2 @ points_h.T).T
        reproj2 = reproj2[:, :2] / reproj2[:, 2:3]
        
        # 에러 계산
        error1 = np.linalg.norm(pts1 - reproj1, axis=1)
        error2 = np.linalg.norm(pts2 - reproj2, axis=1)
        
        return float(np.mean(error1) + np.mean(error2)) / 2
    
    def triangulate_dlt(self, P1: np.ndarray, P2: np.ndarray,
                        pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
        """
        단일 점에 대해 DLT 삼각측량을 직접 수행합니다.
        
        Ax = 0 형태의 선형 시스템을 SVD로 풀어 3D 점을 계산합니다.
        
        Args:
            P1, P2: 3x4 투영 행렬
            pt1, pt2: 2D 점 (x, y)
        
        Returns:
            np.ndarray: 3D 점 (x, y, z)
        """
        # A 행렬 구성
        A = np.zeros((4, 4))
        
        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0], pt2[1]
        
        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]
        
        # SVD 풀이
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # 가장 작은 특이값에 해당하는 벡터
        
        # 동차 좌표 → 3D 좌표
        X = X[:3] / X[3]
        
        return X


def triangulate_multiple_views(projections: list, 
                               point_tracks: list) -> np.ndarray:
    """
    여러 뷰에서 관찰된 점을 삼각측량합니다.
    
    Args:
        projections: List of 3x4 투영 행렬
        point_tracks: List of 2D 점 (각 뷰에서의 관측)
    
    Returns:
        np.ndarray: 3D 점
    """
    num_views = len(projections)
    
    if num_views < 2:
        raise ValueError("최소 2개의 뷰가 필요합니다.")
    
    # A 행렬 구성 (2*num_views x 4)
    A = np.zeros((2 * num_views, 4))
    
    for i, (P, pt) in enumerate(zip(projections, point_tracks)):
        x, y = pt[0], pt[1]
        A[2*i]     = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]
    
    # SVD 풀이
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]
    
    return X


if __name__ == "__main__":
    print("=== 삼각측량 테스트 ===\n")
    
    # 테스트용 카메라 설정
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 첫 번째 카메라: 원점
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    # 두 번째 카메라: 우측으로 이동
    R2 = np.eye(3)
    t2 = np.array([[2.0], [0.0], [0.0]])
    P2 = K @ np.hstack([R2, t2])
    
    # 가상 3D 점
    np.random.seed(42)
    num_points = 50
    true_points = np.random.rand(num_points, 3) * 10
    true_points[:, 2] += 10  # 카메라 앞에 위치
    
    # 2D 투영
    pts1 = (P1 @ np.hstack([true_points, np.ones((num_points, 1))]).T).T
    pts1 = pts1[:, :2] / pts1[:, 2:3]
    
    pts2 = (P2 @ np.hstack([true_points, np.ones((num_points, 1))]).T).T
    pts2 = pts2[:, :2] / pts2[:, 2:3]
    
    # 삼각측량
    triangulator = Triangulator()
    result = triangulator.triangulate(P1, P2, pts1, pts2)
    
    print(f"복원된 3D 점: {len(result.points_3d)}개")
    print(f"유효한 점: {np.sum(result.valid_mask)}개")
    print(f"평균 재투영 에러: {result.reprojection_error:.4f} 픽셀")
    
    # Ground truth와 비교
    reconstruction_error = np.linalg.norm(result.points_3d - true_points, axis=1)
    print(f"평균 3D 재구성 에러: {np.mean(reconstruction_error):.4f}")
    print(f"최대 3D 재구성 에러: {np.max(reconstruction_error):.4f}")
