"""
Camera Pose Estimation Module

에센셜 행렬로부터 카메라의 상대적 포즈(R, t)를 추정합니다.
에센셜 행렬은 E = K'^T F K 로 기초 행렬에서 계산되며,
SVD 분해를 통해 회전 행렬(R)과 평행이동 벡터(t)로 분해됩니다.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CameraPose:
    """카메라 포즈 데이터 클래스"""
    R: np.ndarray       # 3x3 회전 행렬
    t: np.ndarray       # 3x1 평행이동 벡터
    
    def to_projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """
        카메라 내부 파라미터를 결합하여 투영 행렬을 생성합니다.
        
        P = K [R | t]
        
        Args:
            K: 3x3 카메라 내부 파라미터 행렬
        
        Returns:
            np.ndarray: 3x4 투영 행렬
        """
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        P = K @ Rt
        return P


class CameraPoseEstimator:
    """
    카메라 포즈 추정 클래스
    
    에센셜 행렬을 분해하여 두 카메라 간의 상대적 포즈를 계산합니다.
    
    Attributes:
        K: 카메라 내부 파라미터 행렬
    """
    
    def __init__(self, K: np.ndarray):
        """
        카메라 포즈 추정기 초기화
        
        Args:
            K: 3x3 카메라 내부 파라미터 행렬
               [[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]]
        """
        self.K = K.astype(np.float64)
    
    def estimate_essential_matrix(self, pts1: np.ndarray, 
                                  pts2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        대응점으로부터 에센셜 행렬을 직접 계산합니다.
        
        Args:
            pts1, pts2: 대응점 (Nx2)
        
        Returns:
            Tuple[E, mask]: 에센셜 행렬과 인라이어 마스크
        """
        if len(pts1) < 5:
            return None, None
        
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        return E, mask
    
    def decompose_essential_matrix(self, E: np.ndarray, 
                                   pts1: np.ndarray,
                                   pts2: np.ndarray) -> Optional[CameraPose]:
        """
        에센셜 행렬을 R, t로 분해합니다.
        
        에센셜 행렬은 4가지 가능한 (R, t) 조합으로 분해되지만,
        cheirality check를 통해 올바른 조합을 선택합니다.
        
        Args:
            E: 3x3 에센셜 행렬
            pts1, pts2: 대응점 (cheirality check용)
        
        Returns:
            CameraPose: 추정된 카메라 포즈
        """
        # recoverPose는 내부적으로 cheirality check를 수행
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        return CameraPose(R=R, t=t)
    
    def estimate_pose(self, pts1: np.ndarray, 
                      pts2: np.ndarray) -> Optional[CameraPose]:
        """
        대응점으로부터 카메라 포즈를 한 번에 추정합니다.
        
        Args:
            pts1, pts2: 대응점 (Nx2)
        
        Returns:
            CameraPose: 추정된 카메라 포즈
        """
        E, mask = self.estimate_essential_matrix(pts1, pts2)
        
        if E is None:
            return None
        
        # 인라이어만 사용
        if mask is not None:
            mask = mask.ravel().astype(bool)
            pts1_inlier = pts1[mask]
            pts2_inlier = pts2[mask]
        else:
            pts1_inlier = pts1
            pts2_inlier = pts2
        
        return self.decompose_essential_matrix(E, pts1_inlier, pts2_inlier)
    
    def get_projection_matrices(self, pose: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
        """
        첫 번째 카메라(원점)와 두 번째 카메라의 투영 행렬을 반환합니다.
        
        Args:
            pose: 두 번째 카메라의 상대적 포즈
        
        Returns:
            Tuple[P1, P2]: 두 카메라의 3x4 투영 행렬
        """
        # 첫 번째 카메라: 원점, 단위 회전
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # 두 번째 카메라
        P2 = pose.to_projection_matrix(self.K)
        
        return P1, P2


def decompose_essential_matrix_all(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    에센셜 행렬을 가능한 모든 (R, t) 조합으로 분해합니다.
    
    SVD 분해: E = U Σ V^T
    가능한 조합:
        R1 = U W V^T,  t1 = +u3
        R2 = U W V^T,  t2 = -u3
        R3 = U W^T V^T, t3 = +u3
        R4 = U W^T V^T, t4 = -u3
    
    Args:
        E: 3x3 에센셜 행렬
    
    Returns:
        List[Tuple[R, t]]: 4가지 (R, t) 조합
    """
    U, S, Vt = np.linalg.svd(E)
    
    # Rotation matrix constraint: det(R) = 1
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2].reshape(3, 1)
    
    # 4가지 조합
    poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    
    return poses


def cheirality_check(R: np.ndarray, t: np.ndarray,
                     K: np.ndarray,
                     pts1: np.ndarray, pts2: np.ndarray) -> int:
    """
    Cheirality check: 3D 점이 두 카메라 앞에 있는지 확인합니다.
    
    Args:
        R, t: 카메라 포즈
        K: 카메라 내부 파라미터
        pts1, pts2: 대응점
    
    Returns:
        int: 양쪽 카메라 앞에 있는 점의 수
    """
    # 투영 행렬
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    
    # 삼각 측량
    pts1_h = pts1.reshape(-1, 1, 2)
    pts2_h = pts2.reshape(-1, 1, 2)
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d[:3] / points_4d[3]
    
    # 카메라 1 앞에 있는지 (Z > 0)
    in_front_1 = points_3d[2, :] > 0
    
    # 카메라 2 앞에 있는지
    points_cam2 = R @ points_3d + t
    in_front_2 = points_cam2[2, :] > 0
    
    # 둘 다 앞에 있는 점의 수
    return np.sum(in_front_1 & in_front_2)


if __name__ == "__main__":
    print("=== 카메라 포즈 추정 테스트 ===\n")
    
    # 테스트용 카메라 파라미터
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 가상 3D 점
    np.random.seed(42)
    num_points = 100
    X = np.random.rand(num_points, 3) * 10
    X[:, 2] += 10  # 카메라 앞에 위치
    
    # Ground truth 포즈
    true_R = cv2.Rodrigues(np.array([0.1, 0.2, 0.05]))[0]
    true_t = np.array([[2.0], [0.5], [0.1]])
    
    # 투영
    X_h = np.hstack([X, np.ones((num_points, 1))])
    
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([true_R, true_t])
    
    pts1 = (P1 @ X_h.T).T
    pts1 = pts1[:, :2] / pts1[:, 2:3]
    
    pts2 = (P2 @ X_h.T).T
    pts2 = pts2[:, :2] / pts2[:, 2:3]
    
    # 포즈 추정
    estimator = CameraPoseEstimator(K)
    pose = estimator.estimate_pose(pts1, pts2)
    
    if pose is not None:
        print("추정된 회전 행렬 R:")
        print(pose.R)
        print(f"\n실제 R과의 차이 (Frobenius norm): {np.linalg.norm(pose.R - true_R):.6f}")
        
        print("\n추정된 평행이동 벡터 t:")
        print(pose.t.flatten())
        
        # t는 스케일까지만 복원 가능
        true_t_normalized = true_t / np.linalg.norm(true_t)
        estimated_t_normalized = pose.t / np.linalg.norm(pose.t)
        print(f"\n정규화된 t 차이: {np.linalg.norm(estimated_t_normalized - true_t_normalized):.6f}")
    else:
        print("포즈 추정 실패")
