"""
Bundle Adjustment Module (Simplified)

번들 조정은 모든 카메라 파라미터와 3D 점을 동시에 최적화하여
재투영 에러를 최소화합니다.

이 구현은 scipy.optimize를 사용한 간단한 버전입니다.
실제 대규모 프로젝트에서는 Ceres Solver나 g2o를 권장합니다.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BundleAdjustmentResult:
    """번들 조정 결과"""
    camera_params: np.ndarray   # 최적화된 카메라 파라미터
    points_3d: np.ndarray       # 최적화된 3D 점
    initial_error: float        # 최적화 전 재투영 에러
    final_error: float          # 최적화 후 재투영 에러
    success: bool


class BundleAdjustment:
    """
    번들 조정 클래스
    
    비선형 최소제곱법을 사용하여 카메라 파라미터와 3D 점을
    동시에 최적화합니다.
    
    최소화 대상:
        sum_i sum_j || x_ij - π(C_i, X_j) ||^2
        
    여기서:
        - x_ij: j번째 점의 i번째 카메라에서의 2D 관측
        - π: 투영 함수
        - C_i: i번째 카메라 파라미터
        - X_j: j번째 3D 점
    """
    
    def __init__(self, K: np.ndarray, fix_intrinsics: bool = True):
        """
        번들 조정 초기화
        
        Args:
            K: 카메라 내부 파라미터 (고정 또는 초기값)
            fix_intrinsics: True면 내부 파라미터를 고정
        """
        self.K = K.astype(np.float64)
        self.fix_intrinsics = fix_intrinsics
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
    
    def project(self, points_3d: np.ndarray, 
                camera_params: np.ndarray) -> np.ndarray:
        """
        3D 점을 카메라 파라미터를 사용하여 2D로 투영합니다.
        
        Args:
            points_3d: Nx3 3D 점
            camera_params: 6-vector (rvec(3), tvec(3))
        
        Returns:
            np.ndarray: Nx2 투영된 2D 점
        """
        rvec = camera_params[:3]
        tvec = camera_params[3:6]
        
        # 회전 행렬로 변환
        R, _ = cv2.Rodrigues(rvec)
        
        # 변환 및 투영
        points_cam = (R @ points_3d.T).T + tvec
        
        # 이미지 좌표로 변환
        x = points_cam[:, 0] / points_cam[:, 2]
        y = points_cam[:, 1] / points_cam[:, 2]
        
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        
        return np.column_stack([u, v])
    
    def compute_residuals(self, params: np.ndarray,
                          n_cameras: int,
                          n_points: int,
                          camera_indices: np.ndarray,
                          point_indices: np.ndarray,
                          points_2d: np.ndarray) -> np.ndarray:
        """
        잔차(residuals)를 계산합니다.
        
        Args:
            params: 모든 파라미터 벡터 (카메라 + 3D 점)
            n_cameras: 카메라 수
            n_points: 3D 점 수
            camera_indices: 각 관측에 대한 카메라 인덱스
            point_indices: 각 관측에 대한 점 인덱스
            points_2d: 2D 관측값
        
        Returns:
            np.ndarray: 잔차 벡터
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        
        residuals = []
        
        for cam_idx, pt_idx, obs in zip(camera_indices, point_indices, points_2d):
            point_3d = points_3d[pt_idx].reshape(1, 3)
            projected = self.project(point_3d, camera_params[cam_idx])
            
            residuals.append(obs[0] - projected[0, 0])
            residuals.append(obs[1] - projected[0, 1])
        
        return np.array(residuals)
    
    def bundle_adjustment_sparsity(self, n_cameras: int, n_points: int,
                                   camera_indices: np.ndarray,
                                   point_indices: np.ndarray) -> lil_matrix:
        """
        자코비안 행렬의 희소 구조를 계산합니다.
        
        번들 조정의 자코비안은 매우 희소하므로, 
        이를 활용하여 계산 효율을 높입니다.
        
        Args:
            n_cameras, n_points: 카메라와 점의 수
            camera_indices, point_indices: 관측 인덱스
        
        Returns:
            lil_matrix: 희소 자코비안 구조
        """
        m = len(camera_indices) * 2  # 관측 수 * 2 (x, y)
        n = n_cameras * 6 + n_points * 3  # 파라미터 수
        
        A = lil_matrix((m, n), dtype=int)
        
        for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
            # 카메라 파라미터에 대한 미분
            for s in range(6):
                A[2*i, cam_idx * 6 + s] = 1
                A[2*i + 1, cam_idx * 6 + s] = 1
            
            # 3D 점에 대한 미분
            for s in range(3):
                A[2*i, n_cameras * 6 + pt_idx * 3 + s] = 1
                A[2*i + 1, n_cameras * 6 + pt_idx * 3 + s] = 1
        
        return A
    
    def optimize(self, 
                 camera_params: np.ndarray,
                 points_3d: np.ndarray,
                 camera_indices: np.ndarray,
                 point_indices: np.ndarray,
                 points_2d: np.ndarray,
                 max_iterations: int = 100) -> BundleAdjustmentResult:
        """
        번들 조정을 수행합니다.
        
        Args:
            camera_params: 초기 카메라 파라미터 (N_cameras x 6)
            points_3d: 초기 3D 점 (N_points x 3)
            camera_indices: 각 관측의 카메라 인덱스
            point_indices: 각 관측의 점 인덱스
            points_2d: 2D 관측값 (N_observations x 2)
            max_iterations: 최대 반복 횟수
        
        Returns:
            BundleAdjustmentResult: 최적화 결과
        """
        n_cameras = len(camera_params)
        n_points = len(points_3d)
        
        # 초기 파라미터 벡터 구성
        x0 = np.hstack([camera_params.ravel(), points_3d.ravel()])
        
        # 초기 에러 계산
        initial_residuals = self.compute_residuals(
            x0, n_cameras, n_points,
            camera_indices, point_indices, points_2d
        )
        initial_error = np.sqrt(np.mean(initial_residuals ** 2))
        
        # 희소 자코비안 구조
        A = self.bundle_adjustment_sparsity(
            n_cameras, n_points,
            camera_indices, point_indices
        )
        
        # 최적화 수행
        result = least_squares(
            self.compute_residuals,
            x0,
            jac_sparsity=A,
            verbose=0,
            x_scale='jac',
            ftol=1e-4,
            max_nfev=max_iterations,
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d)
        )
        
        # 결과 추출
        optimized_camera_params = result.x[:n_cameras * 6].reshape((n_cameras, 6))
        optimized_points_3d = result.x[n_cameras * 6:].reshape((n_points, 3))
        
        final_error = np.sqrt(np.mean(result.fun ** 2))
        
        return BundleAdjustmentResult(
            camera_params=optimized_camera_params,
            points_3d=optimized_points_3d,
            initial_error=initial_error,
            final_error=final_error,
            success=result.success
        )


# cv2 import at module level may fail if opencv not installed
try:
    import cv2
except ImportError:
    # Fallback Rodrigues implementation
    def rodrigues_to_rotation(rvec):
        theta = np.linalg.norm(rvec)
        if theta < 1e-10:
            return np.eye(3)
        r = rvec / theta
        K = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


if __name__ == "__main__":
    import cv2
    
    print("=== 번들 조정 테스트 ===\n")
    
    # 간단한 테스트 설정
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 2개의 카메라
    true_camera_params = np.array([
        [0, 0, 0, 0, 0, 0],           # 카메라 1: 원점
        [0.1, 0.05, 0, 2, 0.5, 0.1]   # 카메라 2: 이동 + 약간의 회전
    ])
    
    # 3D 점 생성
    np.random.seed(42)
    n_points = 30
    true_points = np.random.rand(n_points, 3) * 5
    true_points[:, 2] += 8
    
    # 관측 데이터 생성
    ba = BundleAdjustment(K)
    
    camera_indices = []
    point_indices = []
    points_2d = []
    
    for cam_idx in range(2):
        for pt_idx in range(n_points):
            proj = ba.project(
                true_points[pt_idx:pt_idx+1], 
                true_camera_params[cam_idx]
            )
            
            # 이미지 범위 내인 경우만 추가
            if 0 <= proj[0, 0] < 640 and 0 <= proj[0, 1] < 480:
                camera_indices.append(cam_idx)
                point_indices.append(pt_idx)
                # 약간의 노이즈 추가
                points_2d.append(proj[0] + np.random.randn(2) * 0.5)
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    
    print(f"카메라 수: 2")
    print(f"3D 점 수: {n_points}")
    print(f"관측 수: {len(points_2d)}")
    
    # 초기 추정 (노이즈 추가)
    init_camera_params = true_camera_params + np.random.randn(2, 6) * 0.1
    init_points = true_points + np.random.randn(n_points, 3) * 0.3
    
    # 번들 조정 수행
    result = ba.optimize(
        init_camera_params, init_points,
        camera_indices, point_indices, points_2d
    )
    
    print(f"\n최적화 성공: {result.success}")
    print(f"초기 재투영 에러: {result.initial_error:.4f} 픽셀")
    print(f"최종 재투영 에러: {result.final_error:.4f} 픽셀")
    print(f"에러 감소율: {(1 - result.final_error/result.initial_error)*100:.1f}%")
