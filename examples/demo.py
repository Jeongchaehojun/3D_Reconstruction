"""
SfM Demo Script

Structure from Motion 데모 스크립트입니다.
간단한 예제 이미지로 3D 재구성을 시연합니다.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sfm_pipeline import SfMPipeline, create_default_camera_matrix
from visualization.point_cloud_viewer import PointCloudVisualizer


def create_synthetic_scene():
    """
    합성 테스트 데이터를 생성합니다.
    
    실제 이미지 없이 알고리즘을 테스트할 수 있습니다.
    """
    print("=== 합성 테스트 데이터 생성 ===\n")
    
    np.random.seed(42)
    
    # 카메라 설정
    image_size = (640, 480)
    K = create_default_camera_matrix(image_size[0], image_size[1], fov_degrees=60)
    
    # 3D 점 생성 (큐브 형태)
    n_points = 100
    points_3d = np.random.rand(n_points, 3) * 4 - 2  # -2 ~ 2
    points_3d[:, 2] += 8  # 카메라 앞에 위치
    
    # 색상
    colors = np.random.randint(0, 255, (n_points, 3), dtype=np.uint8)
    
    print(f"생성된 3D 점: {n_points}개")
    print(f"카메라 행렬:\n{K}")
    
    return points_3d, colors, K


def demo_feature_detection():
    """특징점 검출 데모"""
    from src.feature_detection import FeatureDetector, compare_algorithms
    
    print("\n" + "="*50)
    print("데모 1: 특징점 검출 비교")
    print("="*50 + "\n")
    
    # 테스트 이미지 생성 (체커보드 패턴)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 체커보드 패턴
    square_size = 50
    for i in range(0, 480, square_size):
        for j in range(0, 640, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                image[i:i+square_size, j:j+square_size] = [255, 255, 255]
    
    # 노이즈 추가
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # 알고리즘 비교
    results = compare_algorithms(image)
    
    for algo, stats in results.items():
        print(f"{algo.upper()}:")
        print(f"  검출된 특징점: {stats['num_features']}개")
        print(f"  실행 시간: {stats['time_ms']:.2f}ms")
        print(f"  디스크립터 크기: {stats['descriptor_size']}")
        print()
    
    # 시각화 저장
    detector = FeatureDetector(algorithm="sift")
    _, vis_image = detector.detect_and_draw(image)
    
    output_path = project_root / "tests" / "test_images" / "feature_demo.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)
    print(f"시각화 저장됨: {output_path}")


def demo_fundamental_matrix():
    """기초 행렬 데모"""
    from src.fundamental_matrix import FundamentalMatrixEstimator, compute_essential_matrix
    
    print("\n" + "="*50)
    print("데모 2: 기초 행렬 계산")
    print("="*50 + "\n")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 3D 점
    n_points = 100
    X = np.random.rand(n_points, 3) * 10
    X[:, 2] += 15
    
    # 두 카메라 설정
    R1 = np.eye(3)
    t1 = np.zeros(3)
    R2 = cv2.Rodrigues(np.array([0.1, 0.15, 0.05]))[0]
    t2 = np.array([3.0, 0.5, 0.2])
    
    # 투영
    X_h = np.hstack([X, np.ones((n_points, 1))])
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    
    pts1 = (P1 @ X_h.T).T
    pts1 = pts1[:, :2] / pts1[:, 2:3]
    
    pts2 = (P2 @ X_h.T).T
    pts2 = pts2[:, :2] / pts2[:, 2:3]
    
    # 기초 행렬 계산
    estimator = FundamentalMatrixEstimator(method="ransac")
    result = estimator.estimate(pts1, pts2)
    
    if result is not None:
        print("기초 행렬 F:")
        print(result.F)
        print(f"\n인라이어: {result.num_inliers}/{n_points} ({result.inlier_ratio:.2%})")
        
        # 에피폴라 제약식 검증
        mean_error, _ = estimator.verify_epipolar_constraint(pts1, pts2, result.F)
        print(f"평균 에피폴라 에러: {mean_error:.6f}")
        
        # 에센셜 행렬 계산
        E = compute_essential_matrix(result.F, K)
        print(f"\n에센셜 행렬 E:")
        print(E)
    else:
        print("기초 행렬 계산 실패")


def demo_triangulation():
    """삼각측량 데모"""
    from src.camera_pose import CameraPoseEstimator
    from src.triangulation import Triangulator
    
    print("\n" + "="*50)
    print("데모 3: 삼각측량")
    print("="*50 + "\n")
    
    # 테스트 데이터
    np.random.seed(42)
    
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 실제 3D 점
    n_points = 50
    true_points = np.random.rand(n_points, 3) * 5
    true_points[:, 2] += 10
    
    # 카메라 설정
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    R2 = np.eye(3)
    t2 = np.array([[2.0], [0.0], [0.0]])
    P2 = K @ np.hstack([R2, t2])
    
    # 2D 투영
    pts1 = (P1 @ np.hstack([true_points, np.ones((n_points, 1))]).T).T
    pts1 = pts1[:, :2] / pts1[:, 2:3]
    
    pts2 = (P2 @ np.hstack([true_points, np.ones((n_points, 1))]).T).T
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


def demo_visualization():
    """시각화 데모"""
    print("\n" + "="*50)
    print("데모 4: 포인트 클라우드 시각화")
    print("="*50 + "\n")
    
    points, colors, _ = create_synthetic_scene()
    
    # 스크린샷 저장
    viz = PointCloudVisualizer(backend="matplotlib")
    
    output_path = project_root / "docs" / "images" / "demo_pointcloud.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    viz.save_screenshot(points, colors, str(output_path))
    
    print("시각화 완료!")
    print("인터랙티브 시각화를 보려면 Open3D를 설치하세요:")
    print("  pip install open3d")


def main():
    """모든 데모 실행"""
    print("="*60)
    print("Structure from Motion (SfM) 데모")
    print("="*60)
    
    demo_feature_detection()
    demo_fundamental_matrix()
    demo_triangulation()
    demo_visualization()
    
    print("\n" + "="*60)
    print("모든 데모 완료!")
    print("="*60)
    print("\n실제 이미지로 SfM을 실행하려면:")
    print("  python -m src.sfm_pipeline image1.jpg image2.jpg image3.jpg")


if __name__ == "__main__":
    main()
