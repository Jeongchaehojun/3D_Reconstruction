"""
SfM 테스트 모듈

각 모듈의 기능을 테스트합니다.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import unittest

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFeatureDetection(unittest.TestCase):
    """특징점 검출 테스트"""
    
    def setUp(self):
        """테스트 이미지 생성"""
        # 텍스처가 풍부한 이미지 (체커보드)
        self.rich_texture = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(0, 480, 40):
            for j in range(0, 640, 40):
                if (i // 40 + j // 40) % 2 == 0:
                    self.rich_texture[i:i+40, j:j+40] = [255, 255, 255]
        
        # 노이즈 추가
        noise = np.random.randint(0, 30, self.rich_texture.shape, dtype=np.uint8)
        self.rich_texture = cv2.add(self.rich_texture, noise)
        
        # 텍스처가 없는 이미지 (단색)
        self.no_texture = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    def test_sift_detection(self):
        """SIFT 검출 테스트"""
        from src.feature_detection import FeatureDetector
        
        detector = FeatureDetector(algorithm="sift")
        result = detector.detect(self.rich_texture)
        
        # 충분한 특징점이 검출되어야 함
        self.assertGreater(result.num_features, 100)
        self.assertEqual(result.algorithm, "SIFT")
        self.assertIsNotNone(result.descriptors)
    
    def test_orb_detection(self):
        """ORB 검출 테스트"""
        from src.feature_detection import FeatureDetector
        
        detector = FeatureDetector(algorithm="orb", nfeatures=500)
        result = detector.detect(self.rich_texture)
        
        self.assertGreater(result.num_features, 0)
        self.assertEqual(result.algorithm, "ORB")
    
    def test_no_texture_detection(self):
        """텍스처 없는 이미지 테스트"""
        from src.feature_detection import FeatureDetector
        
        detector = FeatureDetector(algorithm="sift")
        result = detector.detect(self.no_texture)
        
        # 특징점이 거의 없어야 함
        self.assertLess(result.num_features, 50)


class TestFeatureMatching(unittest.TestCase):
    """특징점 매칭 테스트"""
    
    def setUp(self):
        """테스트 데이터 생성"""
        from src.feature_detection import FeatureDetector
        
        # 유사한 두 이미지 생성
        base_image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(0, 480, 40):
            for j in range(0, 640, 40):
                if (i // 40 + j // 40) % 2 == 0:
                    base_image[i:i+40, j:j+40] = [255, 255, 255]
        
        noise1 = np.random.randint(0, 30, base_image.shape, dtype=np.uint8)
        noise2 = np.random.randint(0, 30, base_image.shape, dtype=np.uint8)
        
        self.image1 = cv2.add(base_image, noise1)
        self.image2 = cv2.add(base_image, noise2)
        
        # 특징점 검출
        detector = FeatureDetector(algorithm="sift")
        result1 = detector.detect(self.image1)
        result2 = detector.detect(self.image2)
        
        self.kp1 = result1.keypoints
        self.kp2 = result2.keypoints
        self.desc1 = result1.descriptors
        self.desc2 = result2.descriptors
    
    def test_bf_matching(self):
        """BFMatcher 테스트"""
        from src.feature_matching import FeatureMatcher
        
        matcher = FeatureMatcher(matcher_type="bf", descriptor_type="sift")
        result = matcher.match(self.desc1, self.desc2)
        
        # 유사한 이미지이므로 많은 매칭이 있어야 함
        self.assertGreater(result.num_good, 50)
        self.assertGreater(result.inlier_ratio, 0.05)
    
    def test_empty_descriptors(self):
        """빈 디스크립터 테스트"""
        from src.feature_matching import FeatureMatcher
        
        matcher = FeatureMatcher()
        result = matcher.match(np.array([]), self.desc2)
        
        self.assertEqual(result.num_good, 0)


class TestFundamentalMatrix(unittest.TestCase):
    """기초 행렬 테스트"""
    
    def setUp(self):
        """테스트 데이터 생성"""
        np.random.seed(42)
        
        self.K = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 3D 점
        n_points = 100
        self.X = np.random.rand(n_points, 3) * 10
        self.X[:, 2] += 15
        
        # 카메라 설정
        R1 = np.eye(3)
        t1 = np.zeros(3)
        R2 = cv2.Rodrigues(np.array([0.1, 0.15, 0.05]))[0]
        t2 = np.array([3.0, 0.5, 0.2])
        
        # 투영
        X_h = np.hstack([self.X, np.ones((n_points, 1))])
        P1 = self.K @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = self.K @ np.hstack([R2, t2.reshape(3, 1)])
        
        self.pts1 = (P1 @ X_h.T).T
        self.pts1 = self.pts1[:, :2] / self.pts1[:, 2:3]
        
        self.pts2 = (P2 @ X_h.T).T
        self.pts2 = self.pts2[:, :2] / self.pts2[:, 2:3]
    
    def test_fundamental_matrix_estimation(self):
        """기초 행렬 추정 테스트"""
        from src.fundamental_matrix import FundamentalMatrixEstimator
        
        estimator = FundamentalMatrixEstimator(method="ransac")
        result = estimator.estimate(self.pts1, self.pts2)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.F.shape, (3, 3))
        self.assertGreater(result.inlier_ratio, 0.8)
    
    def test_epipolar_constraint(self):
        """에피폴라 제약식 테스트"""
        from src.fundamental_matrix import FundamentalMatrixEstimator
        
        estimator = FundamentalMatrixEstimator()
        result = estimator.estimate(self.pts1, self.pts2)
        
        mean_error, _ = estimator.verify_epipolar_constraint(
            self.pts1, self.pts2, result.F
        )
        
        # 에러가 충분히 작아야 함
        self.assertLess(mean_error, 1.0)


class TestTriangulation(unittest.TestCase):
    """삼각측량 테스트"""
    
    def test_triangulation_accuracy(self):
        """삼각측량 정확도 테스트"""
        from src.triangulation import Triangulator
        
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
        
        # 재투영 에러가 충분히 작아야 함
        self.assertLess(result.reprojection_error, 1.0)
        
        # 3D 점이 정확하게 복원되어야 함
        reconstruction_error = np.linalg.norm(result.points_3d - true_points, axis=1)
        self.assertLess(np.mean(reconstruction_error), 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
