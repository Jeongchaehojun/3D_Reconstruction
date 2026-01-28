"""
Structure from Motion Pipeline

전체 SfM 파이프라인을 통합하는 메인 모듈입니다.

파이프라인 흐름:
1. 이미지 로드
2. 특징점 검출 (SIFT/ORB)
3. 특징점 매칭
4. 기초 행렬/에센셜 행렬 계산
5. 카메라 포즈 추정
6. 삼각측량으로 3D 점 복원
7. (선택) 번들 조정으로 정제
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from .feature_detection import FeatureDetector, FeatureResult
from .feature_matching import FeatureMatcher, MatchResult, extract_matched_points
from .fundamental_matrix import FundamentalMatrixEstimator
from .camera_pose import CameraPoseEstimator, CameraPose
from .triangulation import Triangulator, TriangulationResult


@dataclass
class SfMResult:
    """SfM 파이프라인 결과"""
    points_3d: np.ndarray               # Nx3 포인트 클라우드
    colors: np.ndarray                  # Nx3 RGB 컬러
    camera_poses: List[CameraPose]      # 카메라 포즈 리스트
    reprojection_error: float           # 평균 재투영 에러
    num_images: int                     # 사용된 이미지 수
    num_points: int                     # 복원된 3D 점 수


@dataclass
class ImageData:
    """이미지 및 관련 데이터"""
    image: np.ndarray
    keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    descriptors: np.ndarray = None
    path: str = ""


class SfMPipeline:
    """
    Structure from Motion 파이프라인
    
    여러 이미지로부터 3D 포인트 클라우드를 생성합니다.
    
    Attributes:
        K: 카메라 내부 파라미터 행렬
        feature_algorithm: 특징점 알고리즘 ("sift" 또는 "orb")
        matcher_type: 매칭 알고리즘 ("bf" 또는 "flann")
    
    Example:
        >>> pipeline = SfMPipeline(K)
        >>> result = pipeline.run(image_paths)
        >>> print(f"복원된 3D 점: {result.num_points}개")
    """
    
    def __init__(self, 
                 K: np.ndarray,
                 feature_algorithm: str = "sift",
                 matcher_type: str = "bf",
                 min_matches: int = 30):
        """
        SfM 파이프라인 초기화
        
        Args:
            K: 3x3 카메라 내부 파라미터 행렬
            feature_algorithm: "sift" 또는 "orb"
            matcher_type: "bf" 또는 "flann"
            min_matches: 최소 필요 매칭 수
        """
        self.K = K.astype(np.float64)
        self.feature_algorithm = feature_algorithm
        self.matcher_type = matcher_type
        self.min_matches = min_matches
        
        # 모듈 초기화
        self.detector = FeatureDetector(algorithm=feature_algorithm)
        self.matcher = FeatureMatcher(
            matcher_type=matcher_type,
            descriptor_type=feature_algorithm
        )
        self.fundamental_estimator = FundamentalMatrixEstimator()
        self.pose_estimator = CameraPoseEstimator(K)
        self.triangulator = Triangulator()
        
        # 데이터 저장
        self.images: List[ImageData] = []
        self.matches: Dict[Tuple[int, int], MatchResult] = {}
    
    def load_images(self, image_paths: List[str]) -> int:
        """
        이미지를 로드합니다.
        
        Args:
            image_paths: 이미지 파일 경로 리스트
        
        Returns:
            int: 로드된 이미지 수
        """
        self.images = []
        
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is not None:
                self.images.append(ImageData(image=image, path=str(path)))
                print(f"로드: {path}")
            else:
                print(f"경고: 이미지 로드 실패 - {path}")
        
        return len(self.images)
    
    def detect_features(self) -> None:
        """모든 이미지에서 특징점을 검출합니다."""
        print(f"\n=== 특징점 검출 ({self.feature_algorithm.upper()}) ===")
        
        for i, img_data in enumerate(self.images):
            result = self.detector.detect(img_data.image)
            img_data.keypoints = result.keypoints
            img_data.descriptors = result.descriptors
            print(f"  이미지 {i+1}: {result.num_features}개 특징점")
    
    def match_features(self) -> Dict[Tuple[int, int], MatchResult]:
        """모든 이미지 쌍에 대해 특징점을 매칭합니다."""
        print(f"\n=== 특징점 매칭 ===")
        
        self.matches = {}
        n = len(self.images)
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                result = self.matcher.match(
                    self.images[i].descriptors,
                    self.images[j].descriptors
                )
                
                if result.num_good >= self.min_matches:
                    self.matches[(i, j)] = result
                    print(f"  이미지 {i+1}-{j+1}: {result.num_good}개 매칭 "
                          f"(비율: {result.inlier_ratio:.2%})")
                else:
                    print(f"  이미지 {i+1}-{j+1}: 매칭 부족 ({result.num_good}개)")
        
        return self.matches
    
    def reconstruct_two_view(self, idx1: int, idx2: int) -> Optional[TriangulationResult]:
        """
        두 이미지로부터 초기 3D 재구성을 수행합니다.
        
        Args:
            idx1, idx2: 이미지 인덱스
        
        Returns:
            TriangulationResult: 3D 재구성 결과
        """
        key = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
        
        if key not in self.matches:
            print(f"경고: 이미지 {idx1}-{idx2} 사이에 매칭이 없습니다.")
            return None
        
        match_result = self.matches[key]
        
        # 매칭된 점 추출
        pts1, pts2 = extract_matched_points(
            self.images[idx1].keypoints,
            self.images[idx2].keypoints,
            match_result.good_matches
        )
        
        # 카메라 포즈 추정
        pose = self.pose_estimator.estimate_pose(pts1, pts2)
        
        if pose is None:
            print("포즈 추정 실패")
            return None
        
        # 투영 행렬 계산
        P1, P2 = self.pose_estimator.get_projection_matrices(pose)
        
        # 삼각측량
        result = self.triangulator.triangulate(P1, P2, pts1, pts2)
        
        return result
    
    def run(self, image_paths: List[str]) -> Optional[SfMResult]:
        """
        전체 SfM 파이프라인을 실행합니다.
        
        Args:
            image_paths: 이미지 파일 경로 리스트
        
        Returns:
            SfMResult: SfM 결과
        """
        print("=" * 50)
        print("Structure from Motion 파이프라인 시작")
        print("=" * 50)
        
        # 1. 이미지 로드
        n_loaded = self.load_images(image_paths)
        if n_loaded < 2:
            print("오류: 최소 2개의 이미지가 필요합니다.")
            return None
        
        # 2. 특징점 검출
        self.detect_features()
        
        # 3. 특징점 매칭
        self.match_features()
        
        if not self.matches:
            print("오류: 유효한 매칭이 없습니다.")
            return None
        
        # 4. 초기 재구성 (처음 두 이미지 사용)
        print(f"\n=== 초기 3D 재구성 ===")
        
        # 가장 매칭이 많은 쌍 선택
        best_pair = max(self.matches.keys(), 
                        key=lambda k: self.matches[k].num_good)
        
        print(f"  선택된 이미지 쌍: {best_pair[0]+1}, {best_pair[1]+1}")
        
        result = self.reconstruct_two_view(best_pair[0], best_pair[1])
        
        if result is None:
            return None
        
        # 3D 점에 색상 할당
        match_result = self.matches[best_pair]
        colors = self._extract_colors(best_pair[0], match_result.good_matches)
        
        # 유효한 점만 필터링
        valid_points = result.points_3d[result.valid_mask]
        valid_colors = colors[result.valid_mask]
        
        print(f"\n=== 결과 ===")
        print(f"  복원된 3D 점: {len(valid_points)}개")
        print(f"  재투영 에러: {result.reprojection_error:.4f} 픽셀")
        
        # 결과 반환
        return SfMResult(
            points_3d=valid_points,
            colors=valid_colors,
            camera_poses=[],  # TODO: 모든 카메라 포즈 저장
            reprojection_error=result.reprojection_error,
            num_images=n_loaded,
            num_points=len(valid_points)
        )
    
    def _extract_colors(self, img_idx: int, 
                        matches: List[cv2.DMatch]) -> np.ndarray:
        """매칭된 점의 색상을 추출합니다."""
        image = self.images[img_idx].image
        keypoints = self.images[img_idx].keypoints
        
        colors = []
        for match in matches:
            x, y = map(int, keypoints[match.queryIdx].pt)
            x = min(max(x, 0), image.shape[1] - 1)
            y = min(max(y, 0), image.shape[0] - 1)
            
            # BGR -> RGB
            bgr = image[y, x]
            colors.append([bgr[2], bgr[1], bgr[0]])
        
        return np.array(colors, dtype=np.uint8)
    
    def save_point_cloud(self, result: SfMResult, 
                         output_path: str,
                         format: str = "ply") -> None:
        """
        포인트 클라우드를 파일로 저장합니다.
        
        Args:
            result: SfM 결과
            output_path: 출력 파일 경로
            format: "ply" 또는 "pcd"
        """
        if format.lower() == "ply":
            self._save_ply(result, output_path)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def _save_ply(self, result: SfMResult, output_path: str) -> None:
        """PLY 형식으로 저장"""
        with open(output_path, 'w') as f:
            # 헤더
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {result.num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # 데이터
            for point, color in zip(result.points_3d, result.colors):
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                        f"{color[0]} {color[1]} {color[2]}\n")
        
        print(f"포인트 클라우드 저장됨: {output_path}")


def create_default_camera_matrix(image_width: int, 
                                  image_height: int,
                                  fov_degrees: float = 60) -> np.ndarray:
    """
    기본 카메라 내부 파라미터를 생성합니다.
    
    Args:
        image_width, image_height: 이미지 크기
        fov_degrees: 수평 시야각 (도)
    
    Returns:
        np.ndarray: 3x3 카메라 행렬
    """
    fov_rad = np.radians(fov_degrees)
    fx = image_width / (2 * np.tan(fov_rad / 2))
    fy = fx  # 정사각형 픽셀 가정
    cx = image_width / 2
    cy = image_height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("사용법: python sfm_pipeline.py <image1> <image2> [image3] ...")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    # 첫 번째 이미지에서 크기 확인
    test_img = cv2.imread(image_paths[0])
    if test_img is None:
        print(f"오류: 첫 번째 이미지를 불러올 수 없습니다: {image_paths[0]}")
        sys.exit(1)
    
    h, w = test_img.shape[:2]
    
    # 기본 카메라 행렬 생성
    K = create_default_camera_matrix(w, h)
    print(f"카메라 행렬:\n{K}\n")
    
    # SfM 실행
    pipeline = SfMPipeline(K)
    result = pipeline.run(image_paths)
    
    if result is not None:
        # 결과 저장
        output_path = "output_pointcloud.ply"
        pipeline.save_point_cloud(result, output_path)
