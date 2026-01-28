"""
Feature Detection Module

특징점 검출을 위한 SIFT/ORB 알고리즘 구현.
이미지에서 키포인트와 디스크립터를 추출합니다.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FeatureResult:
    """특징점 검출 결과를 저장하는 데이터 클래스"""
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    algorithm: str
    num_features: int


class FeatureDetector:
    """
    특징점 검출기 클래스
    
    SIFT와 ORB 알고리즘을 지원합니다.
    
    Attributes:
        algorithm: 사용할 알고리즘 ("sift" 또는 "orb")
        detector: OpenCV 특징점 검출기 객체
    
    Example:
        >>> detector = FeatureDetector(algorithm="sift")
        >>> result = detector.detect(image)
        >>> print(f"검출된 특징점: {result.num_features}개")
    """
    
    def __init__(self, algorithm: str = "sift", **kwargs):
        """
        특징점 검출기 초기화
        
        Args:
            algorithm: "sift" 또는 "orb"
            **kwargs: 알고리즘별 추가 파라미터
                - sift: nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold
                - orb: nfeatures, scaleFactor, nlevels, edgeThreshold
        """
        self.algorithm = algorithm.lower()
        self.detector = self._create_detector(**kwargs)
    
    def _create_detector(self, **kwargs):
        """알고리즘에 맞는 검출기 생성"""
        if self.algorithm == "sift":
            return cv2.SIFT_create(
                nfeatures=kwargs.get("nfeatures", 0),
                nOctaveLayers=kwargs.get("nOctaveLayers", 3),
                contrastThreshold=kwargs.get("contrastThreshold", 0.04),
                edgeThreshold=kwargs.get("edgeThreshold", 10),
                sigma=kwargs.get("sigma", 1.6)
            )
        elif self.algorithm == "orb":
            return cv2.ORB_create(
                nfeatures=kwargs.get("nfeatures", 500),
                scaleFactor=kwargs.get("scaleFactor", 1.2),
                nlevels=kwargs.get("nlevels", 8),
                edgeThreshold=kwargs.get("edgeThreshold", 31),
                patchSize=kwargs.get("patchSize", 31)
            )
        else:
            raise ValueError(f"지원하지 않는 알고리즘: {self.algorithm}")
    
    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> FeatureResult:
        """
        이미지에서 특징점을 검출합니다.
        
        Args:
            image: 입력 이미지 (BGR 또는 그레이스케일)
            mask: 검출 영역 마스크 (선택)
        
        Returns:
            FeatureResult: 검출된 키포인트와 디스크립터
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 특징점 검출 및 디스크립터 계산
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)
        
        return FeatureResult(
            keypoints=keypoints,
            descriptors=descriptors if descriptors is not None else np.array([]),
            algorithm=self.algorithm.upper(),
            num_features=len(keypoints)
        )
    
    def detect_and_draw(self, image: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> Tuple[FeatureResult, np.ndarray]:
        """
        특징점을 검출하고 시각화합니다.
        
        Args:
            image: 입력 이미지
            mask: 검출 영역 마스크
        
        Returns:
            Tuple[FeatureResult, np.ndarray]: 검출 결과와 시각화 이미지
        """
        result = self.detect(image, mask)
        
        # 특징점 시각화
        output_image = cv2.drawKeypoints(
            image, 
            result.keypoints, 
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return result, output_image


def compare_algorithms(image: np.ndarray) -> dict:
    """
    SIFT와 ORB 알고리즘의 성능을 비교합니다.
    
    Args:
        image: 입력 이미지
    
    Returns:
        dict: 각 알고리즘의 검출 결과 및 실행 시간
    """
    import time
    results = {}
    
    for algo in ["sift", "orb"]:
        detector = FeatureDetector(algorithm=algo)
        
        start_time = time.time()
        result = detector.detect(image)
        elapsed_time = time.time() - start_time
        
        results[algo] = {
            "num_features": result.num_features,
            "time_ms": elapsed_time * 1000,
            "descriptor_size": result.descriptors.shape[1] if len(result.descriptors) > 0 else 0
        }
    
    return results


if __name__ == "__main__":
    # 단위 테스트
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is not None:
            print("=== 특징점 검출 테스트 ===\n")
            comparison = compare_algorithms(image)
            
            for algo, stats in comparison.items():
                print(f"{algo.upper()}:")
                print(f"  - 검출된 특징점: {stats['num_features']}개")
                print(f"  - 실행 시간: {stats['time_ms']:.2f}ms")
                print(f"  - 디스크립터 크기: {stats['descriptor_size']}")
                print()
        else:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
    else:
        print("사용법: python feature_detection.py <image_path>")
