"""
Feature Matching Module

두 이미지 간의 특징점 매칭을 수행합니다.
BFMatcher와 FLANN 기반 매칭을 지원하며, Lowe's ratio test를 통해
좋은 매칭만 필터링합니다.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchResult:
    """매칭 결과를 저장하는 데이터 클래스"""
    matches: List[cv2.DMatch]
    good_matches: List[cv2.DMatch]
    inlier_ratio: float
    num_total: int
    num_good: int


class FeatureMatcher:
    """
    특징점 매칭 클래스
    
    BFMatcher(Brute-Force)와 FLANN 기반 매칭을 지원합니다.
    
    알고리즘의 정확도는 정답 매칭 개수와 전체 매칭 시도 횟수 사이의 비율로 계산됩니다.
    
    Attributes:
        matcher_type: "bf" 또는 "flann"
        matcher: OpenCV 매처 객체
        ratio_threshold: Lowe's ratio test 임계값
    """
    
    def __init__(self, matcher_type: str = "bf", 
                 descriptor_type: str = "sift",
                 ratio_threshold: float = 0.75):
        """
        매처 초기화
        
        Args:
            matcher_type: "bf" (Brute-Force) 또는 "flann"
            descriptor_type: "sift" 또는 "orb" (디스크립터 타입에 맞는 거리 계산법 선택)
            ratio_threshold: Lowe's ratio test 임계값 (기본값 0.75)
        """
        self.matcher_type = matcher_type.lower()
        self.descriptor_type = descriptor_type.lower()
        self.ratio_threshold = ratio_threshold
        self.matcher = self._create_matcher()
    
    def _create_matcher(self):
        """매처 타입에 맞는 매처 생성"""
        # 거리 계산 방식 선택
        if self.descriptor_type == "orb":
            norm_type = cv2.NORM_HAMMING
        else:  # SIFT, SURF 등
            norm_type = cv2.NORM_L2
        
        if self.matcher_type == "bf":
            return cv2.BFMatcher(norm_type, crossCheck=False)
        
        elif self.matcher_type == "flann":
            if self.descriptor_type == "orb":
                # ORB용 FLANN 설정
                index_params = dict(
                    algorithm=6,  # FLANN_INDEX_LSH
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                )
            else:
                # SIFT용 FLANN 설정
                index_params = dict(
                    algorithm=1,  # FLANN_INDEX_KDTREE
                    trees=5
                )
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        
        else:
            raise ValueError(f"지원하지 않는 매처 타입: {self.matcher_type}")
    
    def match(self, descriptors1: np.ndarray, 
              descriptors2: np.ndarray) -> MatchResult:
        """
        두 디스크립터 집합 간의 매칭을 수행합니다.
        
        Args:
            descriptors1: 첫 번째 이미지의 디스크립터
            descriptors2: 두 번째 이미지의 디스크립터
        
        Returns:
            MatchResult: 매칭 결과
        """
        if descriptors1 is None or descriptors2 is None:
            return MatchResult(
                matches=[],
                good_matches=[],
                inlier_ratio=0.0,
                num_total=0,
                num_good=0
            )
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return MatchResult(
                matches=[],
                good_matches=[],
                inlier_ratio=0.0,
                num_total=0,
                num_good=0
            )
        
        # FLANN에서 float32 필요
        if self.matcher_type == "flann" and self.descriptor_type == "sift":
            descriptors1 = descriptors1.astype(np.float32)
            descriptors2 = descriptors2.astype(np.float32)
        
        # kNN 매칭 (k=2)
        try:
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        except cv2.error:
            return MatchResult(
                matches=[],
                good_matches=[],
                inlier_ratio=0.0,
                num_total=0,
                num_good=0
            )
        
        # Lowe's ratio test
        good_matches = []
        all_matches = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                all_matches.append(m)
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # 인라이어 비율 계산
        inlier_ratio = len(good_matches) / len(all_matches) if all_matches else 0.0
        
        return MatchResult(
            matches=all_matches,
            good_matches=good_matches,
            inlier_ratio=inlier_ratio,
            num_total=len(all_matches),
            num_good=len(good_matches)
        )
    
    def match_and_draw(self, 
                       image1: np.ndarray, keypoints1: List[cv2.KeyPoint],
                       image2: np.ndarray, keypoints2: List[cv2.KeyPoint],
                       descriptors1: np.ndarray, descriptors2: np.ndarray,
                       max_matches: int = 50) -> Tuple[MatchResult, np.ndarray]:
        """
        매칭을 수행하고 결과를 시각화합니다.
        
        Args:
            image1, image2: 입력 이미지
            keypoints1, keypoints2: 각 이미지의 키포인트
            descriptors1, descriptors2: 각 이미지의 디스크립터
            max_matches: 시각화할 최대 매칭 수
        
        Returns:
            Tuple[MatchResult, np.ndarray]: 매칭 결과와 시각화 이미지
        """
        result = self.match(descriptors1, descriptors2)
        
        # 거리 기준으로 정렬하여 상위 매칭만 시각화
        sorted_matches = sorted(result.good_matches, key=lambda x: x.distance)
        display_matches = sorted_matches[:max_matches]
        
        # 매칭 시각화
        match_image = cv2.drawMatches(
            image1, keypoints1,
            image2, keypoints2,
            display_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return result, match_image


def extract_matched_points(keypoints1: List[cv2.KeyPoint],
                           keypoints2: List[cv2.KeyPoint],
                           matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
    """
    매칭된 키포인트에서 좌표를 추출합니다.
    
    Args:
        keypoints1, keypoints2: 각 이미지의 키포인트
        matches: 매칭 결과
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 매칭된 점들의 좌표 (pts1, pts2)
    """
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    return pts1, pts2


if __name__ == "__main__":
    import sys
    from feature_detection import FeatureDetector
    
    if len(sys.argv) >= 3:
        image1 = cv2.imread(sys.argv[1])
        image2 = cv2.imread(sys.argv[2])
        
        if image1 is not None and image2 is not None:
            print("=== 특징점 매칭 테스트 ===\n")
            
            # 특징점 검출
            detector = FeatureDetector(algorithm="sift")
            result1 = detector.detect(image1)
            result2 = detector.detect(image2)
            
            print(f"이미지 1: {result1.num_features}개 특징점")
            print(f"이미지 2: {result2.num_features}개 특징점\n")
            
            # 매칭
            matcher = FeatureMatcher(matcher_type="bf", descriptor_type="sift")
            match_result = matcher.match(result1.descriptors, result2.descriptors)
            
            print(f"전체 매칭: {match_result.num_total}개")
            print(f"좋은 매칭: {match_result.num_good}개")
            print(f"인라이어 비율: {match_result.inlier_ratio:.2%}")
        else:
            print("이미지를 불러올 수 없습니다.")
    else:
        print("사용법: python feature_matching.py <image1> <image2>")
