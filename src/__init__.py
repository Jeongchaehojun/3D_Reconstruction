"""
3D Reconstruction - Structure from Motion Library

This package provides tools for 3D reconstruction from multiple 2D images
using Structure from Motion (SfM) techniques.
"""

from .feature_detection import FeatureDetector
from .feature_matching import FeatureMatcher
from .fundamental_matrix import FundamentalMatrixEstimator
from .camera_pose import CameraPoseEstimator
from .triangulation import Triangulator
from .sfm_pipeline import SfMPipeline

__version__ = "0.1.0"
__all__ = [
    "FeatureDetector",
    "FeatureMatcher", 
    "FundamentalMatrixEstimator",
    "CameraPoseEstimator",
    "Triangulator",
    "SfMPipeline",
]
