"""
Point Cloud Visualization Module

3D 포인트 클라우드를 시각화하는 도구입니다.
Open3D와 Matplotlib를 사용하여 인터랙티브한 시각화를 제공합니다.
"""

import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


class PointCloudVisualizer:
    """
    포인트 클라우드 시각화 클래스
    
    Open3D 또는 Matplotlib를 사용하여 3D 포인트 클라우드를 시각화합니다.
    """
    
    def __init__(self, backend: str = "auto"):
        """
        시각화 초기화
        
        Args:
            backend: "open3d", "matplotlib", 또는 "auto"
        """
        self.backend = self._detect_backend(backend)
        print(f"시각화 백엔드: {self.backend}")
    
    def _detect_backend(self, backend: str) -> str:
        """사용 가능한 백엔드를 감지"""
        if backend == "auto":
            try:
                import open3d
                return "open3d"
            except ImportError:
                try:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    return "matplotlib"
                except ImportError:
                    raise ImportError("Open3D 또는 Matplotlib가 필요합니다.")
        return backend
    
    def visualize(self, points: np.ndarray, 
                  colors: Optional[np.ndarray] = None,
                  point_size: float = 2.0,
                  title: str = "3D Point Cloud") -> None:
        """
        포인트 클라우드를 시각화합니다.
        
        Args:
            points: Nx3 3D 점
            colors: Nx3 RGB 컬러 (0-255)
            point_size: 포인트 크기
            title: 윈도우 제목
        """
        if self.backend == "open3d":
            self._visualize_open3d(points, colors, point_size, title)
        else:
            self._visualize_matplotlib(points, colors, point_size, title)
    
    def _visualize_open3d(self, points: np.ndarray,
                          colors: Optional[np.ndarray],
                          point_size: float,
                          title: str) -> None:
        """Open3D를 사용한 시각화"""
        import open3d as o3d
        
        # 포인트 클라우드 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            # 0-255 -> 0-1 변환
            if colors.max() > 1:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 시각화 설정
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd)
        
        # 렌더링 옵션
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.1, 0.1, 0.15])
        
        # 좌표축 표시
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(axis)
        
        vis.run()
        vis.destroy_window()
    
    def _visualize_matplotlib(self, points: np.ndarray,
                              colors: Optional[np.ndarray],
                              point_size: float,
                              title: str) -> None:
        """Matplotlib를 사용한 시각화"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 컬러 처리
        if colors is not None:
            if colors.max() > 1:
                colors = colors / 255.0
            c = colors
        else:
            c = points[:, 2]  # Z값으로 색상 지정
        
        # 산점도
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=c, s=point_size, alpha=0.8
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 축 비율 동일하게
        max_range = np.max([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]) / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
        mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
        mid_z = (points[:, 2].max() + points[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_cameras(self, camera_poses: List, 
                          K: np.ndarray,
                          scale: float = 0.5) -> None:
        """
        카메라 위치와 방향을 시각화합니다.
        
        Args:
            camera_poses: CameraPose 리스트
            K: 카메라 내부 파라미터
            scale: 카메라 프러스텀 크기
        """
        import open3d as o3d
        
        geometries = []
        
        for i, pose in enumerate(camera_poses):
            # 카메라 중심 계산
            R = pose.R
            t = pose.t.flatten()
            center = -R.T @ t
            
            # 카메라 방향 (Z축)
            direction = R.T @ np.array([0, 0, 1]) * scale
            
            # 라인으로 표시
            points = [center, center + direction]
            lines = [[0, 1]]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 빨간색
            
            geometries.append(line_set)
            
            # 카메라 중심에 구 표시
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.1)
            sphere.translate(center)
            sphere.paint_uniform_color([0, 0.5, 1])  # 파란색
            geometries.append(sphere)
        
        o3d.visualization.draw_geometries(geometries)
    
    def save_screenshot(self, points: np.ndarray,
                        colors: Optional[np.ndarray],
                        output_path: str,
                        elevation: float = 30,
                        azimuth: float = 45) -> None:
        """
        포인트 클라우드 스크린샷을 저장합니다.
        
        Args:
            points, colors: 포인트 클라우드 데이터
            output_path: 저장 경로
            elevation, azimuth: 시점 각도
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 12), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is not None:
            if colors.max() > 1:
                colors = colors / 255.0
            c = colors
        else:
            c = 'steelblue'
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=c, s=1, alpha=0.8)
        
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Cloud')
        
        # 배경색
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        
        plt.savefig(output_path, facecolor='#1a1a2e', edgecolor='none', 
                    bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        print(f"스크린샷 저장됨: {output_path}")


def load_ply(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    PLY 파일을 로드합니다.
    
    Args:
        filepath: PLY 파일 경로
    
    Returns:
        Tuple[points, colors]: 3D 점과 컬러
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255 if pcd.has_colors() else None
        return points, colors
    except ImportError:
        # 수동 파싱
        points = []
        colors = []
        
        with open(filepath, 'r') as f:
            # 헤더 건너뛰기
            while True:
                line = f.readline().strip()
                if line == "end_header":
                    break
            
            # 데이터 읽기
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if len(parts) >= 6:
                        colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
        
        points = np.array(points)
        colors = np.array(colors) if colors else None
        
        return points, colors


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
        
        # PLY 로드
        points, colors = load_ply(ply_path)
        print(f"로드된 점: {len(points)}개")
        
        # 시각화
        viz = PointCloudVisualizer()
        viz.visualize(points, colors)
    else:
        # 테스트용 랜덤 데이터
        print("테스트 데이터로 시각화 실행...")
        
        np.random.seed(42)
        n_points = 1000
        
        # 구 형태의 포인트 클라우드
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = 5 + np.random.randn(n_points) * 0.3
        
        points = np.zeros((n_points, 3))
        points[:, 0] = r * np.sin(phi) * np.cos(theta)
        points[:, 1] = r * np.sin(phi) * np.sin(theta)
        points[:, 2] = r * np.cos(phi)
        
        # 높이에 따른 색상
        colors = np.zeros((n_points, 3))
        normalized_z = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
        colors[:, 0] = (normalized_z * 255).astype(np.uint8)  # R
        colors[:, 2] = ((1 - normalized_z) * 255).astype(np.uint8)  # B
        
        viz = PointCloudVisualizer()
        viz.visualize(points, colors, title="Test Point Cloud")
