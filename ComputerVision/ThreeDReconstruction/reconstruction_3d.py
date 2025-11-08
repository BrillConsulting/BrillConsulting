"""
3D Reconstruction from Multiple Views
Structure from Motion (SfM) and Multi-View Stereo (MVS)
"""

import cv2
import numpy as np
from typing import List, Tuple


class ThreeDReconstructor:
    """
    3D Reconstruction from images
    - Feature matching (SIFT, ORB, SuperPoint)
    - Camera pose estimation
    - Triangulation
    - Dense reconstruction
    """

    def __init__(self, method: str = "sift"):
        self.method = method
        self.detector = self._init_detector()
        print(f"✓ 3D Reconstructor initialized with {method}")

    def _init_detector(self):
        """Initialize feature detector"""
        if self.method == "sift":
            return cv2.SIFT_create()
        elif self.method == "orb":
            return cv2.ORB_create(nfeatures=5000)
        else:
            return cv2.SIFT_create()

    def reconstruct(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct 3D points from multiple views

        Args:
            images: List of images (minimum 2)

        Returns:
            points_3d: 3D points (Nx3)
            colors: RGB colors (Nx3)
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images")

        # 1. Extract features
        keypoints_list, descriptors_list = [], []
        for img in images:
            kp, desc = self.detector.detectAndCompute(img, None)
            keypoints_list.append(kp)
            descriptors_list.append(desc)

        # 2. Match features
        matches = self._match_features(descriptors_list[0], descriptors_list[1])

        # 3. Estimate fundamental matrix
        pts1 = np.float32([keypoints_list[0][m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints_list[1][m.trainIdx].pt for m in matches])

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # 4. Compute essential matrix (assuming calibrated cameras)
        h, w = images[0].shape[:2]
        K = self._get_intrinsics(w, h)  # Camera intrinsics

        E = K.T @ F @ K

        # 5. Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # 6. Triangulate points
        proj1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        proj2 = K @ np.hstack([R, t])

        points_4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T

        # 7. Get colors
        colors = self._get_point_colors(images[0], pts1)

        print(f"✓ Reconstructed {len(points_3d)} 3D points")
        return points_3d, colors

    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features between images"""
        if self.method == "orb":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)[:500]

        return matches

    def _get_intrinsics(self, width: int, height: int) -> np.ndarray:
        """Estimate camera intrinsics (simplified)"""
        focal_length = width  # Approximation
        cx, cy = width / 2, height / 2

        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        return K

    def _get_point_colors(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Get RGB colors for 3D points"""
        colors = []
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                color = image[y, x]
                colors.append(color[::-1])  # BGR to RGB
            else:
                colors.append([128, 128, 128])

        return np.array(colors, dtype=np.uint8)

    def save_ply(self, points_3d: np.ndarray, colors: np.ndarray, filename: str):
        """Save point cloud as PLY file"""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for pt, color in zip(points_3d, colors):
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")

        print(f"✓ Point cloud saved: {filename}")


# Demo
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs='+', required=True, help="Input images")
    parser.add_argument("--output", default="reconstruction.ply")
    parser.add_argument("--method", default="sift", choices=["sift", "orb"])
    args = parser.parse_args()

    reconstructor = ThreeDReconstructor(args.method)

    images = [cv2.imread(img) for img in args.images]
    points_3d, colors = reconstructor.reconstruct(images)
    reconstructor.save_ply(points_3d, colors, args.output)
