"""
Two-View Stereo Reconstruction Pipeline
========================================
Orchestrates camera intrinsics, feature detection/matching,
RANSAC pose estimation, and sparse triangulation.
"""

import os
import random
import datetime
from dataclasses import dataclass, asdict

import numpy as np
import yaml
import skimage.io as skio
from skimage.color import rgb2gray

from intrinsics import compute_K
from features import get_sift_features, match_features
from triangulation import triangulate_with_reprojection_filter
from ransac import RANSAC

from utils import (
    rotation_matrix_to_euler_angles,
    compute_baseline,
    get_rgb_patches_for_sift,
    setup_camera_coordinate_system,
)
from utils_visualizations import (
    generate_correspondence_colors,
    create_side_by_side_original,
    create_side_by_side_corners,
    create_side_by_side_correspondences,
    plot_3d_points,
    plot_camera_poses,
    plot_epipolar_lines,
    create_pose_summary_plot,
    create_comprehensive_pipeline_grid,
    create_feature_matching_visualization,
    plot_nndr_histogram,
    plot_top_matches,
)
try:
    from visualize_viser import visualize_scene
except ImportError:
    visualize_scene = None


def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class Config:
    """Configuration for the two-view stereo reconstruction pipeline."""
    # Image paths
    img1_path: str = None # TODO: provide
    img2_path: str = None # TODO: provide
    output_dir_path: str = "outputs"

    random_seed: int = None # TODO: provide

    # Step 1 camera params
    # (iPhone 15 Pro 1x, 35mm-derived sensor dims for square pixels)
    optical_focal_length_mm: float = 6.765
    sensor_width_mm: float = 9.757
    sensor_height_mm: float = 7.318

    # Step 2 SIFT parameters
    sift_edge_discard: int = 20
    sift_max_features: int = None # TODO: provide num SIFT features used
    # Note: you shouldn't need to touch these sift params below. If you do, report what you changed.
    sift_contrast_threshold: float = 0.04  # Contrast threshold (lower = more features, but less stable)
    sift_edge_threshold: float = 10  # Edge threshold (higher = more features, including edges)
    sift_n_octave_layers: int = 3  # Number of layers per octave (more = finer scale sampling)
    sift_sigma: float = 1.6  # Initial Gaussian sigma for image smoothing

    # Step 3 Feature matching
    feature_matching_ratio_threshold: float = None # TODO: provide NNDR ratio

    # Step 4 RANSAC
    ransac_s: int = 8  # Minimum 8 points for essential matrix
    ransac_epsilon = None # TODO: provide distance threshold
    ransac_num_iters: int = None # TODO: provide number of ransac iterations

    # Step 5 Triangulation filtering
    # Note: you shouldn't need to touch these triangulation params below. If you do, report what you changed.
    triang_max_reprojection_error: float = 50.0  # Reject points with avg reproj error >= this (pixels)
    triang_min_depth: float = 0.01  # Reject points closer than this in either camera
    triang_max_depth: float = 10000.0  # Reject points farther than this in either camera

    # launch viser server automatically at the end of execution.
    # If false, you can run the CLI script printed at the end.
    launch_viser: bool = False


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main(cfg: Config):
    """Main two-view stereo reconstruction pipeline."""
    seed_everything(cfg.random_seed)

    img1_name = cfg.img1_path.split("/")[-1].split(".")[0]
    img2_name = cfg.img2_path.split("/")[-1].split(".")[0]

    # Create output directory
    run_str = (f"{img1_name}_{img2_name}_pose_estimation_SIFT"
               f"_maxfeat={cfg.sift_max_features}"
               f"_{cfg.feature_matching_ratio_threshold}_NNDR_thresh")

    base_output_dir = cfg.output_dir_path
    os.makedirs(base_output_dir, exist_ok=True)
    run_attempt_num = len([
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith(run_str)
    ])
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.output_dir_path = f"{base_output_dir}/{run_str}_{run_attempt_num}_{timestamp}"
    os.makedirs(cfg.output_dir_path, exist_ok=True)

    # Save configuration
    print(f"\nSaving configuration to YAML...")
    config_yaml_path = f"{cfg.output_dir_path}/config.yml"
    with open(config_yaml_path, 'w') as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
    print(f"      Configuration saved to: {config_yaml_path}")

    # ========================================
    # STEP 0: Load Images
    # ========================================
    print("=" * 60)
    print("POSE ESTIMATION PIPELINE")
    print("=" * 60)
    print(f"\n[0/5] Loading images from {cfg.img1_path} and {cfg.img2_path}")
    img1 = skio.imread(cfg.img1_path, as_gray=False)
    img1_gray = rgb2gray(img1)
    img2 = skio.imread(cfg.img2_path, as_gray=False)
    img2_gray = rgb2gray(img2)
    print(f"      Image 1 shape: {img1.shape}")
    print(f"      Image 2 shape: {img2.shape}")

    print(f"      - Saving original images visualization...")
    create_side_by_side_original(img1, img2, f"{cfg.output_dir_path}/step0_original_images.png")

    # ========================================
    # STEP 1: Camera Intrinsics
    # ========================================
    print(f"\n[1/5] Loading camera intrinsics")
    img_height_px, img_width_px = img1.shape[:2]
    K = compute_K(
        img_width_px=img_width_px,
        img_height_px=img_height_px,
        optical_focal_length_mm=cfg.optical_focal_length_mm,
        sensor_width_mm=cfg.sensor_width_mm,
        sensor_height_mm=cfg.sensor_height_mm,
    )
    print(f"      Camera intrinsic matrix K:")
    for row in K:
        print(f"      [{row[0]:8.2f}  {row[1]:8.2f}  {row[2]:8.2f}]")

    # ========================================
    # STEP 2: SIFT Feature Detection
    # ========================================
    print(f"\n[2/5] Detecting SIFT features (rotation + scale invariant)")

    corners1, img1_features, responses1 = get_sift_features(
        img1_gray,
        edge_discard=cfg.sift_edge_discard,
        max_features=cfg.sift_max_features,
        contrast_threshold=cfg.sift_contrast_threshold,
        edge_threshold=cfg.sift_edge_threshold,
        n_octave_layers=cfg.sift_n_octave_layers,
        sigma=cfg.sift_sigma,
    )
    corners2, img2_features, responses2 = get_sift_features(
        img2_gray,
        edge_discard=cfg.sift_edge_discard,
        max_features=cfg.sift_max_features,
        contrast_threshold=cfg.sift_contrast_threshold,
        edge_threshold=cfg.sift_edge_threshold,
        n_octave_layers=cfg.sift_n_octave_layers,
        sigma=cfg.sift_sigma,
    )
    print(f"      Found {corners1.shape[1]} SIFT features in image 1")
    print(f"      Found {corners2.shape[1]} SIFT features in image 2")

    print(f"      - Saving SIFT keypoints visualization...")
    create_side_by_side_corners(
        img1, img2, corners1.T, corners2.T,
        f"{cfg.output_dir_path}/step2_sift_keypoints.png",
        title_suffix=f"SIFT Keypoint Detection (Top {cfg.sift_max_features} by Response Strength)",
    )

    keypoints1 = corners1.T  # (N, 2) in (row, col)
    keypoints2 = corners2.T  # (N, 2) in (row, col)

    print(f"      Extracting RGB patches for visualization")
    img1_rgb_patches = get_rgb_patches_for_sift(corners1, img1)
    img2_rgb_patches = get_rgb_patches_for_sift(corners2, img2)
    print(f"      Extracted {len(img1_rgb_patches)} / {len(img2_rgb_patches)} RGB patches")

    # ========================================
    # STEP 3: Feature Matching
    # ========================================
    print(f"\n[3/5] Matching features between images (L2 + NNDR)")
    img1_img2_correspondence_pairs, knn_matches, nndr_proportions = match_features(
        img1_features, img2_features, keypoints1, keypoints2,
        ratio_threshold=cfg.feature_matching_ratio_threshold,
    )

    num_correspondences = len(img1_img2_correspondence_pairs)
    print(f"      Found {num_correspondences} correspondence pairs")
    colors = generate_correspondence_colors(num_correspondences)

    # Pre-RANSAC visualizations
    create_side_by_side_correspondences(
        img1, img2, img1_img2_correspondence_pairs, colors,
        f"{cfg.output_dir_path}/step3_correspondences_before_ransac.png",
        title_suffix=f"Before RANSAC (n={num_correspondences})",
    )
    create_feature_matching_visualization(
        img1, img2, keypoints1, keypoints2,
        img1_img2_correspondence_pairs, colors,
        f"{cfg.output_dir_path}/step3_feature_matching.png",
        title_suffix="Before RANSAC",
    )

    if len(nndr_proportions) > 0:
        print(f"      - Saving NNDR histogram...")
        plot_nndr_histogram(
            nndr_proportions, ratio_threshold=cfg.feature_matching_ratio_threshold,
            output_dir=cfg.output_dir_path,
        )

    print(f"      - Saving top 5 NNDR matches...")
    plot_top_matches(
        knn_matches, keypoints1, keypoints2, img1, img2,
        output_dir=cfg.output_dir_path, num_plot=5,
    )

    # ========================================
    # STEP 4: RANSAC Pose Estimation
    # ========================================
    print(f"\n[4/5] Estimating camera pose using RANSAC (Essential Matrix)")
    if num_correspondences < cfg.ransac_s:
        print(f"      Not enough correspondences ({num_correspondences}) for RANSAC")
        result = (None, None, None, None, None)
    else:
        result = RANSAC(
            correspondence_pairs=img1_img2_correspondence_pairs,
            K=K, s=cfg.ransac_s, epsilon=cfg.ransac_epsilon,
            num_iters=cfg.ransac_num_iters,
            output_dir=cfg.output_dir_path,
        )

    if result[0] is None:
        print("\n" + "=" * 60)
        print("ERROR: Pose estimation failed!")
        print("=" * 60)
        create_comprehensive_pipeline_grid(
            img1, img2, corners1.T, corners2.T, keypoints1, keypoints2,
            img1_img2_correspondence_pairs, np.zeros((0, 2, 2)),
            colors, [], np.eye(3), np.zeros(3), np.zeros((0, 3)),
            f"{cfg.output_dir_path}/step4_pipeline_grid_FAILED.png",
            feature_type="sift", sift_use_rootsift=False,
            config=cfg, ransac_failed=True,
        )
        print(f"\nPartial results saved to: {cfg.output_dir_path}")
        return

    R, t, inliers_mask, E, points_3d_ransac = result
    inlier_correspondences = img1_img2_correspondence_pairs[inliers_mask]
    num_inliers = np.sum(inliers_mask)
    print(f"      Inliers: {num_inliers} / {num_correspondences} "
          f"({100 * num_inliers / num_correspondences:.1f}%)")

    # ========================================
    # STEP 5: Triangulation + Visualization
    # ========================================
    print(f"\n[5/5] Triangulating sparse point cloud...")
    inlier_pts1 = inlier_correspondences[:, 0]  # (N, 2) in (row, col)
    inlier_pts2 = inlier_correspondences[:, 1]  # (N, 2) in (row, col)

    points_3d, valid_3d_mask_subset, reprojection_errors = (
        triangulate_with_reprojection_filter(
            R, t, inlier_pts1, inlier_pts2, K,
            max_reprojection_error=cfg.triang_max_reprojection_error,
            min_depth=cfg.triang_min_depth,
            max_depth=cfg.triang_max_depth,
            verbose=True,
        )
    )

    valid_3d_mask = np.zeros(num_correspondences, dtype=bool)
    inlier_indices = np.where(inliers_mask)[0]
    valid_3d_mask[inlier_indices[valid_3d_mask_subset]] = True

    print(f"\n      Final sparse point cloud: {len(points_3d)} valid 3D points")

    # Reprojection error statistics
    print(f"\n      === REPROJECTION ERROR ANALYSIS ===")
    print(f"      All {len(reprojection_errors)} triangulated points:")
    for label, fn in [("Min", np.min), ("Mean", np.mean), ("Median", np.median),
                      ("75th", lambda x: np.percentile(x, 75)),
                      ("90th", lambda x: np.percentile(x, 90)),
                      ("95th", lambda x: np.percentile(x, 95)),
                      ("Max", np.max)]:
        print(f"        {label}: {fn(reprojection_errors):.2f} px")

    if len(points_3d) > 0:
        valid_errors = reprojection_errors[valid_3d_mask_subset]
        print(f"\n      Valid points ({len(valid_errors)}):")
        print(f"        Mean: {np.mean(valid_errors):.2f}, "
              f"Median: {np.median(valid_errors):.2f}, "
              f"Max: {np.max(valid_errors):.2f} px")

    inlier_colors = generate_correspondence_colors(num_inliers)

    # Analyze and Visualize Results
    print(f"\n      Analyzing and visualizing pose estimation results")

    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
    baseline = compute_baseline(t)

    print(f"\n      === POSE ESTIMATION RESULTS ===")
    print(f"      Rotation: Roll={roll:.2f} Pitch={pitch:.2f} Yaw={yaw:.2f}")
    print(f"      Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"      Baseline: {baseline:.4f} units")
    print(f"      3D points: {len(points_3d)}")

    # Post-RANSAC visualizations
    print(f"\n      Creating post-RANSAC visualizations...")
    create_side_by_side_correspondences(
        img1, img2, inlier_correspondences, inlier_colors,
        f"{cfg.output_dir_path}/step4_correspondences_after_ransac.png",
        title_suffix=f"After RANSAC (n={num_inliers} inliers)",
    )
    create_feature_matching_visualization(
        img1, img2, keypoints1, keypoints2,
        inlier_correspondences, inlier_colors,
        f"{cfg.output_dir_path}/step4_feature_matching_after_ransac.png",
        title_suffix="After RANSAC (Inliers Only)",
    )

    if len(points_3d) > 0:
        print(f"      - Saving 3D reconstruction visualizations...")
        plot_3d_points(points_3d, f"{cfg.output_dir_path}/step5_3d_reconstruction.png",
                       title="Triangulated 3D Points")
        plot_camera_poses(R, t, points_3d, f"{cfg.output_dir_path}/step5_camera_poses.png",
                          title="Camera Configuration and 3D Reconstruction")

    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    F = F / F[2, 2]

    print(f"      - Saving epipolar geometry visualizations...")
    plot_epipolar_lines(
        img1, img2,
        inlier_correspondences[:, 0, :], inlier_correspondences[:, 1, :],
        F, f"{cfg.output_dir_path}/step4_epipolar_lines.png",
        num_lines=min(15, num_inliers),
    )

    create_pose_summary_plot(
        R, t, points_3d, roll, pitch, yaw, baseline,
        num_inliers, num_correspondences,
        f"{cfg.output_dir_path}/step5_pose_summary.png",
    )

    print(f"      - Saving comprehensive pipeline grid...")
    create_comprehensive_pipeline_grid(
        img1, img2, corners1.T, corners2.T, keypoints1, keypoints2,
        img1_img2_correspondence_pairs, inlier_correspondences,
        colors, inlier_colors, R, t, points_3d,
        f"{cfg.output_dir_path}/step5_pipeline_grid.png",
        feature_type="sift", sift_use_rootsift=False,
        config=cfg, F=F,
    )

    # Save numerical results
    print(f"\n      Saving numerical results...")
    with open(f"{cfg.output_dir_path}/step4_pose_results.txt", 'w') as f:
        f.write("POSE ESTIMATION RESULTS\n" + "=" * 60 + "\n\n")
        f.write("ESSENTIAL MATRIX:\n")
        for i in range(3):
            f.write(f"  [{E[i, 0]:10.7f}  {E[i, 1]:10.7f}  {E[i, 2]:10.7f}]\n")
        f.write("\nROTATION MATRIX:\n")
        for i in range(3):
            f.write(f"  [{R[i, 0]:8.5f}  {R[i, 1]:8.5f}  {R[i, 2]:8.5f}]\n")
        f.write(f"\nEULER ANGLES:\n  Roll: {roll:.2f}  Pitch: {pitch:.2f}  Yaw: {yaw:.2f}\n")
        f.write(f"\nTRANSLATION: [{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}]\n")
        f.write(f"BASELINE: {baseline:.5f}\n")
        f.write("\nCAMERA INTRINSICS:\n")
        for i in range(3):
            f.write(f"  [{K[i, 0]:8.2f}  {K[i, 1]:8.2f}  {K[i, 2]:8.2f}]\n")
        f.write("\nFUNDAMENTAL MATRIX (F = K^(-T) E K^(-1)):\n")
        for i in range(3):
            f.write(f"  [{F[i, 0]:10.7f}  {F[i, 1]:10.7f}  {F[i, 2]:10.7f}]\n")
        f.write(f"\nRANSAC: {num_inliers}/{num_correspondences} inliers "
                f"({100 * num_inliers / num_correspondences:.2f}%)\n")
        f.write(f"\n3D RECONSTRUCTION (Sparse Triangulation): {len(points_3d)} points\n")
        if len(points_3d) > 0:
            f.write(f"  Mean depth: {np.mean(points_3d[:, 2]):.4f}\n")
            f.write(f"  Depth range: [{np.min(points_3d[:, 2]):.4f}, {np.max(points_3d[:, 2]):.4f}]\n")

    # Export scene data for 3D viewer
    print(f"\n      Exporting scene data for 3D viewer...")
    valid_correspondences = img1_img2_correspondence_pairs[valid_3d_mask]
    point_colors = []
    for pt in valid_correspondences[:, 0, :]:
        y, x = int(pt[0]), int(pt[1])
        if img1.ndim == 3:
            point_colors.append(img1[y, x, :3])
        else:
            g = img1[y, x] * 255 if img1.max() <= 1.0 else img1[y, x]
            point_colors.append(np.array([g, g, g]))
    point_colors = np.array(point_colors)

    T0, T1, _, camera_poses = setup_camera_coordinate_system(R, t, points_3d)

    img1_export = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1
    img2_export = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2

    npz_path = f"{cfg.output_dir_path}/step5_scene_data.npz"

    corr_img1 = valid_correspondences[:, 0, :]
    corr_img2 = valid_correspondences[:, 1, :]

    np.savez(
        npz_path,
        points_3d=points_3d, point_colors=point_colors,
        camera_poses=camera_poses, K=K, R=R, t=t,
        correspondences_img1=corr_img1, correspondences_img2=corr_img2,
        E=E, F=F,
        img1=img1_export, img2=img2_export,
        img1_path=cfg.img1_path, img2_path=cfg.img2_path,
        img1_shape=img1.shape, img2_shape=img2.shape,
        num_inliers=num_inliers, num_valid_3d_points=len(points_3d),
        baseline=baseline,
        reconstruction_method="sparse_triangulation",
    )
    print(f"      Saved scene data to: {npz_path}")

    # Save viser launch command
    with open(f"{cfg.output_dir_path}/launch_viser.txt", 'w') as f:
        f.write(f"# Launch 3D visualization:\npython visualize_viser.py {npz_path}\n")

    print(f"\n" + "=" * 60)
    print(f"POSE ESTIMATION COMPLETE!")
    print(f"Results saved to: {cfg.output_dir_path}")
    print("=" * 60 + "\n")

    if cfg.launch_viser and len(points_3d) > 0:
        print(f"Launching 3D visualization in viser...")
        print(f"\nTo visualize later: python visualize_viser.py {npz_path}\n")
        visualize_scene(npz_path, port=8081)
    else:
        print(f"\nTo visualize in 3D: python visualize_viser.py {npz_path}\n")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
