import numpy as np
from tqdm import tqdm

from triangulation import (
    normalize_points_with_K,
    decompose_E,
    check_cheirality,
    recover_pose,
)
from utils_visualizations import plot_ransac_convergence


def compute_E(
    img1_pts: np.ndarray,
    img2_pts: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute Essential Matrix from calibrated points using the 8-point algorithm.

    Args:
        img1_pts: (N, 2) points from image 1 in (row, col) pixel coordinates, N >= 8
        img2_pts: (N, 2) points from image 2 in (row, col) pixel coordinates, N >= 8
        K: (3, 3) camera intrinsic matrix

    Returns:
        E: (3, 3) Essential matrix with enforced rank-2 constraint (singular values [1, 1, 0])
    """
    assert len(img1_pts) >= 8 and len(img1_pts) == len(img2_pts)

    # (row, col) -> (x, y) for K-normalization
    pts1 = normalize_points_with_K(img1_pts[:, [1, 0]], K)
    pts2 = normalize_points_with_K(img2_pts[:, [1, 0]], K)

    # Build constraint matrix: p2^T E p1 = 0
    A = np.array([
        [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]
        for (x1, y1), (x2, y2) in zip(pts1, pts2)
    ])

    _, _, VT = np.linalg.svd(A, full_matrices=True)
    E = VT[-1].reshape(3, 3)

    # Procrustean projection: enforce rank-2 with equal singular values
    U, S, VT = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ VT
    return E


def sampson_distance(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """
    Compute squared Sampson distance for each correspondence.

    K-normalizes pixel coordinates internally, then evaluates:
    d^2 = (p2^T E p1)^2 / ((Ep1)_1^2 + (Ep1)_2^2 + (E^Tp2)_1^2 + (E^Tp2)_2^2)

    Args:
        E: (3, 3) Essential matrix
        pts1: (N, 2) points from image 1 in (row, col) pixel coordinates
        pts2: (N, 2) points from image 2 in (row, col) pixel coordinates
        K_inv: (3, 3) precomputed inverse of camera intrinsic matrix

    Returns:
        (N,) squared Sampson distances
    """
    # (row, col) -> (x, y), then K-normalize to camera coords (homogeneous)
    pts1_xy = pts1[:, [1, 0]]
    pts2_xy = pts2[:, [1, 0]]
    pts1_h = (K_inv @ np.column_stack([pts1_xy, np.ones(len(pts1_xy))]).T).T  # (N, 3)
    pts2_h = (K_inv @ np.column_stack([pts2_xy, np.ones(len(pts2_xy))]).T).T  # (N, 3)

    Ep1 = pts1_h @ E.T          # (N, 3)
    ETp2 = pts2_h @ E           # (N, 3)
    numerator = np.sum(pts2_h * Ep1, axis=1)  # p2^T E p1
    denominator = Ep1[:, 0]**2 + Ep1[:, 1]**2 + ETp2[:, 0]**2 + ETp2[:, 1]**2
    return numerator**2 / (denominator + 1e-8)


def RANSAC(
    correspondence_pairs: np.ndarray,
    K: np.ndarray,
    s: int,
    epsilon: float,
    num_iters: int,
    output_dir: str,
):
    """
    Standard RANSAC for pose estimation via Essential matrix with Sampson distance.

    Loop: sample 8 points, estimate E, decompose + cheirality, count inliers, keep best.
    Post-loop: recompute inliers from best E, then re-estimate E/R/t from those inliers.

    Args:
        correspondence_pairs: (N, 2, 2) with points in (row, col) format
        K: (3, 3) camera intrinsic matrix
        s: points per sample (>= 8)
        epsilon: squared Sampson distance threshold in normalized coordinates
        num_iters: RANSAC iterations
        output_dir: directory for convergence plot (None to skip)

    Returns:
        (R, t, inliers_mask, E) or all-None tuple on failure
    """
    print(f"\nPerforming RANSAC for Pose Estimation (Essential Matrix)")
    print(f"Using {s} point correspondences per iteration")

    s = max(8, min(len(correspondence_pairs), s))
    img1_pts = correspondence_pairs[:, 0]  # (N, 2) in (row, col)
    img2_pts = correspondence_pairs[:, 1]  # (N, 2) in (row, col)

    # to be used later
    K_inv = np.linalg.inv(K)

    best_num_inliers = 0
    best_E = None
    best_inliers_mask = None

    # Convergence tracking
    inlier_counts, best_so_far = [], []
    improvement_iters, improvement_vals = [], []

    pbar = tqdm(range(num_iters))
    for it in pbar:
        idx = np.random.choice(len(img1_pts), s, replace=False)

        try:
            # ============================================================
            # YOUR CODE HERE:
            # 1. Estimate E from the sampled points using compute_E()
            # 2. Recover pose (R, t) using recover_pose(). If R is None
            #    (cheirality failure), record 0 inliers and continue.
            # 3. Compute inliers over ALL correspondences using
            #    sampson_distance() < epsilon  (hint: pass K_inv)
            # 4. If more inliers than current best, update
            #    best_num_inliers, best_E, best_inliers_mask
            #    and append to improvement_iters / improvement_vals
            #
            # Don't forget to append to inlier_counts and best_so_far
            # each iteration (needed for the convergence plot).
            # ============================================================
            return # fix me plz
            # ============================================================

        except (np.linalg.LinAlgError, AssertionError):
            inlier_counts.append(0)
            best_so_far.append(best_num_inliers)
            continue

    if best_E is None:
        print("WARNING: RANSAC failed to find a valid pose!")
        return None, None, None, None

    # Post-loop refinement: re-estimate E from all inliers (not just
    # the 8-point sample), then recover the final (R, t).
    pre_count = best_num_inliers

    if best_num_inliers >= 8:
        # ============================================================
        # YOUR CODE HERE: Re-estimate E from the inlier set,
        # then recover the final (R, t) via recover_pose()
        # ============================================================
        return # fix me plz
        # ============================================================
    else:
        print("WARNING: Not enough inliers for post-loop refinement!")
        return None, None, None, None

    if best_R is None:
        print("WARNING: Post-loop pose recovery failed!")
        return None, None, None, None

    print(f"Found {best_num_inliers} inliers / {len(img1_pts)} correspondences "
          f"(pre-refinement: {pre_count})")

    # Debug: show all 4 decomposition solutions
    print(f"\n      === E-MATRIX DECOMPOSITION ANALYSIS ===")
    for sol_idx, (R_cand, t_cand) in enumerate(decompose_E(best_E)):
        dc, _ = check_cheirality(
            R_cand, t_cand, img1_pts[best_inliers_mask],
            img2_pts[best_inliers_mask], K, check_reprojection=False,
        )
        rc, _ = check_cheirality(
            R_cand, t_cand, img1_pts[best_inliers_mask],
            img2_pts[best_inliers_mask], K, check_reprojection=True, max_reproj_error=30.0,
        )
        t_n = t_cand / np.linalg.norm(t_cand) if np.linalg.norm(t_cand) > 0 else t_cand
        marker = " <- SELECTED" if np.allclose(t_cand, best_t) and np.allclose(R_cand, best_R) else ""
        print(f"      Solution {sol_idx}: depth={dc}/{best_num_inliers}, reproj={rc}/{best_num_inliers}")
        print(f"        t = [{t_n[0]:7.3f}, {t_n[1]:7.3f}, {t_n[2]:7.3f}]{marker}")
    print(f"      ======================================\n")

    # Plot convergence
    if output_dir:
        plot_ransac_convergence(num_iters, inlier_counts, best_so_far,
                                improvement_iters, improvement_vals,
                                best_num_inliers, epsilon, output_dir)

    return best_R, best_t, best_inliers_mask, best_E