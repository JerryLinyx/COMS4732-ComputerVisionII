import numpy as np


def normalize_points_with_K(pts_xy: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert (x, y) pixel coordinates to normalized camera coordinates via K^{-1}.

    Internal helper — expects (x, y) because K is defined in (x, y) convention.

    Args:
        pts_xy: (N, 2) pixel coordinates in (x, y) format
        K: (3, 3) camera intrinsic matrix

    Returns:
        (N, 2) normalized camera coordinates (x_norm, y_norm)
    """
    pts_homo = np.column_stack([pts_xy, np.ones(len(pts_xy))])
    return (np.linalg.inv(K) @ pts_homo.T).T[:, :2]


def decompose_E(E: np.ndarray):
    """Decompose Essential Matrix into 4 possible (R, t) solutions.

    Args:
        E: (3, 3) Essential matrix

    Returns:
        List of 4 tuples (R, t), where R is (3, 3) rotation and t is (3,) translation
    """
    U, _, VT = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(VT) < 0:
        VT = -VT
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return [
        (U @ W @ VT, U[:, 2]),
        (U @ W @ VT, -U[:, 2]),
        (U @ W.T @ VT, U[:, 2]),
        (U @ W.T @ VT, -U[:, 2]),
    ]


def triangulate_point(pt1, pt2, P1, P2):
    """Triangulate a single point from two views using DLT (Direct Linear Transform).

    Internal helper — pt1, pt2 are (x, y) to match projection matrix convention.

    Args:
        pt1: (2,) point in image 1 in (x, y) pixel coordinates
        pt2: (2,) point in image 2 in (x, y) pixel coordinates
        P1: (3, 4) projection matrix for camera 1
        P2: (3, 4) projection matrix for camera 2

    Returns:
        (4,) homogeneous 3D point [X, Y, Z, 1]
    """
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ])
    _, _, VT = np.linalg.svd(A)
    X = VT[-1]
    return X / X[3]


def check_cheirality(R, t, pts1, pts2, K, check_reprojection=False, max_reproj_error=5.0):
    """
    Check if triangulated points are in front of both cameras (positive depth).
    Optionally also filter by reprojection error.

    Args:
        R: (3, 3) rotation matrix (camera 2 relative to camera 1)
        t: (3,) translation vector
        pts1, pts2: (N, 2) corresponding points in (row, col) pixel coordinates
        K: (3, 3) camera intrinsic matrix
        check_reprojection: if True, also require reprojection error < max_reproj_error
        max_reproj_error: maximum average reprojection error (pixels) to count as valid

    Returns:
        count: number of points passing all checks
        points_3d: (N, 3) triangulated 3D points for ALL input points (not just valid ones)
    """
    # (row, col) -> (x, y) for projection matrix math
    pts1_xy = pts1[:, [1, 0]]
    pts2_xy = pts2[:, [1, 0]]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    count = 0
    points_3d = []

    for pt1, pt2 in zip(pts1_xy, pts2_xy):
        X = triangulate_point(pt1, pt2, P1, P2)
        points_3d.append(X[:3])

        depth1 = X[2]
        depth2 = (R @ X[:3] + t)[2]
        is_valid = depth1 > 0 and depth2 > 0

        if is_valid and check_reprojection:
            X_homo = np.append(X[:3], 1)
            p1_reproj = P1 @ X_homo
            p1_reproj = p1_reproj[:2] / p1_reproj[2]
            p2_reproj = P2 @ X_homo
            p2_reproj = p2_reproj[:2] / p2_reproj[2]
            avg_error = (np.linalg.norm(pt1 - p1_reproj) + np.linalg.norm(pt2 - p2_reproj)) / 2.0
            is_valid = avg_error < max_reproj_error

        if is_valid:
            count += 1

    return count, np.array(points_3d)


def triangulate_with_reprojection_filter(
    R, t, pts1, pts2, K,
    max_reprojection_error=5.0,
    min_depth=0.1,
    max_depth=1000.0,
    verbose=False,
):
    """
    Triangulate correspondences and filter by reprojection error and depth bounds.

    Args:
        R: (3, 3) rotation matrix (camera 2 relative to camera 1)
        t: (3,) translation vector
        pts1, pts2: (N, 2) corresponding points in (row, col) pixel coordinates
        K: (3, 3) camera intrinsic matrix
        max_reprojection_error: reject points with avg reprojection error >= this (pixels)
        min_depth: reject points closer than this in either camera
        max_depth: reject points farther than this in either camera
        verbose: print filtering statistics

    Returns:
        points_3d: (M, 3) valid 3D points (M <= N, only those passing all filters)
        valid_mask: (N,) boolean mask over all input points
        reprojection_errors: (N,) avg reprojection error for every input point
    """
    # (row, col) -> (x, y) for projection matrix math
    pts1_xy = pts1[:, [1, 0]]
    pts2_xy = pts2[:, [1, 0]]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    points_3d, valid_mask, reprojection_errors = [], [], []
    stats = {k: 0 for k in [
        'd1_neg', 'd2_neg', 'd1_close', 'd1_far', 'd2_close', 'd2_far', 'reproj', 'passed'
    ]}

    for pt1, pt2 in zip(pts1_xy, pts2_xy):
        X = triangulate_point(pt1, pt2, P1, P2)
        point_3d = X[:3]
        depth1 = point_3d[2]
        depth2 = (R @ point_3d + t)[2]

        X_homo = np.append(point_3d, 1)
        p1_r = P1 @ X_homo; p1_r = p1_r[:2] / p1_r[2]
        p2_r = P2 @ X_homo; p2_r = p2_r[:2] / p2_r[2]
        avg_error = (np.linalg.norm(pt1 - p1_r) + np.linalg.norm(pt2 - p2_r)) / 2.0

        is_valid = True
        if depth1 <= 0:
            stats['d1_neg'] += 1; is_valid = False
        elif depth1 < min_depth:
            stats['d1_close'] += 1; is_valid = False
        elif depth1 > max_depth:
            stats['d1_far'] += 1; is_valid = False

        if depth2 <= 0:
            stats['d2_neg'] += 1; is_valid = False
        elif depth2 < min_depth:
            stats['d2_close'] += 1; is_valid = False
        elif depth2 > max_depth:
            stats['d2_far'] += 1; is_valid = False

        if avg_error >= max_reprojection_error:
            stats['reproj'] += 1; is_valid = False
        if is_valid:
            stats['passed'] += 1

        points_3d.append(point_3d)
        valid_mask.append(is_valid)
        reprojection_errors.append(avg_error)

    points_3d = np.array(points_3d)
    valid_mask = np.array(valid_mask)
    reprojection_errors = np.array(reprojection_errors)

    if verbose:
        print(f"      === DETAILED FILTERING STATISTICS ===")
        print(f"      Total: {len(pts1)}, Passed: {stats['passed']}")
        print(f"      Failed: neg_depth1={stats['d1_neg']}, neg_depth2={stats['d2_neg']}, "
              f"close1={stats['d1_close']}, close2={stats['d2_close']}, "
              f"far1={stats['d1_far']}, far2={stats['d2_far']}, "
              f"reproj>{max_reprojection_error}px={stats['reproj']}")

    return points_3d[valid_mask], valid_mask, reprojection_errors


def recover_pose(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray):
    """
    Decompose E into 4 candidate (R, t) solutions and select the one where the
    most triangulated points have positive depth in both cameras (cheirality check).

    Args:
        E: (3, 3) Essential matrix
        pts1: (N, 2) points from image 1 in (row, col) pixel coordinates
        pts2: (N, 2) points from image 2 in (row, col) pixel coordinates
        K: (3, 3) camera intrinsic matrix

    Returns:
        R: (3, 3) rotation matrix, or None on failure
        t: (3,) translation vector, or None on failure
        points_3d: (N, 3) triangulated 3D points for ALL input points, or None on failure
    """
    solutions = decompose_E(E)

    best_solution = None
    best_depth_count = 0
    best_reproj_count = float('inf')
    best_3d = None

    for R_cand, t_cand in solutions:
        depth_count, pts3d = check_cheirality(
            R_cand, t_cand, pts1, pts2, K,
            check_reprojection=False, max_reproj_error=10.0,
        )
        reproj_count, _ = check_cheirality(
            R_cand, t_cand, pts1, pts2, K,
            check_reprojection=True, max_reproj_error=30.0,
        )

        is_better = (
            best_solution is None
            or depth_count > best_depth_count
            or (depth_count == best_depth_count and reproj_count > best_reproj_count)
        )
        if is_better:
            best_depth_count = depth_count
            best_reproj_count = reproj_count
            best_solution = (R_cand, t_cand)
            best_3d = pts3d

    if best_solution is None:
        return None, None, None
    return best_solution[0], best_solution[1], best_3d
