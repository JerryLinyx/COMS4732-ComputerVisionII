import numpy as np
import cv2


def get_sift_features(
    img_gray: np.ndarray,
    edge_discard: int = 0,
    max_features: int = 500,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10,
    n_octave_layers: int = 3,
    sigma: float = 1.6,
):
    """
    Extract SIFT keypoints and descriptors from a grayscale image.

    Returns:
        coords: (2, N) array in (row, col) format
        descriptors: (N, 128) SIFT descriptors
        responses: (N,) keypoint response strengths
    """
    img_uint8 = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)

    sift = cv2.SIFT_create(
        nfeatures=max_features,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(img_uint8, None)

    if descriptors is None or len(keypoints) == 0:
        print("WARNING: No SIFT features detected!")
        return np.array([[], []]), np.array([]), np.array([])

    coords_xy = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])

    if edge_discard > 0:
        h, w = img_gray.shape[:2]
        mask = (
            (coords_xy[:, 1] > edge_discard)
            & (coords_xy[:, 1] < h - edge_discard)
            & (coords_xy[:, 0] > edge_discard)
            & (coords_xy[:, 0] < w - edge_discard)
        )
        coords_xy, descriptors, responses = coords_xy[mask], descriptors[mask], responses[mask]

    coords = coords_xy[:, [1, 0]].T  # (x,y) -> (row,col), then (2, N)
    return coords, descriptors, responses


def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    ratio_threshold: float = 0.75,
):
    """
    Match descriptors using BFMatcher (L2) with NNDR ratio test, then build
    unique correspondence pairs (one-to-one in image 2).

    Args:
        descriptors1: (N1, 128) SIFT descriptors from image 1
        descriptors2: (N2, 128) SIFT descriptors from image 2
        keypoints1: (N1, 2) keypoint coordinates from image 1 in (row, col)
        keypoints2: (N2, 2) keypoint coordinates from image 2 in (row, col)
        ratio_threshold: NNDR threshold — keep match if d1 < ratio_threshold * d2

    Returns:
        correspondence_pairs: (M, 2, 2) unique correspondences in (row, col), M <= N1
        knn_matches: raw KNN match list from BFMatcher (for visualization)
        nndr_proportions: list of d1/d2 ratios for all KNN pairs (for histogram)
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(
        descriptors1.astype(np.float32),
        descriptors2.astype(np.float32),
        k=2,
    )

    # NNDR ratio test
    good = [m for m, n in knn_matches
            if m.distance < ratio_threshold * n.distance]

    # Build unique correspondence pairs in (row, col) format
    seen_img2 = set()
    correspondences = []
    for m in good:
        pt2 = (keypoints2[m.trainIdx, 0], keypoints2[m.trainIdx, 1])
        if pt2 not in seen_img2:
            correspondences.append([keypoints1[m.queryIdx], keypoints2[m.trainIdx]])
            seen_img2.add(pt2)

    correspondence_pairs = (
        np.array(correspondences) if correspondences else np.zeros((0, 2, 2))
    )
    nndr_proportions = [m.distance / n.distance for m, n in knn_matches if n.distance > 0]

    return correspondence_pairs, knn_matches, nndr_proportions
