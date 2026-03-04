import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.transform import resize


def select_feature_regions(img1, img2, corners1, corners2):
    """
    Interactive GUI to select rectangular regions of interest for features.
    
    Args:
        img1, img2: Images to display
        corners1, corners2: Feature corners in (2, N) format (y, x)
        
    Returns:
        region1, region2: Selected regions as (x_min, x_max, y_min, y_max) or None if cancelled
    """
    # Store selected rectangles
    selected_regions = {'img1': None, 'img2': None}
    current_selector = {'active': None}  # Track which selector is active
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('CLICK AND DRAG to draw rectangle on each image (close window when done)', 
                 fontsize=14, fontweight='bold')
    
    # Display images
    ax1.imshow(img1)
    ax1.set_title(f'Image 1 - Click and drag from corner to corner\n({corners1.shape[1]} features detected)', 
                  fontsize=12, color='blue')
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.set_title(f'Image 2 - Click and drag from corner to corner\n({corners2.shape[1]} features detected)', 
                  fontsize=12, color='red')
    ax2.axis('off')
    
    # Callback for rectangle selection
    def onselect1(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        selected_regions['img1'] = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        
        # Update title to show selection
        num_in_region = np.sum(
            (corners1[1, :] >= selected_regions['img1'][0]) &
            (corners1[1, :] <= selected_regions['img1'][1]) &
            (corners1[0, :] >= selected_regions['img1'][2]) &
            (corners1[0, :] <= selected_regions['img1'][3])
        )
        ax1.set_title(f'Image 1 - Region selected!\n({num_in_region}/{corners1.shape[1]} features in region)', 
                      fontsize=12, color='green', fontweight='bold')
        fig.canvas.draw()
    
    def onselect2(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        selected_regions['img2'] = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        
        # Update title to show selection
        num_in_region = np.sum(
            (corners2[1, :] >= selected_regions['img2'][0]) &
            (corners2[1, :] <= selected_regions['img2'][1]) &
            (corners2[0, :] >= selected_regions['img2'][2]) &
            (corners2[0, :] <= selected_regions['img2'][3])
        )
        ax2.set_title(f'Image 2 - Region selected!\n({num_in_region}/{corners2.shape[1]} features in region)', 
                      fontsize=12, color='green', fontweight='bold')
        fig.canvas.draw()
    
    # Create rectangle selectors
    selector1 = RectangleSelector(
        ax1, onselect1,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True,
        props=dict(facecolor='blue', edgecolor='blue', alpha=0.3, fill=True)
    )
    
    selector2 = RectangleSelector(
        ax2, onselect2,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True,
        props=dict(facecolor='red', edgecolor='red', alpha=0.3, fill=True)
    )
    
    # Keep references to prevent garbage collection
    current_selector['selector1'] = selector1
    current_selector['selector2'] = selector2
    
    plt.tight_layout()
    plt.show()
    
    return selected_regions['img1'], selected_regions['img2']


def filter_features_by_region(corners, features, responses, region):
    """
    Filter features to only keep those within the selected rectangular region.
    
    Args:
        corners: Feature corners in (2, N) format (y, x)
        features: Feature descriptors (N, D)
        responses: Feature responses (N,)
        region: (x_min, x_max, y_min, y_max) or None
        
    Returns:
        filtered_corners, filtered_features, filtered_responses
    """
    if region is None:
        return corners, features, responses
    
    x_min, x_max, y_min, y_max = region
    
    # Create mask for features within region
    # corners is (2, N) where corners[0] = y, corners[1] = x
    mask = (
        (corners[1, :] >= x_min) &
        (corners[1, :] <= x_max) &
        (corners[0, :] >= y_min) &
        (corners[0, :] <= y_max)
    )
    
    # Filter features
    filtered_corners = corners[:, mask]
    filtered_features = features[mask]
    filtered_responses = responses[mask]
    
    return filtered_corners, filtered_features, filtered_responses


def normalize_img(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to have values between 0 and 1 using min-max normalization.
    Args:
        img: np.ndarray - The image to normalize.
    Returns:
        np.ndarray - The normalized image.
    """
    # shift to [0, max-min]
    img_intermediate = img - img.min()

    # scale to range [0, 1]
    img = img_intermediate / (img.max() - img.min())
    return img


def dist2(x, c):
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, "Data dimension does not match dimension of centers"

    return (
        (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum((c**2).T, axis=0)
        - 2 * np.inner(x, c)
    )


def NCC(img1: np.ndarray, img2: np.ndarray) -> float:
    assert img1.shape == img2.shape, "Images must have the same shape"
    img1_normalized = (img1 - img1.mean()) / np.linalg.norm(img1)
    img2_normalized = (img2 - img2.mean()) / np.linalg.norm(img2)

    return np.sum(img1_normalized * img2_normalized)


def SSD(img1: np.ndarray, img2: np.ndarray) -> float:
    assert img1.shape == img2.shape, "Images must have the same shape"
    return np.sum((img1 - img2) ** 2)


def get_a_x(im1_pt: np.ndarray, im2_pt: np.ndarray) -> np.ndarray:
    # im_pt: (x,y)
    x1, y1 = im1_pt
    x2, y2 = im2_pt
    a_x = np.array([-x1, -y1, -1, 0, 0, 0, x1 * x2, x2 * y1, x2])
    return a_x


def get_a_y(im1_pt, im2_pt):
    x1, y1 = im1_pt
    x2, y2 = im2_pt
    a_y = np.array([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    return a_y


def compute_H(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    # points come in as (x,y)
    assert len(im1_pts) == len(im2_pts), "correspondence lengths don't match"
    A = []
    for corr in zip(im1_pts, im2_pts):
        a_x, a_y = get_a_x(*corr), get_a_y(*corr)
        A.append(a_x)
        A.append(a_y)
    A = np.array(A)

    # solve Ah = 0
    # or A^T A h = lambda h
    U, S, VT = np.linalg.svd(A, full_matrices=False)

    # want eigenvector with smallest eigenvalue of ATA. this is the last row of VT
    h = VT[-1]

    # normalize h to have norm 1
    # h = h / np.linalg.norm(h)

    h = h.reshape(3, 3)
    h *= 1 / h[2][2]

    return h


def perform_H(pt: np.ndarray, H: np.ndarray) -> np.ndarray:
    # pt comes in as (x,y)
    pt = np.append(pt, [1], axis=0)
    pt = np.array(pt).reshape(-1, 1)  # (3, 1)
    res = H @ pt
    res = res / res[2]  # (x, y, 1)

    # return as (x, y)
    return np.array([res[0], res[1]]).reshape(-1)


def get_rgb_patches_for_sift(coords: np.ndarray, img_rgb: np.ndarray, patch_size: int = 40):
    """
    Extract RGB patches at SIFT keypoint locations for visualization.

    Since SIFT descriptors are 128-D vectors computed by OpenCV, we can't visualize
    them directly like 8x8 patches. Instead, we extract RGB patches at keypoint
    locations to show what image regions the features correspond to.

    Args:
        coords: (2, N) array of keypoint coordinates in (y, x) format
        img_rgb: RGB image
        patch_size: Size of patch to extract (default 40x40)

    Returns:
        rgb_patches: List of RGB patches, each downscaled to 8x8 for consistency
    """
    rgb_patches = []
    coords_yx = coords.T  # Convert to (N, 2)

    for pt_y, pt_x in coords_yx:
        pt_y, pt_x = int(pt_y), int(pt_x)
        half_size = patch_size // 2

        min_x = max(0, pt_x - half_size)
        max_x = min(img_rgb.shape[1], pt_x + half_size)
        min_y = max(0, pt_y - half_size)
        max_y = min(img_rgb.shape[0], pt_y + half_size)

        # Extract RGB patch
        rgb_window = img_rgb[min_y:max_y, min_x:max_x]

        # Downscale to 8x8 for consistency with Harris visualization
        if rgb_window.size > 0:
            downscaled_rgb = resize(rgb_window, (8, 8), anti_aliasing=True)
            rgb_patches.append(downscaled_rgb)
        else:
            # Handle edge case where patch is too small
            rgb_patches.append(np.zeros((8, 8, 3)))

    return rgb_patches


# ========================================
# POSE UTILITIES
# ========================================

def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (in degrees).
    Uses XYZ convention (roll, pitch, yaw).
    
    Args:
        R: (3, 3) rotation matrix
    
    Returns:
        roll, pitch, yaw in degrees
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def compute_baseline(t):
    """
    Compute the baseline (magnitude of translation) between two cameras.
    
    Args:
        t: (3,) translation vector
    
    Returns:
        baseline: scalar baseline distance
    """
    return np.linalg.norm(t)


def setup_camera_coordinate_system(R, t, points_3d=None):
    """
    Set up a standard camera coordinate system where Camera 0 is the reference frame.
    
    ================================================================================
    COORDINATE SYSTEM CONVENTION:
    ================================================================================
    Camera 0 (Reference Frame):
    - Position: [0, 0, 0] (world origin)
    - Rotation: Identity matrix (no rotation)
    - Forward direction: +Z axis (points in front of camera have positive Z)
    - Right direction: +X axis (right side of camera is positive X)
    - Up direction: +Y axis (top of camera is positive Y)
    
    This is a right-handed coordinate system matching standard computer graphics 
    conventions. The world coordinate system IS Camera 0's coordinate system.
    
    Camera 1:
    - Position: -R^T @ t (world position derived from w2c transform [R|t])
    - Rotation: R^T (camera-to-world rotation)
    - All transformations are camera-to-world (c2w)

    Note on convention:
    - The pipeline uses P2 = K[R|t] where R,t are world-to-camera (w2c)
    - This function converts to camera-to-world (c2w) for visualization:
      R_c2w = R^T, position = -R^T @ t

    Visualization:
    - In matplotlib: X=Right, Y=Up, Z=Forward
    - In viser: Same convention, with coordinate frame displayed at origin

    ================================================================================

    Args:
        R: (3, 3) world-to-camera rotation matrix (from P2 = K[R|t])
        t: (3,) world-to-camera translation vector (from P2 = K[R|t])
        points_3d: (N, 3) optional 3D points (already in Camera 0's coordinate system)

    Returns:
        T0: (4, 4) camera-to-world matrix for Camera 0 (identity, at origin)
        T1: (4, 4) camera-to-world matrix for Camera 1
        points_3d_transformed: (N, 3) transformed 3D points (if provided, same as input)
        camera_poses: (2, 4, 4) array of camera poses [T0, T1]
    """
    # Camera 0: Reference frame at origin with identity rotation
    T0 = np.eye(4)

    # Camera 1: Convert from w2c [R|t] to c2w
    # In P2 = K[R|t], a world point X projects as x ~ K(RX + t)
    # Camera center: RX + t = 0 => X = -R^T @ t
    # Camera axes in world: columns of R^T
    T1 = np.eye(4)
    T1[:3, :3] = R.T  # c2w rotation (camera axes in world)
    T1[:3, 3] = (-R.T @ t).flatten()  # Camera 1 position in world
    
    # Stack camera poses
    camera_poses = np.stack([T0, T1])
    
    # Transform points if provided
    if points_3d is not None and len(points_3d) > 0:
        # Points are already in Camera 0's coordinate system (world = Camera 0)
        points_3d_transformed = points_3d.copy()
        return T0, T1, points_3d_transformed, camera_poses
    else:
        return T0, T1, None, camera_poses


def get_camera_axes(T, scale=1.0):
    """
    Get camera coordinate axes from transformation matrix.
    
    Args:
        T: (4, 4) camera transformation matrix
        scale: Scale factor for axis length
    
    Returns:
        origin: (3,) camera position
        x_axis: (3,) right direction (+X)
        y_axis: (3,) up direction (+Y)
        z_axis: (3,) forward direction (+Z)
    """
    origin = T[:3, 3]
    R = T[:3, :3]
    
    # Camera axes in world coordinates
    x_axis = R @ np.array([scale, 0, 0])  # Right
    y_axis = R @ np.array([0, scale, 0])   # Up
    z_axis = R @ np.array([0, 0, scale])   # Forward
    
    return origin, x_axis, y_axis, z_axis
