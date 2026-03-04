import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgba
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize as sk_resize
import os


def generate_correspondence_colors(num_correspondences):
    """Generate unique high-contrast colors for correspondence pairs.
    
    Args:
        num_correspondences: Number of correspondences to generate colors for
        
    Returns:
        List of RGBA color tuples
    """
    # Define a high-contrast color palette with bold, easily distinguishable colors
    base_colors = [
        '#FF0000',  # Red
        '#0000FF',  # Blue
        '#00FF00',  # Lime
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#FF8000',  # Orange
        '#8000FF',  # Purple
        '#FF0080',  # Hot Pink
        '#00FF80',  # Spring Green
        '#0080FF',  # Azure
        '#FFFF00',  # Yellow
        '#FF0040',  # Rose
        '#8000FF',  # Violet
        '#00FF40',  # Green
        '#4000FF',  # Indigo
    ]
    
    # Cycle through the base colors and interpolate if we need more colors
    if num_correspondences <= len(base_colors):
        colors = [to_rgba(base_colors[i]) for i in range(num_correspondences)]
    else:
        # If we need more colors, use tab20 which is designed for categorical data
        colors = cm.tab20(np.linspace(0, 1, num_correspondences))
    
    return colors


def plot_corners(img, corners, output_path, title=None, figsize=(12, 8)):
    """Plot an image with corner points overlaid.
    
    Args:
        img: Input image array
        corners: Corner coordinates (either 2xN array or Nx2 array)
        output_path: Path to save the output image
        title: Optional title for the plot
        figsize: Figure size tuple (width, height)
    """
    plt.figure(figsize=figsize)
    plt.imshow(img)
    
    # Handle different corner array formats
    if corners.shape[0] == 2:  # 2xN format (row, col)
        plt.scatter(corners[1], corners[0], s=1, c="r")
    else:  # Nx2 format
        plt.scatter(corners[:, 1], corners[:, 0], s=1, c="r")
    
    if title:
        plt.title(title, fontsize=16, weight="bold")
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correspondences(img, correspondences, colors, output_path, title=None, figsize=(12, 8)):
    """Plot an image with numbered, colored correspondence points.
    
    Args:
        img: Input image array
        correspondences: Nx2 array of (y, x) coordinates
        colors: List of RGBA color tuples for each correspondence
        output_path: Path to save the output image
        title: Optional title for the plot
        figsize: Figure size tuple (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    
    for idx in range(len(correspondences)):
        y, x = correspondences[idx, 0], correspondences[idx, 1]
        ax.scatter(x, y, s=50, c=[colors[idx]], edgecolors="white", linewidths=0.5)
        ax.text(
            x, y, str(idx),
            fontsize=8,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=colors[idx],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            ),
        )
    
    if title:
        ax.set_title(title, fontsize=16, weight="bold")
    
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_side_by_side_original(img1, img2, output_path):
    """Create side-by-side comparison of original images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(img1)
    ax1.set_title("Image 1", fontsize=16, weight="bold")
    ax1.axis("off")
    
    ax2.imshow(img2)
    ax2.set_title("Image 2", fontsize=16, weight="bold")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_side_by_side_corners(img1, img2, corners1, corners2, output_path, title_suffix=""):
    """Create side-by-side comparison of images with corners."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(img1)
    # Handle different corner array formats
    if corners1.shape[0] == 2:
        ax1.scatter(corners1[1], corners1[0], s=4, c="r")
        num_corners1 = corners1.shape[1]
    else:
        ax1.scatter(corners1[:, 1], corners1[:, 0], s=4, c="r")
        num_corners1 = len(corners1)
    ax1.set_title(f"Image 1 - {title_suffix} ({num_corners1} points)", fontsize=16, weight="bold")
    ax1.axis("off")
    
    ax2.imshow(img2)
    if corners2.shape[0] == 2:
        ax2.scatter(corners2[1], corners2[0], s=4, c="r")
        num_corners2 = corners2.shape[1]
    else:
        ax2.scatter(corners2[:, 1], corners2[:, 0], s=4, c="r")
        num_corners2 = len(corners2)
    ax2.set_title(f"Image 2 - {title_suffix} ({num_corners2} points)", fontsize=16, weight="bold")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_side_by_side_correspondences(img1, img2, correspondences, colors, output_path, title_suffix=""):
    """Create side-by-side comparison of images with correspondences.
    
    Args:
        img1: First image
        img2: Second image
        correspondences: Nx2x2 array where correspondences[i, 0] is img1 point and correspondences[i, 1] is img2 point
        colors: List of RGBA color tuples
        output_path: Path to save the output
        title_suffix: Suffix for the title (e.g., "Before RANSAC", "After RANSAC")
    """
    num_correspondences = len(correspondences)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot image 1 correspondences
    ax1.imshow(img1)
    for idx in range(num_correspondences):
        y, x = correspondences[idx, 0, 0], correspondences[idx, 0, 1]
        ax1.scatter(x, y, s=50, c=[colors[idx]], edgecolors="white", linewidths=0.5)
        ax1.text(
            x, y, str(idx),
            fontsize=8,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=colors[idx],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            ),
        )
    ax1.set_title(f"Image 1 - {title_suffix} ({num_correspondences} correspondences)", fontsize=16, weight="bold")
    ax1.axis("off")
    
    # Plot image 2 correspondences
    ax2.imshow(img2)
    for idx in range(num_correspondences):
        y, x = correspondences[idx, 1, 0], correspondences[idx, 1, 1]
        ax2.scatter(x, y, s=50, c=[colors[idx]], edgecolors="white", linewidths=0.5)
        ax2.text(
            x, y, str(idx),
            fontsize=8,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=colors[idx],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            ),
        )
    ax2.set_title(f"Image 2 - {title_suffix} ({num_correspondences} correspondences)", fontsize=16, weight="bold")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_feature_matching_visualization(
    img1, img2,
    corners1_anms, corners2_anms,
    correspondences,
    colors,
    output_path,
    title_suffix=""
):
    """
    Create side-by-side visualization showing matched and unmatched features with connecting lines.
    
    Args:
        img1, img2: Original images
        corners1_anms: All detected corners in image 1 (Nx2 in y,x format)
        corners2_anms: All detected corners in image 2 (Nx2 in y,x format)
        correspondences: Matched correspondences (Mx2x2 in y,x format)
        colors: Colors for each correspondence
        output_path: Path to save visualization
        title_suffix: Additional text for title (e.g., "Before RANSAC")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Extract matched points from correspondences
    matched_pts1 = correspondences[:, 0, :]  # (M, 2)
    matched_pts2 = correspondences[:, 1, :]  # (M, 2)
    
    # Create boolean masks for matched points using vectorized comparison
    # This is more efficient than set-based lookup for numpy arrays
    matched_mask1 = np.zeros(len(corners1_anms), dtype=bool)
    matched_mask2 = np.zeros(len(corners2_anms), dtype=bool)
    
    for matched_pt in matched_pts1:
        # Find exact matches (using small tolerance for floating point comparison)
        distances = np.sum((corners1_anms - matched_pt)**2, axis=1)
        matched_mask1[distances < 1e-6] = True
    
    for matched_pt in matched_pts2:
        distances = np.sum((corners2_anms - matched_pt)**2, axis=1)
        matched_mask2[distances < 1e-6] = True
    
    unmatched_pts1 = corners1_anms[~matched_mask1]
    unmatched_pts2 = corners2_anms[~matched_mask2]
    
    # === IMAGE 1 ===
    ax1.imshow(img1)
    
    # Set explicit axis limits to prevent lines from extending beyond image
    ax1.set_xlim(-0.5, img1.shape[1] - 0.5)
    ax1.set_ylim(img1.shape[0] - 0.5, -0.5)
    
    # Plot unmatched points (red)
    if len(unmatched_pts1) > 0:
        ax1.scatter(unmatched_pts1[:, 1], unmatched_pts1[:, 0], 
                   s=20, c='red', marker='o', alpha=0.6, label='Unmatched')
    
    # Plot matched points (green) and draw connection lines
    neon_green = '#39FF14'  # Neon green color
    for idx in range(len(matched_pts1)):
        y1, x1 = matched_pts1[idx]
        y2, x2 = matched_pts2[idx]
        
        # Draw green point
        ax1.scatter(x1, y1, s=30, c=neon_green, marker='o', 
                   edgecolors='white', linewidths=0.5, alpha=0.9)
        
        # Draw neon green line from this point to the right edge
        # Line goes to the edge but stays within image bounds
        ax1.plot([x1, img1.shape[1] - 1], [y1, y1], 
                color=neon_green, linewidth=2.5, alpha=0.7)
    
    # Add legend to first image only
    matched_label = ax1.scatter([], [], s=30, c=neon_green, marker='o', 
                               edgecolors='white', linewidths=0.5, label='Matched')
    ax1.legend(loc='upper right', fontsize=10)
    
    ax1.set_title(f'Image 1 (After Feature Matching)\n{title_suffix}\n'
                 f'Matched: {len(matched_pts1)} | Unmatched: {len(unmatched_pts1)} | Total: {len(corners1_anms)}',
                 fontsize=14, weight='bold')
    ax1.axis('off')
    
    # === IMAGE 2 ===
    ax2.imshow(img2)
    
    # Set explicit axis limits to prevent lines from extending beyond image
    ax2.set_xlim(-0.5, img2.shape[1] - 0.5)
    ax2.set_ylim(img2.shape[0] - 0.5, -0.5)
    
    # Plot unmatched points (red)
    if len(unmatched_pts2) > 0:
        ax2.scatter(unmatched_pts2[:, 1], unmatched_pts2[:, 0], 
                   s=20, c='red', marker='o', alpha=0.6, label='Unmatched')
    
    # Plot matched points (green) and draw connection lines
    neon_green = '#39FF14'  # Neon green color
    for idx in range(len(matched_pts2)):
        y1, x1 = matched_pts1[idx]
        y2, x2 = matched_pts2[idx]
        
        # Draw green point
        ax2.scatter(x2, y2, s=30, c=neon_green, marker='o', 
                   edgecolors='white', linewidths=0.5, alpha=0.9)
        
        # Draw neon green line from the left edge to this point
        # Line starts from the edge but stays within image bounds
        ax2.plot([0, x2], [y2, y2], 
                color=neon_green, linewidth=2.5, alpha=0.7)
    
    # Add legend
    ax2.scatter([], [], s=30, c=neon_green, marker='o', 
               edgecolors='white', linewidths=0.5, label='Matched')
    ax2.legend(loc='upper right', fontsize=10)
    
    ax2.set_title(f'Image 2 (After Feature Matching)\n{title_suffix}\n'
                 f'Matched: {len(matched_pts2)} | Unmatched: {len(unmatched_pts2)} | Total: {len(corners2_anms)}',
                 fontsize=14, weight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ========================================
# NEW: POSE ESTIMATION VISUALIZATIONS
# ========================================

def plot_3d_points(points_3d, output_path, title="3D Reconstruction", figsize=(12, 10)):
    """
    Plot triangulated 3D points.
    
    Args:
        points_3d: (N, 3) array of 3D points
        output_path: Path to save the plot
        title: Title for the plot
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c='b', marker='o', s=20, alpha=0.6)
    
    ax.set_xlabel('X (Right)', fontsize=12)
    ax.set_ylabel('Y (Up)', fontsize=12)
    ax.set_zlabel('Z (Forward)', fontsize=12)
    ax.set_title(title, fontsize=16, weight="bold")
    
    # Set equal aspect ratio
    max_range = np.array([
        points_3d[:, 0].max() - points_3d[:, 0].min(),
        points_3d[:, 1].max() - points_3d[:, 1].min(),
        points_3d[:, 2].max() - points_3d[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_camera_poses(R, t, points_3d, output_path, title="Camera Poses and 3D Points", figsize=(14, 10)):
    """
    Visualize camera poses and 3D points using Camera 0 as reference coordinate system.
    
    Coordinate system:
    - Camera 0 at origin [0, 0, 0] with identity rotation
    - Forward: +Z, Right: +X, Up: +Y
    
    Args:
        R: (3, 3) rotation matrix from Camera 0 to Camera 1
        t: (3,) translation vector from Camera 0 to Camera 1
        points_3d: (N, 3) array of 3D points (in Camera 0's coordinate system)
        output_path: Path to save the plot
        title: Title for the plot
        figsize: Figure size
    """
    from utils import setup_camera_coordinate_system, get_camera_axes
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up coordinate system with Camera 0 as reference
    T0, T1, _, _ = setup_camera_coordinate_system(R, t)
    
    # Get camera positions and axes
    cam0_pos, cam0_x, cam0_y, cam0_z = get_camera_axes(T0, scale=0.5)
    cam1_pos, cam1_x, cam1_y, cam1_z = get_camera_axes(T1, scale=0.5)
    
    # Plot 3D points
    if points_3d is not None and len(points_3d) > 0:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                   c='b', marker='o', s=10, alpha=0.3, label='3D Points')
    
    # Plot Camera 0 (reference) with axes
    ax.scatter(*cam0_pos, c='r', marker='^', s=200, label='Camera 0 (Reference)')
    ax.quiver(*cam0_pos, *cam0_z, length=0.5, color='r', arrow_length_ratio=0.3, label='Forward (Z)')
    ax.quiver(*cam0_pos, *cam0_x, length=0.3, color='orange', arrow_length_ratio=0.3, alpha=0.7, linestyle='--')
    ax.quiver(*cam0_pos, *cam0_y, length=0.3, color='yellow', arrow_length_ratio=0.3, alpha=0.7, linestyle='--')
    
    # Plot Camera 1 with axes
    ax.scatter(*cam1_pos, c='g', marker='^', s=200, label='Camera 1')
    ax.quiver(*cam1_pos, *cam1_z, length=0.5, color='g', arrow_length_ratio=0.3)
    
    # Draw line between cameras
    ax.plot([cam0_pos[0], cam1_pos[0]], 
            [cam0_pos[1], cam1_pos[1]], 
            [cam0_pos[2], cam1_pos[2]], 
            'k--', linewidth=2, label='Baseline')
    
    ax.set_xlabel('X (Right)', fontsize=12)
    ax.set_ylabel('Y (Up)', fontsize=12)
    ax.set_zlabel('Z (Forward)', fontsize=12)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.legend(loc='upper left')
    
    # Set viewing angle to show coordinate system clearly
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_epipolar_lines(img1, img2, pts1, pts2, F, output_path, num_lines=10, figsize=(20, 10)):
    """
    Plot epipolar lines on both images.
    
    Args:
        img1, img2: Images
        pts1, pts2: Corresponding points (N, 2) in (y, x) format
        F: Fundamental matrix
        output_path: Path to save the plot
        num_lines: Number of epipolar lines to draw
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Convert to (x, y) format for computation
    pts1_xy = pts1[:, [1, 0]]
    pts2_xy = pts2[:, [1, 0]]
    
    # Select subset of points to visualize
    num_lines = min(num_lines, len(pts1))
    indices = np.linspace(0, len(pts1) - 1, num_lines, dtype=int)
    
    # Image 1: Draw points and their epipolar lines in image 2
    ax1.imshow(img1)
    for i, idx in enumerate(indices):
        color = cm.hsv(i / num_lines)
        y, x = pts1[idx]
        ax1.scatter(x, y, c=[color], s=100, marker='o', edgecolors='white', linewidths=2)
    ax1.set_title("Image 1 - Feature Points", fontsize=16, weight="bold")
    ax1.axis("off")
    
    # Image 2: Draw corresponding points and epipolar lines
    ax2.imshow(img2)
    h, w = img2.shape[:2]
    
    for i, idx in enumerate(indices):
        color = cm.hsv(i / num_lines)
        
        # Draw point in image 2
        y, x = pts2[idx]
        ax2.scatter(x, y, c=[color], s=100, marker='o', edgecolors='white', linewidths=2)
        
        # Compute epipolar line in image 2 from point in image 1
        pt1_homo = np.array([pts1_xy[idx, 0], pts1_xy[idx, 1], 1])
        epipolar_line = F @ pt1_homo  # line in image 2: ax + by + c = 0
        
        # Draw the line
        a, b, c = epipolar_line
        if np.abs(b) > 1e-6:  # Not vertical
            x_line = np.array([0, w])
            y_line = -(a * x_line + c) / b
        else:  # Vertical line
            x_line = np.array([-c/a, -c/a])
            y_line = np.array([0, h])
        
        ax2.plot(x_line, y_line, c=color, linewidth=2, alpha=0.7)
    
    ax2.set_title("Image 2 - Epipolar Lines", fontsize=16, weight="bold")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_pose_summary_plot(R, t, points_3d, roll, pitch, yaw, baseline, 
                             num_inliers, total_correspondences, output_path):
    """
    Create a comprehensive summary plot showing pose estimation results.
    
    Args:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        points_3d: (N, 3) triangulated 3D points
        roll, pitch, yaw: Euler angles in degrees
        baseline: Baseline distance
        num_inliers: Number of inliers from RANSAC
        total_correspondences: Total number of correspondences
        output_path: Path to save the plot
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 3D visualization of cameras and points
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    from utils import setup_camera_coordinate_system, get_camera_axes
    
    # Set up coordinate system with Camera 0 as reference
    T0, T1, _, _ = setup_camera_coordinate_system(R, t)
    cam0_pos, _, _, cam0_z = get_camera_axes(T0, scale=baseline*0.3)
    cam1_pos, _, _, cam1_z = get_camera_axes(T1, scale=baseline*0.3)
    
    # Plot 3D points
    if points_3d is not None and len(points_3d) > 0:
        ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                   c='b', marker='o', s=5, alpha=0.3)
    
    # Plot cameras
    ax1.scatter(*cam0_pos, c='r', marker='^', s=200)
    ax1.quiver(*cam0_pos, *cam0_z, length=baseline*0.3, color='r', arrow_length_ratio=0.3)
    ax1.scatter(*cam1_pos, c='g', marker='^', s=200)
    ax1.quiver(*cam1_pos, *cam1_z, length=baseline*0.3, color='g', arrow_length_ratio=0.3)
    ax1.plot([cam0_pos[0], cam1_pos[0]], 
            [cam0_pos[1], cam1_pos[1]], 
            [cam0_pos[2], cam1_pos[2]], 
            'k--', linewidth=2)
    
    ax1.set_xlabel('X (Right)')
    ax1.set_ylabel('Y (Up)')
    ax1.set_zlabel('Z (Forward)')
    ax1.set_title('Camera Configuration & 3D Points', fontsize=14, weight="bold")
    ax1.view_init(elev=20, azim=45)
    
    # Rotation matrix visualization
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(R, cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_title('Rotation Matrix', fontsize=14, weight="bold")
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{R[i, j]:.3f}', ha='center', va='center', 
                    color='white' if abs(R[i, j]) > 0.5 else 'black', fontsize=12, weight='bold')
    plt.colorbar(im, ax=ax2)
    
    # Text summary
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    
    summary_text = f"""
    POSE ESTIMATION SUMMARY
    
    Rotation (Euler Angles):
      Roll:  {roll:8.2f}°
      Pitch: {pitch:8.2f}°
      Yaw:   {yaw:8.2f}°
    
    Translation:
      X: {t[0]:8.4f}
      Y: {t[1]:8.4f}
      Z: {t[2]:8.4f}
    
    Baseline: {baseline:.4f} units
    
    RANSAC Results:
      Inliers: {num_inliers} / {total_correspondences}
      Inlier Ratio: {100*num_inliers/total_correspondences:.1f}%
    
    3D Reconstruction:
      {len(points_3d) if points_3d is not None else 0} points triangulated
    """
    
    ax3.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3D points histogram (depth distribution)
    ax4 = fig.add_subplot(2, 2, 4)
    if points_3d is not None and len(points_3d) > 0:
        depths = points_3d[:, 2]
        ax4.hist(depths, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Depth (Z)', fontsize=12)
        ax4.set_ylabel('Number of Points', fontsize=12)
        ax4.set_title('Depth Distribution of 3D Points', fontsize=14, weight="bold")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No 3D points available', ha='center', va='center', fontsize=14)
        ax4.axis('off')
    
    plt.suptitle('Pose Estimation Results', fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_comprehensive_pipeline_grid(
    img1, img2,
    corners1, corners2,
    corners1_anms, corners2_anms,
    correspondences_before,
    correspondences_after,
    colors_before,
    colors_after,
    R, t, points_3d,
    output_path,
    feature_type="harris",
    sift_use_rootsift=False,
    config=None,
    ransac_failed=False,
    F=None
):
    """
    Create a comprehensive visualization grid showing all pipeline steps.
    
    Args:
        img1, img2: Original images
        corners1, corners2: All detected features (2xN or Nx2)
        corners1_anms, corners2_anms: ANMS features (or all SIFT features if ANMS not used)
        correspondences_before: Correspondences before RANSAC (Nx2x2)
        correspondences_after: Correspondences after RANSAC (Nx2x2)
        colors_before, colors_after: Colors for correspondences
        R, t: Rotation and translation
        points_3d: Triangulated 3D points
        output_path: Path to save the grid
        feature_type: "harris" or "sift"
        sift_use_rootsift: Whether RootSIFT is being used (only relevant if feature_type="sift")
        config: Configuration object with pipeline parameters (optional)
        ransac_failed: Whether RANSAC pose estimation failed (optional)
    """
    fig = plt.figure(figsize=(24, 20))
    
    # Row 1: Original images
    ax1 = plt.subplot(5, 4, 1)
    ax1.imshow(img1)
    ax1.set_title('Image 1: Original', fontsize=12, weight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(5, 4, 2)
    ax2.imshow(img2)
    ax2.set_title('Image 2: Original', fontsize=12, weight='bold')
    ax2.axis('off')
    
    # Row 1: Feature detection (Harris corners or SIFT)
    # Determine feature detection method label
    if feature_type == "sift":
        feature_label = "RootSIFT" if sift_use_rootsift else "SIFT"
    else:
        feature_label = "Harris Corners"
    
    ax3 = plt.subplot(5, 4, 3)
    ax3.imshow(img1)
    if corners1.shape[0] == 2:
        ax3.scatter(corners1[1], corners1[0], s=1, c='r', alpha=0.5)
        num_corners1 = corners1.shape[1]
    else:
        ax3.scatter(corners1[:, 1], corners1[:, 0], s=1, c='r', alpha=0.5)
        num_corners1 = len(corners1)
    ax3.set_title(f'{feature_label} ({num_corners1})', fontsize=12, weight='bold')
    ax3.axis('off')
    
    ax4 = plt.subplot(5, 4, 4)
    ax4.imshow(img2)
    if corners2.shape[0] == 2:
        ax4.scatter(corners2[1], corners2[0], s=1, c='r', alpha=0.5)
        num_corners2 = corners2.shape[1]
    else:
        ax4.scatter(corners2[:, 1], corners2[:, 0], s=1, c='r', alpha=0.5)
        num_corners2 = len(corners2)
    ax4.set_title(f'{feature_label} ({num_corners2})', fontsize=12, weight='bold')
    ax4.axis('off')
    
    # Row 2: ANMS (or raw features if ANMS not used)
    use_anms = getattr(config, 'sift_use_anms', False) if config else False
    anms_label = "ANMS" if use_anms else f"{feature_label} (No ANMS)"

    ax5 = plt.subplot(5, 4, 5)
    ax5.imshow(img1)
    ax5.scatter(corners1_anms[:, 1], corners1_anms[:, 0], s=3, c='r')
    ax5.set_title(f'{anms_label} ({len(corners1_anms)})', fontsize=12, weight='bold')
    ax5.axis('off')

    ax6 = plt.subplot(5, 4, 6)
    ax6.imshow(img2)
    ax6.scatter(corners2_anms[:, 1], corners2_anms[:, 0], s=3, c='r')
    ax6.set_title(f'{anms_label} ({len(corners2_anms)})', fontsize=12, weight='bold')
    ax6.axis('off')
    
    # Row 2: NNDR Top 5 visualization and Epipolar Lines
    ax7 = plt.subplot(5, 4, 7)
    ax7.axis('off')
    # Load and display NNDR top 5 visualization if it exists
    output_dir = os.path.dirname(output_path)
    nndr_top5_path = os.path.join(output_dir, 'step3_feature_matching_top5_nndr.png')
    if os.path.exists(nndr_top5_path):
        import skimage.io as skio
        nndr_img = skio.imread(nndr_top5_path)
        ax7.imshow(nndr_img)
        ax7.set_title('NNDR Top 5 Matches', fontsize=12, weight='bold')
    else:
        ax7.text(0.5, 0.5, 'NNDR visualization\nnot available', ha='center', va='center', fontsize=10)
    
    ax8 = plt.subplot(5, 4, 8)
    ax8.axis('off')
    # Load and display NNDR ratio histogram if it exists
    nndr_hist_path = os.path.join(output_dir, 'step3_feature_matching_nndr_histogram.png')
    if os.path.exists(nndr_hist_path):
        import skimage.io as skio
        nndr_hist_img = skio.imread(nndr_hist_path)
        ax8.imshow(nndr_hist_img)
        ax8.set_title('NNDR Ratio Histogram', fontsize=12, weight='bold')
    else:
        ax8.text(0.5, 0.5, 'NNDR histogram\nnot available', ha='center', va='center', fontsize=10)
    
    # Row 3: 3D reconstruction
    ax9 = plt.subplot(5, 4, 9, projection='3d')
    if points_3d is not None and len(points_3d) > 0:
        ax9.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                    c='b', marker='o', s=10, alpha=0.6)
        ax9.set_xlabel('X (Right)', fontsize=8)
        ax9.set_ylabel('Y (Up)', fontsize=8)
        ax9.set_zlabel('Z (Forward)', fontsize=8)
        ax9.view_init(elev=20, azim=45)
    ax9.set_title(f'3D Points ({len(points_3d) if points_3d is not None else 0})', fontsize=12, weight='bold')
    
    # Row 3: Camera poses
    ax10 = plt.subplot(5, 4, 10, projection='3d')
    from utils import setup_camera_coordinate_system, get_camera_axes
    
    # Set up coordinate system with Camera 0 as reference
    T0, T1, _, _ = setup_camera_coordinate_system(R, t)
    cam0_pos, _, _, cam0_z = get_camera_axes(T0, scale=0.3)
    cam1_pos, _, _, cam1_z = get_camera_axes(T1, scale=0.3)
    
    if points_3d is not None and len(points_3d) > 0:
        ax10.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                    c='b', marker='o', s=5, alpha=0.3)
    
    ax10.scatter(*cam0_pos, c='r', marker='^', s=100, label='Cam 0')
    ax10.quiver(*cam0_pos, *cam0_z, length=0.3, color='r', arrow_length_ratio=0.3)
    ax10.scatter(*cam1_pos, c='g', marker='^', s=100, label='Cam 1')
    ax10.quiver(*cam1_pos, *cam1_z, length=0.3, color='g', arrow_length_ratio=0.3)
    ax10.plot([cam0_pos[0], cam1_pos[0]],
             [cam0_pos[1], cam1_pos[1]],
             [cam0_pos[2], cam1_pos[2]],
             'k--', linewidth=1.5)
    
    ax10.set_xlabel('X (Right)', fontsize=8)
    ax10.set_ylabel('Y (Up)', fontsize=8)
    ax10.set_zlabel('Z (Forward)', fontsize=8)
    ax10.set_title('Camera Poses', fontsize=12, weight='bold')
    ax10.view_init(elev=20, azim=45)
    ax10.legend(fontsize=8)
    
    # Row 3: Matches Before RANSAC (moved from row 2)
    ax11 = plt.subplot(5, 4, 11)
    ax11.imshow(img1)
    num_corr_before = len(correspondences_before)
    show_labels_before = num_corr_before <= 50
    for idx in range(num_corr_before):
        y, x = correspondences_before[idx, 0, 0], correspondences_before[idx, 0, 1]
        if show_labels_before:
            ax11.scatter(x, y, s=30, c=[colors_before[idx]], edgecolors='white', linewidths=0.5, alpha=0.8)
            ax11.text(
                x, y, str(idx),
                fontsize=6,
                ha="center",
                va="center",
                color="white",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=colors_before[idx],
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.3,
                ),
            )
        else:
            ax11.scatter(x, y, s=5, c=[colors_before[idx]], alpha=0.6)
    ax11.set_title(f'Matches Before RANSAC ({num_corr_before})', fontsize=12, weight='bold')
    ax11.axis('off')

    ax12 = plt.subplot(5, 4, 12)
    ax12.imshow(img2)
    for idx in range(num_corr_before):
        y, x = correspondences_before[idx, 1, 0], correspondences_before[idx, 1, 1]
        if show_labels_before:
            ax12.scatter(x, y, s=30, c=[colors_before[idx]], edgecolors='white', linewidths=0.5, alpha=0.8)
            ax12.text(
                x, y, str(idx),
                fontsize=6,
                ha="center",
                va="center",
                color="white",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=colors_before[idx],
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.3,
                ),
            )
        else:
            ax12.scatter(x, y, s=5, c=[colors_before[idx]], alpha=0.6)
    ax12.set_title(f'Matches Before RANSAC ({num_corr_before})', fontsize=12, weight='bold')
    ax12.axis('off')
    
    # Bottom row: 2 combined summary text boxes (left side)
    ax13 = plt.subplot(5, 4, 13)
    ax13.axis('off')
    
    # Build feature detection config text
    if config and feature_type == "sift":
        config_lines = f"""
    Config:
      Max features: {config.sift_max_features}
      Contrast thresh: {config.sift_contrast_threshold}
      Edge thresh: {config.sift_edge_threshold}"""
    elif config and feature_type == "harris":
        config_lines = f"""
    Config:
      Edge discard: {config.harris_corner_edge_discard}
      ANMS points: {config.anms_num_points}
      C_robust: {config.anms_c_robust}"""
    else:
        config_lines = ""
    
    # Build matching config text
    if config:
        matching_config = f"""
    Config:
      Metric: L2
      NNDR threshold: {config.feature_matching_ratio_threshold}"""
    else:
        matching_config = ""
    
    # Combine feature detection and matching text
    combined_left_text = f"""
    FEATURE DETECTION
    
    {feature_label}:
      Image 1: {num_corners1}
      Image 2: {num_corners2}
    
    ANMS Selected:
      Image 1: {len(corners1_anms)}
      Image 2: {len(corners2_anms)}{config_lines}
    
    ─────────────────────────────────
    
    FEATURE MATCHING
    
    Before RANSAC:
      Correspondences: {len(correspondences_before)}
    
    After RANSAC:
      Inliers: {len(correspondences_after)}
      Inlier Ratio: {f'{100*len(correspondences_after)/len(correspondences_before):.1f}%' if len(correspondences_before) > 0 else 'N/A'}{matching_config}
    """
    ax13.text(0.05, 0.5, combined_left_text, fontsize=8, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax14 = plt.subplot(5, 4, 14)
    ax14.axis('off')
    
    # Build RANSAC config text
    if config:
        ransac_config = f"""
    RANSAC Config:
      Sample size (s): {config.ransac_s}
      Epsilon: {config.ransac_epsilon}
      Iterations: {config.ransac_num_iters}"""
    else:
        ransac_config = ""
    
    # Build reconstruction config text
    if config:
        recon_config = f"""
    Triangulation Config:
      Method: Sparse features
      Max reproj error: 50.0px
      Depth range: [0.01, 10000]"""
    else:
        recon_config = ""
    
    # Combine pose estimation and reconstruction text
    combined_right_text = f"""
    POSE ESTIMATION
    
    Translation:
      X: {t[0]:.4f}
      Y: {t[1]:.4f}
      Z: {t[2]:.4f}
    
    Baseline: {np.linalg.norm(t):.4f}{ransac_config}
    
    ─────────────────────────────────
    
    3D RECONSTRUCTION
    
    Method: Sparse
    Points: {len(points_3d) if points_3d is not None else 0}
    Mean depth: {np.mean(points_3d[:, 2]) if points_3d is not None and len(points_3d) > 0 else 0:.2f}{recon_config}
    """
    ax14.text(0.05, 0.5, combined_right_text, fontsize=8, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Bottom row: Inliers after RANSAC (right side)
    ax15 = plt.subplot(5, 4, 15)
    ax15.imshow(img1)
    num_corr_after = len(correspondences_after)
    show_labels_after = num_corr_after <= 50
    for idx in range(num_corr_after):
        y, x = correspondences_after[idx, 0, 0], correspondences_after[idx, 0, 1]
        if show_labels_after:
            ax15.scatter(x, y, s=30, c=[colors_after[idx]], edgecolors='white', linewidths=0.5, alpha=0.9)
            ax15.text(
                x, y, str(idx),
                fontsize=6,
                ha="center",
                va="center",
                color="white",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=colors_after[idx],
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.3,
                ),
            )
        else:
            ax15.scatter(x, y, s=5, c=[colors_after[idx]], alpha=0.7)
    ax15.set_title(f'Inliers After RANSAC ({num_corr_after})', fontsize=12, weight='bold')
    ax15.axis('off')

    ax16 = plt.subplot(5, 4, 16)
    ax16.imshow(img2)
    for idx in range(num_corr_after):
        y, x = correspondences_after[idx, 1, 0], correspondences_after[idx, 1, 1]
        if show_labels_after:
            ax16.scatter(x, y, s=30, c=[colors_after[idx]], edgecolors='white', linewidths=0.5, alpha=0.9)
            ax16.text(
                x, y, str(idx),
                fontsize=6,
                ha="center",
                va="center",
                color="white",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=colors_after[idx],
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.3,
                ),
            )
        else:
            ax16.scatter(x, y, s=5, c=[colors_after[idx]], alpha=0.7)
    ax16.set_title(f'Inliers After RANSAC ({num_corr_after})', fontsize=12, weight='bold')
    ax16.axis('off')
    
    # Row 5: Epipolar lines in both directions
    if F is not None and len(correspondences_after) > 0:
        # Use inlier correspondences for epipolar visualization
        pts1 = correspondences_after[:, 0, :]  # (N, 2) in (y, x)
        pts2 = correspondences_after[:, 1, :]  # (N, 2) in (y, x)
        pts1_xy = pts1[:, [1, 0]]  # Convert to (x, y)
        pts2_xy = pts2[:, [1, 0]]  # Convert to (x, y)

        num_epi_lines = min(15, len(pts1))
        epi_indices = np.linspace(0, len(pts1) - 1, num_epi_lines, dtype=int)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Slot 17-18: Points in img1 → Epipolar lines in img2
        ax17 = plt.subplot(5, 4, 17)
        ax17.imshow(img1)
        for i, idx in enumerate(epi_indices):
            color = cm.hsv(i / num_epi_lines)
            y, x = pts1[idx]
            ax17.scatter(x, y, c=[color], s=60, marker='o', edgecolors='white', linewidths=1.5)
        ax17.set_title('Image 1 - Feature Points', fontsize=12, weight='bold')
        ax17.axis('off')

        ax18 = plt.subplot(5, 4, 18)
        ax18.imshow(img2)
        for i, idx in enumerate(epi_indices):
            color = cm.hsv(i / num_epi_lines)
            y, x = pts2[idx]
            ax18.scatter(x, y, c=[color], s=60, marker='o', edgecolors='white', linewidths=1.5)
            # Epipolar line in img2 from point in img1: l2 = F @ p1
            pt1_homo = np.array([pts1_xy[idx, 0], pts1_xy[idx, 1], 1])
            epi_line = F @ pt1_homo
            a, b, c = epi_line
            if np.abs(b) > 1e-6:
                x_line = np.array([0, w2])
                y_line = -(a * x_line + c) / b
            else:
                x_line = np.array([-c/a, -c/a])
                y_line = np.array([0, h2])
            ax18.plot(x_line, y_line, c=color, linewidth=1.5, alpha=0.7)
        ax18.set_title('Image 2 - Epipolar Lines', fontsize=12, weight='bold')
        ax18.axis('off')

        # Slot 19-20: Points in img2 → Epipolar lines in img1
        ax19 = plt.subplot(5, 4, 19)
        ax19.imshow(img2)
        for i, idx in enumerate(epi_indices):
            color = cm.hsv(i / num_epi_lines)
            y, x = pts2[idx]
            ax19.scatter(x, y, c=[color], s=60, marker='o', edgecolors='white', linewidths=1.5)
        ax19.set_title('Image 2 - Feature Points', fontsize=12, weight='bold')
        ax19.axis('off')

        ax20 = plt.subplot(5, 4, 20)
        ax20.imshow(img1)
        for i, idx in enumerate(epi_indices):
            color = cm.hsv(i / num_epi_lines)
            y, x = pts1[idx]
            ax20.scatter(x, y, c=[color], s=60, marker='o', edgecolors='white', linewidths=1.5)
            # Epipolar line in img1 from point in img2: l1 = F^T @ p2
            pt2_homo = np.array([pts2_xy[idx, 0], pts2_xy[idx, 1], 1])
            epi_line = F.T @ pt2_homo
            a, b, c = epi_line
            if np.abs(b) > 1e-6:
                x_line = np.array([0, w1])
                y_line = -(a * x_line + c) / b
            else:
                x_line = np.array([-c/a, -c/a])
                y_line = np.array([0, h1])
            ax20.plot(x_line, y_line, c=color, linewidth=1.5, alpha=0.7)
        ax20.set_title('Image 1 - Epipolar Lines', fontsize=12, weight='bold')
        ax20.axis('off')
    else:
        # No epipolar data available (e.g., RANSAC failed)
        for slot in [17, 18, 19, 20]:
            ax = plt.subplot(5, 4, slot)
            ax.axis('off')
            ax.text(0.5, 0.5, 'Epipolar visualization\nnot available',
                    ha='center', va='center', fontsize=10)

    plt.suptitle(f'Complete Pose Estimation Pipeline{" (RANSAC FAILED)" if ransac_failed else ""}',
                 fontsize=20, weight='bold', y=0.99, color='red' if ransac_failed else 'black')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_nndr_histogram(proportions, ratio_threshold, output_dir):
    """Plot NNDR ratio distribution histogram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(proportions, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=ratio_threshold, color='red', linestyle='--', linewidth=2,
               label=f'NNDR threshold = {ratio_threshold}')
    ax.set_xlabel('NNDR (Nearest Neighbor Distance Ratio)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'NNDR Distribution (L2)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{output_dir}/step3_feature_matching_nndr_histogram.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"      Saved NNDR histogram to {path}")
    plt.close()


def plot_nndr_histogram_no_threshold(proportions, output_dir):
    """Plot NNDR ratio distribution histogram without any threshold annotation."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(proportions, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('NNDR (Nearest Neighbor Distance Ratio)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'NNDR Distribution (L2)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{output_dir}/step3_feature_matching_nndr_histogram_no_threshold.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"      Saved NNDR histogram (no threshold) to {path}")
    plt.close()


def plot_top_matches(knn_matches, corners1, corners2, img1_rgb, img2_rgb,
                     output_dir, num_plot=5, patch_size=40):
    """Plot top N best feature matches by NNDR using RGB patches."""
    match_info = []
    for m, n in knn_matches:
        if n.distance > 0:
            nndr = m.distance / n.distance
            match_info.append((nndr, m.queryIdx, m.trainIdx, n.trainIdx))
    match_info.sort(key=lambda x: x[0])
    top = match_info[:min(num_plot, len(match_info))]
    if not top:
        return

    def _extract_patch(img, y, x):
        half = patch_size // 2
        y, x = int(y), int(x)
        patch = img[max(0, y - half):min(img.shape[0], y + half),
                     max(0, x - half):min(img.shape[1], x + half)]
        return sk_resize(patch, (8, 8), anti_aliasing=True) if patch.size > 0 else np.zeros((8, 8, 3))

    num_rows = len(top)
    fig, axs = plt.subplots(num_rows, 3, figsize=(9, 3 * num_rows))
    if num_rows == 1:
        axs = axs.reshape(1, -1)

    for i, (nndr, q_idx, t1_idx, t2_idx) in enumerate(top):
        p1 = _extract_patch(img1_rgb, corners1[q_idx, 0], corners1[q_idx, 1])
        p2_nn = _extract_patch(img2_rgb, corners2[t1_idx, 0], corners2[t1_idx, 1])
        p2_2nn = _extract_patch(img2_rgb, corners2[t2_idx, 0], corners2[t2_idx, 1])

        axs[i, 0].imshow(np.clip(p1, 0, 1))
        axs[i, 1].imshow(np.clip(p2_nn, 0, 1))
        axs[i, 2].imshow(np.clip(p2_2nn, 0, 1))

        axs[i, 0].set_title(f'Image 1 Feature #{i+1}', fontsize=10)
        axs[i, 1].set_title(f'Image 2 NN (1st)', fontsize=10)
        axs[i, 2].set_title(f'Image 2 2NN (2nd)', fontsize=10)
        for j in range(3):
            axs[i, j].axis('off')

        fig.text(0.02, 1 - (i + 0.5) / num_rows, f'NNDR={nndr:.3f}',
                 va='center', ha='left', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.text(0.25, 0.96, 'Image 1 Feature', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.52, 0.96, 'Nearest Neighbor (1st)', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.78, 0.96, 'Second Nearest (2nd)', ha='center', fontsize=12, fontweight='bold')
    plt.suptitle(f'Top {len(top)} Best Feature Matches by NNDR (L2) - RGB',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0.08, 0, 1, 0.94])

    path = f"{output_dir}/step3_feature_matching_top{len(top)}_nndr.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"      Saved top {len(top)} NNDR features to {path}")
    plt.close()


def plot_ransac_convergence(num_iters, inlier_counts, best_so_far,
                            improvement_iters, improvement_vals,
                            best_num_inliers, epsilon, output_dir):
    """Plot RANSAC convergence: inlier count per iteration and running best."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(range(num_iters), inlier_counts, alpha=0.3, s=10, color='gray',
               label='Inlier count per iteration')
    ax.plot(range(num_iters), best_so_far, color='blue', linewidth=2,
            label='Best inlier count so far')
    if improvement_iters:
        ax.scatter(improvement_iters, improvement_vals, color='red', s=100,
                   marker='*', zorder=5, label='New best found',
                   edgecolors='darkred', linewidths=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Inliers', fontsize=12)
    ax.set_title(f'RANSAC Convergence (Final: {best_num_inliers} inliers, '
                 f'\u03b5={epsilon})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{output_dir}/step4_ransac_convergence.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"      Saved RANSAC convergence plot to {path}")
    plt.close()
