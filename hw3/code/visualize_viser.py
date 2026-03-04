"""
Visualize pose estimation results in viser.

Usage:
    python visualize_viser.py <path_to_scene_data.npz>

Example:
    python visualize_viser.py outputs/pose_estimation_harris=20_anms=250_metric=NCC_7/scene_data.npz
"""

import numpy as np
import viser
import sys
import os
from pathlib import Path


def visualize_scene(npz_path: str, port: int = 8081):
    """
    Visualize camera poses and 3D point cloud in viser.

    Args:
        npz_path: Path to the NPZ file containing scene data
        port: Starting port to run the viser server on (default: 8081).
              If port is in use, will automatically try next available port.
    """
    # Load data
    print(f"Loading scene data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    points_3d = data['points_3d']
    point_colors = data['point_colors']
    camera_poses = data['camera_poses']
    K = data['K']
    R = data['R']
    t = data['t']
    num_inliers = data['num_inliers']
    baseline = data['baseline']

    # Rotate from OpenCV convention (X-right, Y-down, Z-forward) to viser's
    # Z-up world frame, so Camera 0 appears horizontal and faces into the scene.
    #   OpenCV +X  →  viser -X  (consistent when viewed from behind camera)
    #   OpenCV +Y  →  viser -Z  (down maps to -up)
    #   OpenCV +Z  →  viser -Y  (forward maps into scene, away from viewer)
    R_fix = np.array([
        [-1,  0,  0, 0],
        [ 0,  0, -1, 0],
        [ 0, -1,  0, 0],
        [ 0,  0,  0, 1],
    ], dtype=float)
    camera_poses = np.array([R_fix @ T for T in camera_poses])
    ones = np.ones((len(points_3d), 1))
    points_3d = (R_fix @ np.hstack([points_3d, ones]).T).T[:, :3]

    # Re-derive R and t from the fully-transformed Camera 2 pose so the GUI
    # display stays consistent with the transformed scene.
    # camera_poses[1] is c2w; convert back to w2c: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ pos
    R_c2w_1 = camera_poses[1][:3, :3]
    pos_1 = camera_poses[1][:3, 3]
    R = R_c2w_1.T
    t = -R_c2w_1.T @ pos_1

    # Load images if available
    img1 = data.get('img1', None)
    img2 = data.get('img2', None)

    print(f"\nScene Statistics:")
    print(f"  - 3D Points: {len(points_3d)}")
    print(f"  - Cameras: {len(camera_poses)}")
    print(f"  - Inliers: {num_inliers}")
    print(f"  - Baseline: {baseline:.4f} units")
    if img1 is not None:
        print(f"  - Image 1 shape: {img1.shape}")
    if img2 is not None:
        print(f"  - Image 2 shape: {img2.shape}")

    # Create viser server with automatic port selection
    # Start from the requested port and increment if port is already in use
    current_port = port
    max_attempts = 100  # Try up to 100 ports before giving up
    server = None

    for attempt in range(max_attempts):
        try:
            server = viser.ViserServer(port=current_port)
            break  # Successfully created server
        except OSError as e:
            # Port is already in use, try next port
            if attempt == 0:
                print(f"\n⚠️  Port {current_port} is already in use, trying next available port...")
            current_port += 1
            if attempt >= max_attempts - 1:
                raise RuntimeError(f"Could not find an available port after {max_attempts} attempts (starting from {port})")

    if server is None:
        raise RuntimeError("Failed to create viser server")

    if current_port != port:
        print(f"\n🌐 Viser server running at http://localhost:{current_port} (port {port} was in use)")
    else:
        print(f"\n🌐 Viser server running at http://localhost:{current_port}")
    print("Press Ctrl+C to exit")

    # Set initial viewer position to look straight at the back of Camera 0.
    # Camera 0 after R_fix: at origin, forward = col2 of its c2w rotation,
    # up = negated col1 of its c2w rotation.
    _cam0_forward = camera_poses[0][:3, 2]
    _cam0_up = -camera_poses[0][:3, 1]

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = -_cam0_forward * 3.0 + _cam0_up * 0.5
        client.camera.look_at = np.zeros(3)
        client.camera.up_direction = _cam0_up

    # Add 3D points (in Camera 0's coordinate system = world frame)
    # Normalize colors to [0, 1] range if needed
    if point_colors.max() > 1.0:
        point_colors = point_colors / 255.0

    pc_handle = server.scene.add_point_cloud(
        name="/point_cloud",
        points=points_3d,
        colors=point_colors,
        point_size=0.01,
    )

    # Add camera frustums and images
    # T matrices are camera-to-world (c2w): T[:3,:3] = R_c2w, T[:3,3] = camera position
    # viser frustums use OpenCV convention (+Z forward, +X right, +Y down)
    # so we pass R_c2w directly — no coordinate conversion needed.

    camera_names = ["Camera 1 (Reference)", "Camera 2"]
    camera_colors = [(0, 255, 0), (255, 0, 0)]  # Green for cam1, red for cam2
    camera_images = [img1, img2]

    # Store handles for visibility toggling
    frustum_handles = []
    label_handles = []
    image_handles = []

    for i, (T, name, color, img) in enumerate(zip(camera_poses, camera_names, camera_colors, camera_images)):
        R_c2w = T[:3, :3]
        cam_position = T[:3, 3]

        # Compute vertical FOV from intrinsics (viser fov is vertical in radians)
        fy = K[1, 1]
        if img is not None:
            img_height = img.shape[0]
            fov = 2 * np.arctan(img_height / 2 / fy)  # Vertical FOV
            aspect = img.shape[1] / img.shape[0]
        else:
            cy = K[1, 2]
            fov = 2 * np.arctan(cy / fy)
            aspect = 1.0

        frustum_handles.append(server.scene.add_camera_frustum(
            name=f"/camera_{i}",
            fov=fov,
            aspect=aspect,
            scale=0.3,
            wxyz=viser.transforms.SO3.from_matrix(R_c2w).wxyz,
            position=cam_position,
            color=color,
        ))

        # Add camera label
        label_handles.append(server.scene.add_label(
            name=f"/camera_{i}_label",
            text=name,
            position=cam_position + np.array([0, 0, 0.15]),
        ))

        # Add image plane in front of camera
        # viser's add_image internally applies a π rotation around X,
        # making the image face -Z in the local frame and mapping row 0 to top.
        # With R_c2w as orientation: -Z local = -forward = faces back toward camera.
        if img is not None:
            # Ensure image is uint8
            if img.dtype != np.uint8:
                img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            else:
                img_uint8 = img

            # Handle grayscale images
            if img_uint8.ndim == 2:
                img_uint8 = np.stack([img_uint8] * 3, axis=-1)

            # Compute image plane size from vertical FOV
            image_distance = 0.5
            image_height = 2 * image_distance * np.tan(fov / 2)
            image_width = image_height * aspect

            # Camera's forward direction in world (column 2 of c2w rotation)
            forward = R_c2w[:, 2]
            image_center = cam_position + forward * image_distance

            image_handles.append(server.scene.add_image(
                name=f"/camera_{i}_image",
                image=img_uint8,
                render_width=image_width,
                render_height=image_height,
                wxyz=viser.transforms.SO3.from_matrix(R_c2w).wxyz,
                position=image_center,
            ))
        else:
            image_handles.append(None)

    # Add coordinate frame at world origin (Camera 0's coordinate system)
    server.scene.add_frame(
        name="/world",
        wxyz=(1.0, 0.0, 0.0, 0.0),  # Identity — world frame = Camera 0 frame
        position=np.array([0.0, 0.0, 0.0]),
        axes_length=0.3,
        axes_radius=0.01,
    )

    # Add baseline visualization (line between cameras)
    baseline_handle = server.scene.add_spline_catmull_rom(
        name="/baseline",
        positions=np.array([camera_poses[0][:3, 3], camera_poses[1][:3, 3]]),
        color=(255, 255, 0),  # Yellow
        line_width=3.0,
    )

    # Add GUI controls
    with server.gui.add_folder("Scene Info"):
        server.gui.add_text("Points", initial_value=str(len(points_3d)), disabled=True)
        server.gui.add_text("Inliers", initial_value=str(num_inliers), disabled=True)
        server.gui.add_text("Baseline", initial_value=f"{baseline:.4f}", disabled=True)

    with server.gui.add_folder("Camera Info"):
        server.gui.add_text("fx", initial_value=f"{K[0,0]:.2f}", disabled=True)
        server.gui.add_text("fy", initial_value=f"{K[1,1]:.2f}", disabled=True)
        server.gui.add_text("cx", initial_value=f"{K[0,2]:.2f}", disabled=True)
        server.gui.add_text("cy", initial_value=f"{K[1,2]:.2f}", disabled=True)

    with server.gui.add_folder("Pose (Camera 2)"):
        R_text = "\n".join([f"[{R[i,0]:7.4f} {R[i,1]:7.4f} {R[i,2]:7.4f}]" for i in range(3)])
        server.gui.add_text("R", initial_value=R_text, disabled=True)
        t_text = f"[{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}]"
        server.gui.add_text("t", initial_value=t_text, disabled=True)

    # Add point cloud controls
    with server.gui.add_folder("Point Cloud Settings"):
        point_size_slider = server.gui.add_slider(
            "Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
        )

        # Depth filter sliders — depth is Z in Camera 0's coordinate system
        all_depths = points_3d[:, 2]
        depth_floor = float(np.floor(all_depths.min()))
        depth_ceil = float(np.ceil(all_depths.max()))

        min_depth_slider = server.gui.add_slider(
            "Min Depth", min=depth_floor, max=depth_ceil,
            step=0.01, initial_value=depth_floor
        )
        max_depth_slider = server.gui.add_slider(
            "Max Depth", min=depth_floor, max=depth_ceil,
            step=0.01, initial_value=depth_ceil
        )
        depth_count_text = server.gui.add_text(
            "Visible Points", initial_value=str(len(points_3d)), disabled=True
        )

        def _update_depth_filter(_=None):
            mask = (all_depths >= min_depth_slider.value) & (all_depths <= max_depth_slider.value)
            filtered_pts = points_3d[mask]
            filtered_colors = point_colors[mask]
            depth_count_text.value = str(len(filtered_pts))
            if len(filtered_pts) > 0:
                pc_handle.points = filtered_pts
                pc_handle.colors = filtered_colors
                pc_handle.visible = True
            else:
                pc_handle.visible = False

        @point_size_slider.on_update
        def _(_):
            pc_handle.point_size = point_size_slider.value

        @min_depth_slider.on_update
        def _(_):
            _update_depth_filter()

        @max_depth_slider.on_update
        def _(_):
            _update_depth_filter()

    # Add visibility controls — toggle handle.visible instead of removing/re-adding
    with server.gui.add_folder("Visibility"):
        point_cloud_visible = server.gui.add_checkbox("Point Cloud", initial_value=True)
        camera_visible = server.gui.add_checkbox("Cameras", initial_value=True)
        images_visible = server.gui.add_checkbox("Camera Images", initial_value=True)
        baseline_visible = server.gui.add_checkbox("Baseline", initial_value=True)

        @point_cloud_visible.on_update
        def _(_):
            pc_handle.visible = point_cloud_visible.value

        @camera_visible.on_update
        def _(_):
            for h in frustum_handles:
                h.visible = camera_visible.value
            for h in label_handles:
                h.visible = camera_visible.value

        @images_visible.on_update
        def _(_):
            for h in image_handles:
                if h is not None:
                    h.visible = images_visible.value

        @baseline_visible.on_update
        def _(_):
            baseline_handle.visible = baseline_visible.value

    # Snap-to-camera buttons
    with server.gui.add_folder("Snap to Camera"):
        snap_cam1_btn = server.gui.add_button("Snap to Camera 1")
        snap_cam2_btn = server.gui.add_button("Snap to Camera 2")

        def _snap_to_camera(event: viser.GuiEvent, cam_index: int):
            """Move the viewer to the exact COP of the specified camera."""
            client = event.client
            if client is None:
                return

            T = camera_poses[cam_index]
            R_c2w = T[:3, :3]
            cam_position = T[:3, 3]

            # OpenCV convention: +Z = forward, +Y = down
            forward = R_c2w[:, 2]
            up = -R_c2w[:, 1]  # Negate Y since OpenCV Y points down

            # Set the viewer camera to match this scene camera's position & orientation.
            # We intentionally don't override client.camera.fov — the physical camera's
            # vertical FOV (~44-57°) is much narrower than the comfortable viewer default
            # (~75°), which would make the view feel extremely zoomed in.
            look_at = cam_position + forward * 1.0
            client.camera.position = cam_position
            client.camera.look_at = look_at
            client.camera.up_direction = up

        @snap_cam1_btn.on_click
        def _(event: viser.GuiEvent):
            _snap_to_camera(event, 0)

        @snap_cam2_btn.on_click
        def _(event: viser.GuiEvent):
            _snap_to_camera(event, 1)

    # Keep the server running
    try:
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down viser server...")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_viser.py <path_to_scene_data.npz>")
        print("\nExample:")
        print("  python visualize_viser.py outputs/pose_estimation_harris=20_anms=250_metric=NCC_7/scene_data.npz")
        sys.exit(1)

    npz_path = sys.argv[1]

    if not os.path.exists(npz_path):
        print(f"Error: File not found: {npz_path}")
        sys.exit(1)

    port = 8081  # Default starting port (will auto-increment if in use)
    if len(sys.argv) > 2:
        port = int(sys.argv[2])

    visualize_scene(npz_path, port)


if __name__ == "__main__":
    main()