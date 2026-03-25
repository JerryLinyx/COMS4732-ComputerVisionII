import torch
from torch.utils.data import Dataset
import numpy as np


def load_data(
    data_path: str, 
):
    """Load and preprocess dataset from an npz file.

    Args:
        data_path: str representing the path to the .npz data file

    Returns:
        images_train: torch.Tensor of shape (num_train, H, W, 3) normalized to [0, 1]
        c2ws_train: torch.Tensor of shape (num_train, 4, 4) representing camera-to-world matrices
        images_val: torch.Tensor of shape (num_val, H, W, 3) normalized to [0, 1]
        c2ws_val: torch.Tensor of shape (num_val, 4, 4) representing camera-to-world matrices
        c2ws_test: torch.Tensor of shape (num_test, 4, 4) representing camera-to-world matrices
        K: torch.Tensor of shape (3, 3) representing the camera intrinsics matrix
    """
    data = np.load(data_path)


    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0

    # Cameras for the training images
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]

    # Validation images:
    images_val = data["images_val"] / 255.0

    # Cameras for the validation images: [10, 4, 4]
    # (camera-to-world transformation matrix): [10, 200, 200, 3]
    c2ws_val = data["c2ws_val"]

    # Test cameras for novel-view video rendering:
    # (camera-to-world transformation matrix): [60, 4, 4]
    c2ws_test = data["c2ws_test"]

    # Camera focal length
    focal = data["focal"]  # float

    h, w = images_train.shape[1], images_train.shape[2]
    o_x = w / 2
    o_y = h / 2
    K = torch.as_tensor([[focal.item(), 0, o_x], [0, focal.item(), o_y], [0, 0, 1]])

    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K


def pixel_to_camera(
    K: torch.Tensor,
    uvs: torch.Tensor,
    s: float = 1,
    device: str = "cuda",
):
    """Pixel to camera transformation on a batch of pixels.

    Args:
        K: torch.Tensor of shape (3, 3) representing the camera intrinsics matrix
        uvs: torch.Tensor of shape (num_pixels, 3) representing the homogeneous pixel coordinates
        s: float representing the scaling factor
        device: str representing the device to run on

    Returns:
        torch.Tensor of shape (num_pixels, 3) representing the camera-space coordinates
    """

    K_inv = torch.linalg.inv(K).to(device)
    unnormalized_x_cs = (K_inv @ uvs.T).T  # (num_pixels, 3)

    return s * unnormalized_x_cs


def pixels_to_rays(
    K: torch.Tensor,
    c2w: torch.Tensor,
    uvs: torch.Tensor,
    verbose: bool = False,
    device: str = "cuda",
):
    """Convert pixels to rays.

    Args:
        K: torch.Tensor of shape (3, 3) representing the camera intrinsics matrix
        c2w: torch.Tensor of shape (4, 4) representing the camera-to-world matrix
        uvs: torch.Tensor of shape (num_pixels, 2) representing the pixel coordinates
        verbose: bool representing whether to print verbose output
        device: str representing the device to run on

    Returns:
        r_os: torch.Tensor of shape (num_pixels, 3) representing the ray origins
        r_ds: torch.Tensor of shape (num_pixels, 3) representing the ray directions
    """
    K = K.to(device)
    c2w = c2w.to(torch.float64).to(device)
    uvs = uvs.to(device)

    if verbose:
        print(f"K shape: {K.shape} should be of the form (3,3)")
        print(f"c2ws shape: {c2w.shape} should be of the form (4, 4)")
        print(f"uvs shape: {uvs.shape} should be of the form (num_pixels, 2)")

    num_pixels = uvs.shape[0]
    R = c2w[:3, :3]  # (3, 3)

    r_os = c2w[:3, 3].unsqueeze(0).expand(num_pixels, -1)  # (num_pixels, 3)

    homog_uvs = torch.hstack(
        (uvs, torch.ones(num_pixels, 1, device=device))
    )  # (num_pixels, 3)

    # ray directions: R @ K_inv @ [u, v, 1] for each pixel, then normalize
    K_inv = torch.linalg.inv(K).to(device)
    M = R @ K_inv.to(torch.float64)  # (3, 3) — combines rotation and unprojection
    dirs = (M @ homog_uvs.to(torch.float64).T).T  # (num_pixels, 3)
    r_ds = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True)

    if verbose:
        print(
            f"homog uvs shape: {homog_uvs.shape} should be of the form (num_pixels, 3)"
        )
        print(f"r_os shape: {r_os.shape}")
        print(f"r_ds shape: {r_ds.shape}")

    return r_os, r_ds  # each is (num_pixels, 3)


def image_to_rays(
    image: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    verbose: bool = False,
    device: str = "cuda",
):
    """Convert an image to rays.

    Args:
        image: torch.Tensor of shape (H, W, 3) representing the image
        c2w: torch.Tensor of shape (4, 4) representing the camera-to-world matrix
        K: torch.Tensor of shape (3, 3) representing the camera intrinsics matrix
        verbose: bool representing whether to print verbose output
        device: str representing the device to run on

    Returns:
        torch.Tensor of shape (H, W, 6) where [:, :, :3] are ray origins
        and [:, :, 3:] are ray directions
    """
    # TODO implement yourself
    pass


def images_to_rays(
    images: torch.Tensor,
    c2ws: torch.Tensor,
    K: torch.Tensor,
    verbose: bool = False,
    device: str = "cuda",
):
    """Convert a batch of images to rays.

    Args:
        images: torch.Tensor of shape (num_images, H, W, 3) representing the images
        c2ws: torch.Tensor of shape (num_images, 4, 4) representing the camera-to-world matrices
        K: torch.Tensor of shape (3, 3) representing the camera intrinsics matrix
        verbose: bool representing whether to print verbose output
        device: str representing the device to run on

    Returns:
        torch.Tensor of shape (num_images, H, W, 6) where [:, :, :, :3] are ray origins
        and [:, :, :, 3:] are ray directions
    """
    # TODO implement yourself
    pass


class RaysData(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        K: torch.Tensor,
        c2ws: torch.Tensor,
        split: str = "train",
        device: str = "cuda",
    ):
        """Initialize the RaysData dataset by precomputing rays for all pixels across all images.

        Args:
            images: torch.Tensor of shape (num_images, H, W, 3) representing the images
            K: torch.Tensor of shape (3, 3) representing the camera intrinsics matrix
            c2ws: torch.Tensor of shape (num_images, 4, 4) representing the camera-to-world matrices
            split: str representing the data split ("train", "val", or "test")
            device: str representing the device to run on

        Returns:
            None, but should define the following attributes:
            - self.uvs: torch.Tensor of shape (num_images * H * W, 2) — integer (x, y) pixel
              coordinates for every pixel across all images, used to verify ray-pixel correspondence
              Used in `visualize_viser.py`
            - self.rays_o: torch.Tensor of shape (num_images * H * W, 3) — ray origins for every
              pixel across all images
            - self.rays_d: torch.Tensor of shape (num_images * H * W, 3) — ray directions for every
              pixel across all images
            - self.gt_rgbs: torch.Tensor of shape (num_images * H * W, 3) — all pixel
              colors flattened (images.reshape(-1, 3)), used by __len__() and sample_rays()
              to return ground truth colors for sampled rays
        """
        self.images = images
        self.K = K
        self.c2ws = c2ws
        self.h, self.w = self.images.shape[1:3]
        self.num_images = self.images.shape[0]

        # self.uvs: integer (x, y) pixel coordinates for every pixel across all images.
        # Shape: (num_images * H * W, 2)
        # For a single image, the uvs are all (x, y) pairs: (0,0), (1,0), ..., (W-1,0), (0,1), ...
        # These are repeated for each image so that uvs[i] tells you which pixel
        # in which image the i-th ray corresponds to.
        # Convention: uvs[:, 0] = x (column), uvs[:, 1] = y (row)
        # This is used in visualize_viser.py to verify that rays match the correct pixels:
        #   assert images[0, uvs[:, 1], uvs[:, 0]] == dataset.pixels[:]
        # Hint: torch.meshgrid with torch.arange(W) and torch.arange(H)

        # TODO implement yourself
        pass

    def __len__(self):
        """Return the total number of rays in the dataset.

        Returns:
            int representing num_images * H * W
        """
        # TODO implement yourself
        pass

    def sample_rays(self, num_rays: int):
        """Sample random rays from the dataset.

        Args:
            num_rays: int representing the number of rays to sample

        Returns:
            r_os: torch.Tensor of shape (num_rays, 3) representing the ray origins
            r_ds: torch.Tensor of shape (num_rays, 3) representing the ray directions
            gt_rgbs: torch.Tensor of shape (num_rays, 3) representing the ground truth colors

        Hints:
            You need to randomly sample the rays and pixels using num_rays
        """
        # TODO implement yourself
        return None, None, None
