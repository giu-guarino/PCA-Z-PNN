import torch

def pca(ms_lr):
    """
    Perform Principal Component Analysis (PCA) on a PyTorch tensor image.

    Args:
    ms_lr (torch.Tensor): Input image tensor of shape (1, B, H, W).

    Returns:
    pca_image (torch.Tensor): PCA-transformed image tensor with the same shape.
    pca_matrix (torch.Tensor): PCA transformation matrix.
    mean (torch.Tensor): Tensor of mean values.
    """
    # Reshape the input tensor to (B, H * W) and mean-center the data
    _, B, H, W = ms_lr.shape
    flattened = torch.reshape(ms_lr, (B, H*W))
    mean = torch.mean(flattened, dim=1).unsqueeze(1)
    centered = flattened - mean

    # Compute the covariance matrix
    cov_matrix = torch.matmul(centered, centered.t()) / (H * W - 1)

    # Perform PCA using SVD
    U, S, _ = torch.svd(cov_matrix)

    # PCA-transformed image
    pca_image = torch.matmul(-U.t(), centered).view(1, B, H, W)

    return pca_image, U, mean


def inverse_pca(pca_image, pca_matrix, mean):
    """
    Perform the inverse of Principal Component Analysis (PCA) on a PCA-transformed image.

    Args:
    pca_image (torch.Tensor): PCA-transformed image tensor with the same shape as the input image.
    pca_matrix (torch.Tensor): PCA transformation matrix obtained from the 'pca' function.
    mean (torch.Tensor): Tensor of mean values.

    Returns:
    original_image (torch.Tensor): Inverse PCA-reconstructed image tensor.
    """
    _, B, H, W = pca_image.shape
    flattened_pca = torch.reshape(pca_image, (B, H*W))

    flattened_image = torch.matmul(-pca_matrix, flattened_pca) + mean

    # Reconstruct the original image
    original_image = flattened_image.view(1, B, H, W)

    return original_image