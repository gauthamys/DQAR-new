"""
Evaluation Metrics Module

Implements FID (Fréchet Inception Distance) and CLIP score
computation for evaluating generated image quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import numpy as np
from pathlib import Path


class FIDCalculator:
    """
    Calculator for Fréchet Inception Distance (FID).

    FID measures the distance between the distribution of generated
    images and real images in InceptionV3 feature space.
    """

    def __init__(
        self,
        device: str = "cuda",
        feature_dim: int = 2048,
    ):
        """
        Args:
            device: Device to run computations on
            feature_dim: Dimension of Inception features
        """
        self.device = device
        self.feature_dim = feature_dim
        self._inception = None

    def _load_inception(self):
        """Lazy load InceptionV3 model."""
        if self._inception is None:
            try:
                from torchvision.models import inception_v3, Inception_V3_Weights

                self._inception = inception_v3(
                    weights=Inception_V3_Weights.IMAGENET1K_V1,
                    transform_input=False,
                )
                self._inception.fc = nn.Identity()  # Remove final classification layer
                self._inception.eval()
                self._inception.to(self.device)
            except ImportError:
                raise ImportError("Please install torchvision: pip install torchvision")

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Extract InceptionV3 features from images.

        Args:
            images: Tensor of images (N, C, H, W) in range [0, 1]
            batch_size: Batch size for feature extraction

        Returns:
            Feature tensor of shape (N, feature_dim)
        """
        self._load_inception()

        # Resize to Inception input size
        if images.shape[-1] != 299 or images.shape[-2] != 299:
            images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)
            feat = self._inception(batch)
            features.append(feat.cpu())

        return torch.cat(features, dim=0)

    def compute_statistics(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and covariance of features.

        Args:
            features: Feature tensor (N, D)

        Returns:
            Tuple of (mean, covariance)
        """
        features = features.numpy() if isinstance(features, torch.Tensor) else features

        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        return mu, sigma

    def compute_fid_from_stats(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """
        Compute FID between two sets of statistics.

        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Small constant for numerical stability

        Returns:
            FID score
        """
        from scipy import linalg

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        # Compute squared difference of means
        diff = mu1 - mu2
        diff_sq = diff.dot(diff)

        # Compute sqrt(sigma1 * sigma2)
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Handle numerical issues
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Compute FID
        tr_covmean = np.trace(covmean)
        fid = diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return float(fid)

    def compute(
        self,
        generated_images: torch.Tensor,
        real_images: Optional[torch.Tensor] = None,
        real_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 32,
    ) -> float:
        """
        Compute FID score.

        Args:
            generated_images: Generated images (N, C, H, W)
            real_images: Real reference images (M, C, H, W)
            real_stats: Pre-computed (mu, sigma) for real images
            batch_size: Batch size for feature extraction

        Returns:
            FID score
        """
        # Extract features from generated images
        gen_features = self.extract_features(generated_images, batch_size)
        mu_gen, sigma_gen = self.compute_statistics(gen_features)

        # Get real image statistics
        if real_stats is not None:
            mu_real, sigma_real = real_stats
        elif real_images is not None:
            real_features = self.extract_features(real_images, batch_size)
            mu_real, sigma_real = self.compute_statistics(real_features)
        else:
            raise ValueError("Either real_images or real_stats must be provided")

        return self.compute_fid_from_stats(mu_gen, sigma_gen, mu_real, sigma_real)


class CLIPScoreCalculator:
    """
    Calculator for CLIP score (image-text alignment).

    Measures how well generated images match their text prompts
    using CLIP embeddings.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
    ):
        """
        Args:
            model_name: CLIP model variant
            device: Device to run computations on
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._preprocess = None

    def _load_clip(self):
        """Lazy load CLIP model."""
        if self._model is None:
            try:
                import open_clip

                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self._model.eval()
                self._model.to(self.device)
            except ImportError:
                raise ImportError("Please install open_clip: pip install open-clip-torch")

    @torch.no_grad()
    def encode_images(
        self,
        images: torch.Tensor,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode images to CLIP embeddings.

        Args:
            images: Tensor of images (N, C, H, W) in range [0, 1]
            batch_size: Batch size for encoding

        Returns:
            Image embeddings (N, D)
        """
        self._load_clip()

        # Resize to CLIP input size (224x224)
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)
            emb = self._model.encode_image(batch)
            embeddings.append(emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode texts to CLIP embeddings.

        Args:
            texts: List of text prompts
            batch_size: Batch size for encoding

        Returns:
            Text embeddings (N, D)
        """
        self._load_clip()

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self._tokenizer(batch).to(self.device)
            emb = self._model.encode_text(tokens)
            embeddings.append(emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def compute(
        self,
        images: torch.Tensor,
        texts: List[str],
        batch_size: int = 32,
    ) -> float:
        """
        Compute CLIP score between images and texts.

        Args:
            images: Generated images (N, C, H, W)
            texts: Corresponding text prompts

        Returns:
            Mean CLIP score
        """
        assert len(images) == len(texts), "Number of images must match number of texts"

        image_embeddings = self.encode_images(images, batch_size)
        text_embeddings = self.encode_text(texts, batch_size)

        # Compute cosine similarity
        scores = (image_embeddings * text_embeddings).sum(dim=-1)

        # Scale to [0, 100] range (common convention)
        scores = scores * 100

        return float(scores.mean())

    def compute_per_sample(
        self,
        images: torch.Tensor,
        texts: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Compute per-sample CLIP scores.

        Args:
            images: Generated images (N, C, H, W)
            texts: Corresponding text prompts

        Returns:
            Per-sample CLIP scores (N,)
        """
        image_embeddings = self.encode_images(images, batch_size)
        text_embeddings = self.encode_text(texts, batch_size)

        scores = (image_embeddings * text_embeddings).sum(dim=-1) * 100
        return scores


def compute_fid(
    generated_images: torch.Tensor,
    real_images: Optional[torch.Tensor] = None,
    real_stats_path: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """
    Convenience function to compute FID.

    Args:
        generated_images: Generated images
        real_images: Real reference images
        real_stats_path: Path to pre-computed real statistics (.npz)
        device: Device to use

    Returns:
        FID score
    """
    calculator = FIDCalculator(device=device)

    real_stats = None
    if real_stats_path is not None:
        data = np.load(real_stats_path)
        real_stats = (data["mu"], data["sigma"])

    return calculator.compute(
        generated_images,
        real_images=real_images,
        real_stats=real_stats,
    )


def compute_clip_score(
    images: torch.Tensor,
    texts: List[str],
    device: str = "cuda",
) -> float:
    """
    Convenience function to compute CLIP score.

    Args:
        images: Generated images
        texts: Text prompts
        device: Device to use

    Returns:
        Mean CLIP score
    """
    calculator = CLIPScoreCalculator(device=device)
    return calculator.compute(images, texts)
