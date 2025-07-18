import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for siamese networks.

    Args:
        margin (float): Margin for dissimilar pairs. Default: 1.0
        distance_metric (str): Distance metric to use ('euclidean' or 'cosine'). Default: 'euclidean'
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """

    def __init__(self, margin=2.0, distance_metric="euclidean", reduction="mean"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric.lower()
        self.reduction = reduction

        if self.distance_metric not in ["euclidean", "cosine"]:
            raise ValueError("distance_metric must be either 'euclidean' or 'cosine'")

    def _compute_distance(self, output1, output2):
        """
        Compute distance between two feature embeddings.

        Args:
            output1 (torch.Tensor): Feature embeddings from first input
            output2 (torch.Tensor): Feature embeddings from second input

        Returns:
            torch.Tensor: Computed distance
        """
        if self.distance_metric == "euclidean":
            return F.pairwise_distance(output1, output2, keepdim=True)
        elif self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            cosine_sim = F.cosine_similarity(output1, output2, dim=1)
            return 1 - cosine_sim

    def forward(self, output1, output2, target):
        """
        Forward pass of contrastive loss.

        Args:
            output1 (torch.Tensor): Feature embeddings from first input
            output2 (torch.Tensor): Feature embeddings from second input
            target (torch.Tensor): Binary labels (1 for similar pairs, 0 for dissimilar pairs)

        Returns:
            torch.Tensor: Computed contrastive loss
        """
        # Compute distance based on selected metric
        distance = self._compute_distance(output1, output2)

        # Contrastive loss calculation
        similar_loss = target.float() * torch.pow(distance, 2)
        dissimilar_loss = (1 - target.float()) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )

        loss_per_sample = similar_loss + dissimilar_loss

        if self.reduction == "none":
            return loss_per_sample
        elif self.reduction == "sum":
            return torch.sum(loss_per_sample)
        else:  # reduction == 'mean'
            return torch.mean(loss_per_sample)
