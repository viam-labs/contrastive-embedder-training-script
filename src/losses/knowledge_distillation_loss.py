import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation wrapper that enhances any base loss function.
    This class wraps around your existing loss functions (ContrastiveLoss, etc.)
    """
    
    def __init__(self, base_loss, temperature=4.0, alpha=0.7, feature_weight=0.1, 
                 normalize_embeddings=True):
        super().__init__()
        self.base_loss = base_loss
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.normalize_embeddings = normalize_embeddings
        
        # For soft target distillation
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # Teacher model will be set by the knowledge distillation manager
        self.teacher_model = None
        self._is_teacher_set = False
        
    def set_teacher_model(self, teacher_model, device):
        """Set the teacher model for distillation"""
        self.teacher_model = teacher_model
        self.teacher_model.to(device)
        self.teacher_model.eval()
        
        # Freeze teacher parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self._is_teacher_set = True
        logger.info("Teacher model set in KnowledgeDistillationLoss")
    
    def forward(self, student_emb1, student_emb2, labels, images1=None, images2=None):
        """
        Forward pass with optional knowledge distillation
        
        Args:
            student_emb1, student_emb2: Student model embeddings
            labels: Ground truth labels
            images1, images2: Original images (needed for teacher inference)
        """
        # Always compute base loss (contrastive, triplet, etc.)
        if self.normalize_embeddings:
            student_emb1_norm = F.normalize(student_emb1, p=2, dim=1)
            student_emb2_norm = F.normalize(student_emb2, p=2, dim=1)
        else:
            student_emb1_norm = student_emb1
            student_emb2_norm = student_emb2
            
        base_loss = self.base_loss(student_emb1_norm, student_emb2_norm, labels)
        
        # If no teacher or no images, return base loss only
        if not self._is_teacher_set or images1 is None or images2 is None:
            return base_loss
        
        # Compute knowledge distillation components
        with torch.no_grad():
            # Get teacher embeddings
            teacher_output1 = self.teacher_model(images1)
            teacher_output2 = self.teacher_model(images2)
            
            # Handle different output formats (logits, features) or just features
            if isinstance(teacher_output1, tuple):
                teacher_emb1 = teacher_output1[1]  # (logits, features)
                teacher_emb2 = teacher_output2[1]
            else:
                teacher_emb1 = teacher_output1
                teacher_emb2 = teacher_output2
            
            if self.normalize_embeddings:
                teacher_emb1 = F.normalize(teacher_emb1, p=2, dim=1)
                teacher_emb2 = F.normalize(teacher_emb2, p=2, dim=1)
        
        # Feature matching loss
        feature_loss = (
            F.mse_loss(student_emb1_norm, teacher_emb1) + 
            F.mse_loss(student_emb2_norm, teacher_emb2)
        ) / 2
        
        # Soft target distillation using similarity scores
        student_similarity = F.cosine_similarity(student_emb1_norm, student_emb2_norm, dim=1)
        teacher_similarity = F.cosine_similarity(teacher_emb1, teacher_emb2, dim=1)
        
        # Convert similarities to logits for distillation
        student_logits = torch.stack([-student_similarity, student_similarity], dim=1) / self.temperature
        teacher_logits = torch.stack([-teacher_similarity, teacher_similarity], dim=1) / self.temperature
        
        distillation_loss = self.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1)
        ) * (self.temperature ** 2)
        
        # Combine all losses
        total_loss = (
            (1 - self.alpha) * base_loss +
            self.alpha * distillation_loss +
            self.feature_weight * feature_loss
        )
        
        return total_loss
    
    def _compute_distance(self, output1, output2):
        """Maintain compatibility with existing evaluation code"""
        return self.base_loss._compute_distance(output1, output2)