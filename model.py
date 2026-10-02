import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DesignEmbedder(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights)
        self.features = net.features
        self.pool = net.avgpool
        self.projection = nn.Linear(net.classifier[1].in_features, embedding_dim)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return nn.functional.normalize(self.projection(x), dim=1)


def supervised_contrastive_loss(z, labels, temperature=0.1):
    """Stable SupCon loss; samples without a positive partner are excluded."""
    n = z.shape[0]
    if n < 2:
        raise ValueError("Contrastive batch must contain at least two samples")
    logits = (z @ z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    labels = labels.reshape(-1, 1)
    same = labels.eq(labels.T).to(z.dtype)
    eye = torch.eye(n, device=z.device, dtype=z.dtype)
    positives = same - eye
    exp_logits = torch.exp(logits) * (1 - eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = positives.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        raise ValueError("Each training batch needs at least one repeated design ID")
    return -((positives * log_prob).sum(dim=1)[valid] / pos_count[valid]).mean()
