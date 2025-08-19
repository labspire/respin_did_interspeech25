import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyModule(nn.Module):
    """CrossEntropyLossModule for calculating CE loss.

    Args:
        encoder_output_size: output dimension
        odim: output dimension for cross-entropy
    """

    def __init__(
        self,
        encoder_output_size: int,
        odim: int,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.ce_lo = nn.Linear(eprojs, odim)
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def loss_fn(self, th_pred, th_target) -> torch.Tensor:
        loss = self.ce_loss(th_pred, th_target)
        return loss

    def forward(self, hs_pad, ys_pad):
        
        ys_hat = self.ce_lo(hs_pad)

        # Compute loss using mean-pooled logits
        loss = self.ce_loss(ys_hat, ys_pad).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )
        return loss

    def softmax(self, hs_pad):
        """Softmax of mean-pooled frame activations for CE loss.

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 2d tensor (B, odim)
        """
        ys_hat = self.ce_lo(hs_pad)
        return F.softmax(ys_hat, dim=1)

    def log_softmax(self, hs_pad):
        """Log softmax of mean-pooled frame activations for CE loss.

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 2d tensor (B, odim)
        """
        ys_hat = self.ce_lo(hs_pad)
        return F.log_softmax(ys_hat, dim=1)

    def argmax(self, hs_pad):
        """Argmax of mean-pooled frame activations for CE loss.

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 1d tensor (B,)
        """
        ys_hat = self.ce_lo(hs_pad)
        return torch.argmax(ys_hat, dim=1)
