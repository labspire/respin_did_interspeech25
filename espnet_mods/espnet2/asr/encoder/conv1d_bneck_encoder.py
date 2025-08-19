import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Utility function for masked mean pooling
def mean_pooling(x, mask):
    mask = mask.unsqueeze(-1)  # Expand mask to match x dimensions (batch_size, seq_len, 1)
    x = x * mask  # Zero out masked positions
    sum_x = x.sum(dim=1)  # Sum over the time dimension
    valid_lengths = mask.sum(dim=1).clamp(min=1e-9)  # Avoid division by zero
    return sum_x / valid_lengths

class BottleneckEncoder(nn.Module):
    def __init__(
        self, input_dim=256, cnn_dim1=128, cnn_dim2=64, bottleneck_dim=32, num_heads=4, num_dialects=4, dropout_rate=0.1, get_embeddings=False
    ):
        super().__init__()
        """
        Bottleneck Encoder for Dialect Identification with Conv1d in the bottleneck transformer.
        """
        self.get_embeddings = get_embeddings

        # CNN Encoder
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim1, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim1),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # Dropout after the first CNN layer
            nn.Conv1d(cnn_dim1, cnn_dim2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim2),
            nn.GELU(),
            nn.Dropout(dropout_rate)  # Dropout after the second CNN layer
        )

        # Bottleneck Transformer with Conv1d
        self.conv1 = nn.Conv1d(cnn_dim2, bottleneck_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(bottleneck_dim)
        self.activation1 = nn.GELU()
        self.attention = nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(bottleneck_dim)
        self.conv2 = nn.Conv1d(bottleneck_dim, cnn_dim2, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(cnn_dim2)
        self.activation2 = nn.GELU()
        self.residual_dropout = nn.Dropout(dropout_rate)

        if not self.get_embeddings:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),  # Dropout before the final classification layer
                nn.Linear(cnn_dim2, num_dialects)
            )

    def forward(self, encoder_out, encoder_out_lens: Optional[torch.Tensor] = None):
        """
        Forward pass for Bottleneck Encoder.

        Args:
            encoder_out: Input tensor of shape (batch_size, seq_len, input_dim)
            encoder_out_lens: Lengths of the encoder outputs for masking
        """
        # Step 1: CNN Encoding
        x = encoder_out.permute(0, 2, 1)  # (batch_size, features, seq_len)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, features)

        # Step 2: Bottleneck Transformer
        residual = x.permute(0, 2, 1)  # (batch_size, features, seq_len)
        x = self.conv1(residual)
        x = self.bn1(x)
        x = self.activation1(x)

        x = x.permute(2, 0, 1)  # (seq_len, batch_size, bottleneck_dim)
        key_padding_mask = None
        if encoder_out_lens is not None:
            attention_mask = torch.arange(x.size(0)).unsqueeze(0).expand(x.size(1), x.size(0)).to(x.device)
            attention_mask = (attention_mask < encoder_out_lens.unsqueeze(1)).int()
            key_padding_mask = attention_mask == 0  # Convert to key_padding_mask

        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)  # Attention
        x = self.norm1(x)
        x = x.permute(1, 2, 0)  # (batch_size, bottleneck_dim, seq_len)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = x + self.residual_dropout(residual)  # Residual connection
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, features)

        if self.get_embeddings:
            return x

        # Step 3: Mask creation
        batch_size, seq_len, _ = x.shape
        mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).to(encoder_out.device)
        mask = (mask < encoder_out_lens.unsqueeze(1)).float()

        # Step 4: Mean pooling
        pooled_out = mean_pooling(x, mask)

        # Step 5: Classification
        logits = self.classifier(pooled_out)
        return logits
