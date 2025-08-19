import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility function for mean pooling
def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

class AttentionEncoder(nn.Module):
    def __init__(self, feature_dim, num_heads=8, hidden_dim=512, num_layers=1, dropout=0.1):
        """
        Attention Encoder with explicit feed-forward network layers.

        Args:
            feature_dim (int): Input feature dimension.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of the feed-forward network.
            num_layers (int): Number of attention encoder layers.
            dropout (float): Dropout rate.
        """
        super(AttentionEncoder, self).__init__()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout),
                "fc1": nn.Linear(feature_dim, hidden_dim),
                "activation": nn.GELU(),
                "dropout": nn.Dropout(dropout),
                "fc2": nn.Linear(hidden_dim, feature_dim),
                "norm1": nn.LayerNorm(feature_dim),
                "norm2": nn.LayerNorm(feature_dim),
            }) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        Forward pass for the Attention Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).
            mask (torch.Tensor): Mask for padded inputs of shape (B, T).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, F).
        """
        # Permute for Multi-Head Attention (T, B, F)
        x = x.permute(1, 0, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()  # MultiHeadAttention expects True for masked positions

        for layer in self.layers:
            # Self-Attention
            attn_output, _ = layer["attention"](x, x, x, key_padding_mask=attn_mask)
            x = x + layer["dropout"](attn_output)  # Residual connection
            x = layer["norm1"](x)

            # Feed-Forward Network
            ff_output = layer["fc2"](layer["dropout"](layer["activation"](layer["fc1"](x))))
            x = x + layer["dropout"](ff_output)  # Residual connection
            x = layer["norm2"](x)

        # Permute back to (B, T, F)
        x = x.permute(1, 0, 2)
        return x

class CombinedEncoder(nn.Module):
    def __init__(self, output_dim=256, feature_dim1=128, feature_dim2=64, num_heads=8, hidden_dim=512, 
                 num_layers=2, num_dialects=4, dropout_rate=0.1, get_embeddings=False):
        """
        Improved Combined Encoder for Dialect Identification.

        Args:
            output_dim (int): Output feature dimension.
            feature_dim1 (int): Input feature dimension for enc_out1.
            feature_dim2 (int): Input feature dimension for enc_out2.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of the feed-forward network.
            num_layers (int): Number of attention encoder layers.
            dropout_rate (float): Dropout rate.
        """
        super(CombinedEncoder, self).__init__()

        # Multi-layer attention encoder
        self.attention_encoder = AttentionEncoder(
            feature_dim=feature_dim1, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout_rate
        )
        self.norm = nn.LayerNorm(feature_dim1)
        self.linear = nn.Sequential(
            nn.Linear(feature_dim1, output_dim),
            nn.GELU(),  # Activation
            nn.Dropout(dropout_rate)  # Dropout for regularization
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, num_dialects),
            nn.Dropout(dropout_rate)  # Dropout before classification
        )
        self.dropout = nn.Dropout(dropout_rate)

        # Learnable gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim1 * 2, feature_dim1),
            nn.Sigmoid()
        )
        self.get_embeddings = get_embeddings

    def forward(self, enc_out1, enc_out2, encoder_out_lens, device):
        """
        Forward pass for the improved encoder.

        Args:
            enc_out1 (torch.Tensor): First input tensor of shape (B, T, 128).
            enc_out2 (torch.Tensor): Second input tensor of shape (B, T, 64).
            encoder_out_lens (torch.Tensor): Lengths of encoder outputs (B,).
            device (torch.device): Device for computations.

        Returns:
            torch.Tensor: Dialect classification logits of shape (B, num_dialects).
        """
        # Create mask for padded inputs
        mask = torch.arange(enc_out1.size(1), device=device).unsqueeze(0) < encoder_out_lens.unsqueeze(-1)

        # Compute gating weights
        combined_inputs = torch.cat((enc_out1, enc_out2), dim=-1)  # (B, T, 128 * 2)
        gating_weights = self.gate(combined_inputs)  # (B, T, 128)

        # Weighted combination of enc_out1 and enc_out2
        gated_features = gating_weights * enc_out2 + (1 - gating_weights) * enc_out1

        # Pass through the attention encoder
        attention_output = self.attention_encoder(gated_features, mask)
        
        # Apply layer normalization
        attention_output = self.norm(attention_output)
        attention_output = self.dropout(attention_output)

        # Linear projection
        linear_output = self.linear(attention_output)

        # Mean pooling on the linear output
        pooled_output = mean_pooling(linear_output, mask)

        # Classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if self.get_embeddings:
            return logits, linear_output.detach()
        
        return logits
