import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig

# Utility function for mean pooling
def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

class RobertaEncoder(nn.Module):
    def __init__(self, ctc, lin_out_dim=5, input_size=72, hidden_size=64, num_hidden_layers=2, dropout_rate=0.1, get_embeddings=False):
        super().__init__()
        self.ctc = ctc
        self.get_embeddings = get_embeddings
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),  # Activation function for non-linearity
            nn.Dropout(dropout_rate)  # Dropout to prevent overfitting
        )
        
        # Roberta configuration
        config = RobertaConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,  # Adjust for efficiency
            max_position_embeddings=4096,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout_rate,  # Dropout within Roberta layers
            attention_probs_dropout_prob=dropout_rate  # Dropout for attention weights
        )
        
        # Initialize Roberta model
        self.roberta = RobertaModel(config)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        if not self.get_embeddings:
            self.output_linear = nn.Sequential(
                nn.Dropout(dropout_rate),  # Dropout before the final classification layer
                nn.Linear(hidden_size, lin_out_dim)
            )

    def forward(self, encoder_out, encoder_out_lens):
        """
        Forward pass of the Roberta-based encoder.
        :param encoder_out: Output from the encoder.
        :param encoder_out_lens: Lengths of encoder output sequences.
        """
        # Apply CTC to get logits
        logits = self.ctc.softmax(encoder_out)

        # Create attention mask using encoder_out_lens
        attention_mask = (
            torch.arange(logits.size(1), device=logits.device)
            .expand(len(encoder_out_lens), logits.size(1))
            < encoder_out_lens.unsqueeze(1)
        ).float()

        # Adapt input size to Roberta's hidden size
        x = self.linear(logits)
        
        # Generate embeddings using Roberta
        outputs = self.roberta(inputs_embeds=x, attention_mask=attention_mask)
        
        # Apply layer normalization
        rob_embed = self.norm(outputs.last_hidden_state)
        
        if self.get_embeddings:
            return rob_embed
        
        pooled_output = mean_pooling(rob_embed, attention_mask)  # Use global mean pooling
        # Fully connected layer for logits
        rob_out = self.output_linear(pooled_output)
        return rob_out
