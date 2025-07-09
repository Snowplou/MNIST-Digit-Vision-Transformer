import torch
from torch import nn
from config import *

class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.k_dim = int(EMBED_DIM / NUM_HEADS)
        self.k_sqrt = self.k_dim ** 0.5

        self.query_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.key_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.value_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_O = nn.Linear(EMBED_DIM, EMBED_DIM)

        self.normalization = nn.LayerNorm(EMBED_DIM)

        self.linear_1 = nn.Linear(EMBED_DIM, EMBED_DIM * 4)
        self.linear_2 = nn.Linear(EMBED_DIM * 4, EMBED_DIM)
        self.gelu = nn.GELU()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, seq_len, NUM_HEADS, self.k_dim)
        K = K.view(batch_size, seq_len, NUM_HEADS, self.k_dim)
        V = V.view(batch_size, seq_len, NUM_HEADS, self.k_dim)

        # Transpose to (batch_size, NUM_HEADS, sequence_length, k_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.k_sqrt
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        Z = torch.matmul(attention_weights, V)

        # Concatenate the attention outputs and project back to original dimension
        Z = Z.transpose(1, 2).contiguous() # (batch_size, sequence_length, NUM_HEADS, k_dim)
        Z = Z.view(batch_size, seq_len, EMBED_DIM)

        # Apply W_O (linear projection of concatenated heads)
        Z = self.W_O(Z)

        # Apply addition and normalization
        x = x + Z
        x = self.normalization(x)

        # Apply the feed-forward network
        L1 = self.linear_1(x)
        L1 = self.gelu(L1)
        L2 = self.linear_2(L1)

        # Apply addition and normalization
        x = x + L2
        x = self.normalization(x)

        return x

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.patch_embedding = nn.Conv2d(1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))
        self.positional_embedding = nn.Parameter(torch.rand(1, IMAGE_SIZE[0] * IMAGE_SIZE[1] + 1, EMBED_DIM)) # +1 for the cls token

        self.attention_blocks = nn.ModuleList(
            [AttentionBlock() for _ in range(NUM_ATTENTION_LAYERS)]
        )

        self.final_linear = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        # Reshape to (batch_size, 1 (channel), INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1])
        x = x.view(-1, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]).unsqueeze(1)

        # Apply patch embedding
        x = self.patch_embedding(x)

        # Reshape to (batch_size, EMBED_DIM, IMAGE_SIZE[0] * IMAGE_SIZE[1])
        x = x.view(x.size(0), EMBED_DIM, -1)

        # Transpose to (batch_size, IMAGE_SIZE[0] * IMAGE_SIZE[1], EMBED_DIM)
        x = x.transpose(1, 2)

        # Prepend classification token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.positional_embedding

        # Pass through attention blocks
        for block in self.attention_blocks:
            x = block(x)

        # Use the output of the classification token for classification
        x = x[:, 0]

        # Final linear layer to output logits
        x = self.final_linear(x)

        return x # CrossEntropyLoss automatically applies softmax