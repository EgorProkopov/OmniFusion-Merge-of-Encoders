import math

import torch
import torch.nn as nn


class MLPAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out


class VisualToGPTMapping(nn.Module):
    def __init__(self, visual_emb_dim, gpt_emb_dim, num_gpt_embs, num_heads):
        super(VisualToGPTMapping, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=visual_emb_dim, nhead=num_heads, batch_first=True, norm_first=False)
        self.linear = nn.Linear(visual_emb_dim, gpt_emb_dim)
        self.n_embeddings = num_gpt_embs
        self.embedding_dim = gpt_emb_dim
    def forward(self, visual_embs):
        out = self.transformer_layer(visual_embs)
        out = self.linear(out).view(-1, self.n_embeddings, self.embedding_dim)
        return out


class QFormer(nn.Module):
    def __init__(self,
                 visual_hidden_dim: int,
                 query_dim: int,
                 num_queries: int,
                 transformer_hidden_dim: int,
                 num_transformer_layers: int,
                 num_heads: int):
        super(QFormer, self).__init__()

        # Learnable query vectors
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim))

        # Linear projection from visual encoder dimension to transformer hidden dimension
        self.visual_projection = nn.Linear(visual_hidden_dim, transformer_hidden_dim)

        # Transformer encoder to process queries and visual features
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * transformer_hidden_dim,
                dropout=0.1,
                activation='relu'
            ),
            num_layers=num_transformer_layers
        )

        # Output projection to get final query embeddings
        self.output_projection = nn.Linear(transformer_hidden_dim, query_dim)

    def forward(self, visual_features: torch.Tensor):
        """
        visual_features: Tensor of shape (batch_size, num_patches, visual_hidden_dim)
        """
        batch_size = visual_features.size(0)

        # Project visual features to the transformer hidden dimension
        projected_visual_features = self.visual_projection(
            visual_features)  # Shape: (batch_size, num_patches, transformer_hidden_dim)

        # Repeat the queries across the batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_queries, query_dim)

        # Concatenate queries and visual features
        combined_features = torch.cat([queries, projected_visual_features],
                                      dim=1)  # Shape: (batch_size, num_queries + num_patches, transformer_hidden_dim)

        # Apply the transformer
        transformed_features = self.transformer(combined_features.permute(1, 0,
                                                                          2))  # Shape: (num_queries + num_patches, batch_size, transformer_hidden_dim)
        transformed_features = transformed_features.permute(1, 0,
                                                            2)  # Shape: (batch_size, num_queries + num_patches, transformer_hidden_dim)

        # Extract the transformed queries
        transformed_queries = transformed_features[:, :self.queries.size(0),
                              :]  # Shape: (batch_size, num_queries, transformer_hidden_dim)

        # Project the transformed queries to the query dimension
        output_queries = self.output_projection(transformed_queries)  # Shape: (batch_size, num_queries, query_dim)

        return output_queries


class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens, f"In downsampling {h}^2 != {num_tokens}"
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens, f"In PE {h}^2 != {num_tokens}"
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x


class LDPNetV2Projector(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, num_mm_tokens):
        super().__init__()
        inc, ouc = mm_hidden_size, hidden_size
        n_tokens = int(math.sqrt(num_mm_tokens))
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer((n_tokens, n_tokens))
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x
