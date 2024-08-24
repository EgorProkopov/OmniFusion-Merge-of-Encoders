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
