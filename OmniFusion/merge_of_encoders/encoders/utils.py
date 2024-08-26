import torch


def initialize_special_embs(emb_dim=896, device="cpu", dtype=torch.float16):
    special_embs = {}
    special_embs['USER'] = torch.normal(
        torch.zeros((emb_dim, )), torch.ones((emb_dim, )) / emb_dim ** 0.5
    ).to(device=device, dtype=dtype)
    special_embs['BOT'] = torch.normal(
        torch.zeros((emb_dim, )), torch.ones((emb_dim, )) / emb_dim ** 0.5
    ).to(device=device, dtype=dtype)
    special_embs['SOI'] = torch.normal(
        torch.zeros((emb_dim, )), torch.ones((emb_dim, )) / emb_dim ** 0.5
    ).to(device=device, dtype=dtype)
    special_embs['EOI'] = torch.normal(
        torch.zeros((emb_dim, )), torch.ones((emb_dim, )) / emb_dim ** 0.5
    ).to(device=device, dtype=dtype)

    special_embs['SOI'].requires_grad_()
    special_embs['EOI'].requires_grad_()
    special_embs['USER'].requires_grad_()
    special_embs['BOT'].requires_grad_()
    return special_embs
