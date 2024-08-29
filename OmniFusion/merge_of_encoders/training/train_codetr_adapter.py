import random
import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import CSVLogger, WandbLogger  # Import WandbLogger

from OmniFusion.merge_of_encoders.encoders.clip import CLIPVisionTower
from OmniFusion.merge_of_encoders.encoders.utils import initialize_special_embs
from OmniFusion.merge_of_encoders.adapters import VisualToGPTMapping
from OmniFusion.merge_of_encoders.datasets.object_detection_dataset import get_dataset, get_collate_function

from OmniFusion.merge_of_encoders.encoders.codetr import CoDETRVisionTower
class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Config(**v))
            else:
                setattr(self, k, v)

    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in self.__dict__.items())

    def __repr__(self):
        return self.__str__()


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)


class Model_pl(pl.LightningModule):
    def __init__(
            self, cfg, clip, encoder,
            special_embs, model, clip_projection,
            encoder_projection, encoder_name,
            train_dataset, collate_function
    ):
        super().__init__()
        self.DTYPE = torch.float32
        self.cfg = cfg
        self.clip = clip
        self.encoder = encoder
        self.special_embs = special_embs
        self.projection = clip_projection
        self.encoder_projection = encoder_projection
        self.encoder_name = encoder_name
        self.model = model.to(self.DTYPE)
        self.n_embeddings = model.model.embed_tokens.weight.shape[0]
        self.loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=cfg.pad_id)
        self.train_dataset = train_dataset
        self.collate_function = collate_function
        self.n_iters = len(self.train_dataloader())
        self.save_hyperparameters('cfg')
        # self.automatic_optimization = False


    def configure_optimizers(self):

        optimizer = Adafactor(list(self.special_embs.values()) + list(self.projection.parameters()),
                              lr=self.cfg.learning_rate, relative_step=False)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.n_iters // self.cfg.grad_accum * 0.01,
                                                    num_training_steps=self.n_iters // self.cfg.grad_accum)
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1

        }}

    def on_train_epoch_end(self):
        torch.save(self.projection, f"ckpts/{self.cfg.exp_name}/projection.pt")
        torch.save(self.encoder_projection, f"ckpts/{self.cfg.exp_name}/{self.encoder_name}_projection.pt")
        torch.save(self.special_embs, f"ckpts/{self.cfg.exp_name}/special_embeddings.pt")

    def training_step(self, batch, batch_idx):
        clip_embs, encoder_embs, images_mask, labels, mask, positions = batch
        if images_mask.sum() > 0:
            clip_embedding = self.clip(clip_embs).to(dtype=self.DTYPE)  # preprocessing!!!
            encoder_embedding = self.encoder(encoder_embs).to(dtype=self.DTYPE)

            projected_clip_embeddings = self.projection(clip_embedding).to(dtype=self.DTYPE)
            projected_encoder_embeddings = self.encoder_projection(encoder_embedding).to(dtype=self.DTYPE)

            projected_vision_embeddings = torch.cat([
                projected_clip_embeddings, projected_encoder_embeddings
            ], dim=-1).to(dtype=self.DTYPE)

        embeddings = self.model.model.embed_tokens(labels).to(dtype=DTYPE)
        img_idx_counter = 0
        for i in range(len(embeddings)):
            for pos in positions[i]:

                if pos['type'] in self.special_embs.keys():
                    embeddings[i][pos['position']] = self.special_embs[pos['type']].to(dtype=self.DTYPE)
                if pos['type'] == 'IMG':
                    embeddings[i][pos['position'][0]:pos['position'][1]] = projected_vision_embeddings[img_idx_counter].to(dtype=self.DTYPE)
                    img_idx_counter += 1

        embeddings = embeddings[:, :self.cfg.max_context_len]
        labels = labels[:, :self.cfg.max_context_len]
        mask = mask[:, :self.cfg.max_context_len]

        with torch.autocast(device_type="cuda", dtype=self.DTYPE):
            logits = self.model(inputs_embeds=embeddings.to(dtype=self.DTYPE), output_hidden_states=True).get("logits")[
                     :, :-1]

        labels = labels[:, 1:]
        mask = mask[:, 1:]

        logits = logits[mask].contiguous().float()
        labels = labels[mask].contiguous()

        loss = self.loss_fct(logits.view(-1, self.n_embeddings), labels.view(-1)).mean()

        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # if batch_idx % 25000 == 0:
        #     os.makedirs(f"ckpts/{self.cfg.exp_name}/{batch_idx}", exist_ok=True)
        #     torch.save(self.projection, f"ckpts/{self.cfg.exp_name}/{batch_idx}/projection.pt")
        #     torch.save(self.special_embs, f"ckpts/{self.cfg.exp_name}/{batch_idx}/special_embeddings.pt")
        #     torch.save(self.encoder_projection, f"ckpts/{self.cfg.exp_name}/{batch_idx}/{self.encoder_name}_projection.pt")

        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.collate_function,
                          num_workers=self.cfg.num_workers, shuffle=True)


if __name__ == "__main__":
    DTYPE = torch.float32

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config-pretrain.json')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)
    cfg = Config(**config_dict)

    ### Define models
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckp, use_fast=False)
    unk_id = tokenizer.encode("<unk>", add_special_tokens=False)[0]
    cfg.pad_id = unk_id
    os.makedirs(f"ckpts/{cfg.exp_name}", exist_ok=True)
    logger = CSVLogger("ckpts", name=cfg.exp_name)
    
    # Initialize Wandb logger
    wandb_logger = WandbLogger(project="merge_of_encoders", name=cfg.exp_name)

    cfg.exp_name = os.path.join(cfg.exp_name, f'version_{logger.version}')

    model = AutoModelForCausalLM.from_pretrained(cfg.model_ckp, torch_dtype=DTYPE, device_map='cpu')

    clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    clip.load_model()
    clip = clip.to(dtype=DTYPE)

    encoder = CoDETRVisionTower(cfg.encoder_ckp)  # TODO: codetr, plots, ocr, etc.
    encoder.load_model()
    encoder = encoder.to(dtype=DTYPE)

    projection = VisualToGPTMapping(256, cfg.emb_dim, cfg.vision_emb_num).to(dtype=DTYPE)
    # projection.transformer_layer.norm_first = False

    clip_projection = VisualToGPTMapping(1024, cfg.emb_dim, cfg.vision_emb_num).to(dtype=DTYPE)
    clip_projection.load_state_dict(torch.load(cfg.clip_adapter_ckp))
    # clip_projection.transformer_layer.norm_first = False

    special_embs = torch.load(cfg.special_embs_ckp)
    freeze(model), freeze(clip), freeze(encoder), freeze(clip_projection)#, freeze(special_embs)

    train_dataset = get_dataset(cfg, tokenizer, clip.image_processor, encoder.image_processor)
    collate_function = get_collate_function(cfg)


    module = Model_pl(
        cfg=cfg, clip=clip, encoder=encoder,
        special_embs=special_embs, model=model,
        clip_projection=clip_projection, encoder_projection=projection,
        encoder_name="codetr",
        train_dataset=train_dataset, collate_function=collate_function
    )
    
    # Use both CSVLogger and WandbLogger
    trainer = pl.Trainer(
        devices=[0,1, 2], max_epochs=cfg.n_epochs,
        logger=[logger, wandb_logger],
        accumulate_grad_batches=cfg.grad_accum,
        strategy="auto"
    )
    trainer.fit(module)
