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
from pytorch_lightning.loggers import CSVLogger

from OmniFusion.merge_of_encoders.encoders.clip import CLIPVisionTower
from OmniFusion.merge_of_encoders.encoders.utils import initialize_special_embs
from OmniFusion.merge_of_encoders.adapters import VisualToGPTMapping, MLPAdapter
from OmniFusion.merge_of_encoders.datasets.image_captioning_dataset import get_dataset, get_collate_function
        

class Config:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
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
    def __init__(self, cfg, clip, special_embs, model, projection, train_dataset, collate_function):
        super().__init__()
        self.cfg = cfg
        self.clip = clip
        self.special_embs = special_embs
        self.projection = projection
        self.model = model
        self.n_embeddings = model.model.embed_tokens.weight.shape[0]
        self.loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=cfg.pad_id)
        self.train_dataset = train_dataset
        self.collate_function = collate_function
        self.n_iters = len(self.train_dataloader())
        self.save_hyperparameters('cfg')
        # self.automatic_optimization = False

        self.DTYPE = torch.float16

    def configure_optimizers(self):
        
        optimizer = Adafactor(list(self.special_embs.values()) + list(self.projection.parameters()), lr=self.cfg.learning_rate, relative_step=False)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.n_iters // self.cfg.grad_accum * 0.01, num_training_steps=self.n_iters // self.cfg.grad_accum)
        # return {'optimizer': optimizer, 'lr_scheduler': {
        #     'scheduler': scheduler,
        #     'interval': 'step',
        #     'frequency': 1
        #
        # }}
        return {'optimizer': optimizer}

    def on_train_epoch_end(self):
        torch.save(self.projection, f"ckpts/{self.cfg.exp_name}/projection.pt")
        torch.save(self.special_embs, f"ckpts/{self.cfg.exp_name}/special_embeddings.pt")

    def training_step(self, batch, batch_idx):
        images, images_mask, labels, mask, positions = batch
        if images_mask.sum() > 0:
            image_embedding = self.clip(images).to(dtype=self.DTYPE) # preprocessing!!!
            projected_vision_embeddings = self.projection(image_embedding)
        
        embeddings = self.model.model.embed_tokens(labels)
        img_idx_counter = 0
        for i in range(len(embeddings)):
            for pos in positions[i]:
                
                if pos['type'] in self.special_embs.keys():
                    embeddings[i][pos['position']] = self.special_embs[pos['type']]
                if pos['type'] == 'IMG':
                    embeddings[i][pos['position'][0]:pos['position'][1]] = projected_vision_embeddings[img_idx_counter]
                    img_idx_counter += 1
        
        embeddings = embeddings[:, :self.cfg.max_context_len]
        labels = labels[:, :self.cfg.max_context_len]
        mask = mask[:, :self.cfg.max_context_len]
        
        with torch.autocast(device_type="cuda", dtype=self.DTYPE):
            logits = self.model(inputs_embeds=embeddings.to(dtype=self.DTYPE), output_hidden_states=True).get("logits")[:, :-1]
        
        labels = labels[:, 1:]
        mask = mask[:, 1:]
          
        logits = logits[mask].contiguous().float()
        labels = labels[mask].contiguous()

        
        loss = self.loss_fct(logits.view(-1, self.n_embeddings), labels.view(-1)).mean()
            
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if batch_idx % 25000 == 0:
            pass
            # if not os.path.exists(f"ckpts/{self.cfg.exp_name}/{batch_idx}"):
            # os.mkdir(f"ckpts/{self.cfg.exp_name}/{batch_idx}", exist_ok = True)
            # os.makedirs(f"ckpts/{self.cfg.exp_name}/{batch_idx}", exist_ok=True)
            # torch.save(self.projection, f"ckpts/{self.cfg.exp_name}/{batch_idx}/projection.pt")
            # torch.save(self.special_embs, f"ckpts/{self.cfg.exp_name}/{batch_idx}/special_embeddings.pt")
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.collate_function, num_workers=self.cfg.num_workers, shuffle=True)


if __name__ == "__main__":
    DTYPE = torch.float16

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
    cfg.exp_name = os.path.join(cfg.exp_name, f'version_{logger.version}')

    model = AutoModelForCausalLM.from_pretrained(cfg.model_ckp, torch_dtype=DTYPE, device_map='cpu')

    clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    clip.load_model()
    clip = clip.to(dtype=DTYPE)

    projection = MLPAdapter(1024, cfg.emb_dim).to(dtype=DTYPE)
    # projection.transformer_layer.norm_first = False
    
    special_embs = initialize_special_embs(emb_dim=cfg.emb_dim, device='cpu', dtype=DTYPE)
    freeze(model), freeze(clip)

    train_dataset = get_dataset(cfg, tokenizer, clip.image_processor)
    collate_function = get_collate_function(cfg)

    module = Model_pl(cfg, clip, special_embs, model, projection, train_dataset, collate_function)
    trainer = pl.Trainer(
        devices=[0, 2], max_epochs=cfg.n_epochs,
        logger=logger, accumulate_grad_batches=cfg.grad_accum,
        # strategy='auto'
    )
    trainer.fit(module)
