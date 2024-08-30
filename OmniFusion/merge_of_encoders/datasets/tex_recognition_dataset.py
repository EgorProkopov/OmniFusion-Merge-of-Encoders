import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import transformers

from datasets import load_dataset, DatasetDict


def load_im2latex_dataset():
    dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")
    print(dataset)    
    return dataset
    # cropped_train = ds['train'].select(range(10))
    #
    # # Create a new DatasetDict with the cropped train dataset
    # cropped_dataset_dict = DatasetDict({
    #     'train': cropped_train
    # })
    # return cropped_dataset_dict

class ImageCaptioning(Dataset):
    def __init__(
            self, 
            cfg, 
            data, 
            tokenizer, 
            image_processor,
            encoder_image_processor,
            transforms=None, 
            augmentation=None, 
            max_length=128
    ):
        super().__init__()
        self.cfg = cfg
        self.data = data

        self.tokenizer = tokenizer
        self.clip_image_processor = image_processor
        self.tex_image_processor = encoder_image_processor

        self.transforms = transforms if transforms is not None else  self.get_train_transform()
        self.augmentation = augmentation
        self.max_length = max_length


    def get_train_transform(self):
        def train_transform(image):
            image = image.resize(self.image_size)
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            return image
        return train_transform
    

    def _sample_question(self):
        questions_pool = [
            "What mathematical expression is depicted in this image?",
            "Identify the LaTeX formula shown in this image.",
            "What is the equation represented in this image?",
            "Translate the formula from the image into LaTeX.",
            "What mathematical symbols are present in this image?",
            "What does this image represent in mathematical notation?",
            "Convert the image content into a LaTeX expression.",
            "Describe the equation displayed in this image.",
            "What is the formula visible in this image?",
            "Interpret the mathematical content of this image."
        ]

        question = np.random.choice(questions_pool, size=1, replace=False)
        return question[0]

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, item):
        data_item = self.data[item]
        
        question = self._sample_question()

        conversations = [{'from': 'human', 'value': question}, {'from':'synt_captioning', 'value': data_item['latex_formula']}]
        
        tokens = []
        positions = []

        prompt_tokens = self.tokenizer.encode(
            f"{self.cfg.prompt}", padding="max_length",
            max_length=self.max_length, truncation=True,
            add_special_tokens=False, return_tensors="pt"
        )
        prompt_len = prompt_tokens.shape[-1]
        tokens.append(prompt_tokens)
        mask = prompt_len * [False]
        
        clip_embs, tex_embs = None, None
        if 'image' in data_item.keys():
            clip_embs = self.clip_image_processor(data_item['image'], return_tensors='pt')['pixel_values'][0]
            tex_embs = self.tex_image_processor(data_item['image'], return_tensors='pt')['pixel_values'][0]
            
            tokens.append(
                torch.tensor(
                    [(self.cfg.vision_emb_num + self.cfg.encoder_emb_num + 2) * [self.cfg.pad_id]],
                    dtype=torch.int64,
                )
            )

            positions += [
                {'type': 'SOI', 'position': prompt_len},
                {'type': 'IMG', 'position': (prompt_len + 1, prompt_len + 1 + self.cfg.vision_emb_num + self.cfg.encoder_emb_num)}, #TODO
                {'type': 'EOI', 'position': prompt_len + 1 + self.cfg.vision_emb_num + self.cfg.encoder_emb_num}
            ]

            mask += (2 + self.cfg.vision_emb_num + self.cfg.encoder_emb_num) * [False]

        for conversation in conversations:
            if conversation['from'] == 'human':
                positions.append({'type': 'USER', 'position': len(mask)})
            else:  # from gpt
                positions.append({'type': 'BOT', 'position': len(mask)})
            mask += [False]
            tokens.append(
                torch.tensor(
                    [[self.cfg.pad_id]],
                    dtype=torch.int64,
                )
            )

            text = conversation['value']
            if conversation['from'] != 'human':
                text += self.tokenizer.eos_token

            text_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            tokens.append(text_tokens)
            if conversation['from'] == 'human':
                mask += text_tokens.shape[-1] * [False]
            else:
                mask += text_tokens.shape[-1] * [True]

        tokens = torch.cat(tokens, dim=-1)[0]
        mask = torch.tensor(mask, dtype=bool)
        return clip_embs, tex_embs, tokens, mask, positions


def get_dataset(cfg, tokenizer, image_processor, encoder_image_processor):
    data = load_im2latex_dataset()['train']
    return ImageCaptioning(cfg, data, tokenizer, image_processor, encoder_image_processor)


def get_collate_function(cfg):
    def colate_fn(data):
        clip_embs, encoder_embs, tokens, masks, positions = zip(*data)

        images_mask = torch.tensor([True if image is not None else False for image in clip_embs], dtype=bool)
        
        if images_mask.sum() > 0:
            clip_embs = torch.stack([image for image in clip_embs if image is not None])
            encoder_embs = torch.stack([image for image in encoder_embs if image is not None])
        else:
            clip_embs = None
            encoder_embs = None
            
        tokens = list(tokens)
        masks = list(masks)
        positions = list(positions)
        max_len = max([token.shape[-1] for token in tokens])
        for i in range(len(tokens)):
            pad_len = max_len - tokens[i].shape[-1]
            masks[i] = torch.cat([masks[i], torch.tensor(pad_len * [False], dtype=bool)], dim=0)
            tokens[i] = torch.cat([tokens[i], torch.tensor(pad_len * [cfg.pad_id], dtype=int)], dim=0)

        masks = torch.stack(masks)
        tokens = torch.stack(tokens)
        return clip_embs, encoder_embs, images_mask, tokens, masks, positions
    return colate_fn


if __name__ == "__main__":
    data = load_im2latex_dataset()
    print(data)
