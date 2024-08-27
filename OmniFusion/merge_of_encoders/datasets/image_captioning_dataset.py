import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import transformers

from datasets import load_dataset, DatasetDict


def load_llava_recap_558k():
    return load_dataset("lmms-lab/LLaVA-ReCap-558K")
    # cropped_train = ds['train'].select(range(10))
    #
    # # Create a new DatasetDict with the cropped train dataset
    # cropped_dataset_dict = DatasetDict({
    #     'train': cropped_train
    # })
    # return cropped_dataset_dict

class ImageCaptioning(Dataset):
    def __init__(
            self, cfg, data, tokenizer, image_processor,
            transforms=None, augmentation=None
    ):
        super().__init__()
        self.cfg = cfg
        self.data = data

        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.transforms = transforms
        self.augmentation = augmentation

    def _sample_question(self):
        questions_pool = [
            "Describe this image.",
            "Describe the image.",

            "Whats happening on this image?",
            "Give the description of this image.",
            "What objects and people are visible on this image?",
            "Describe objects and people that are visible on this image.",
            "What kind of scene is this?",
            "Give the explanation of this image."
        ]
        question = np.random.choice(questions_pool, size=1, replace=False)
        return question

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, item):
        data_item = self.data[item]
        tokens = []
        positions = []

        prompt_tokens = self.tokenizer.encode(f"{self.cfg.prompt}", add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_tokens.shape[-1]
        tokens.append(prompt_tokens)
        mask = prompt_len * [False]

        image = None
        if 'image' in data_item.keys():
            image = self.image_processor(data_item['image'], return_tensors='pt')['pixel_values'][0]

            tokens.append(
                torch.tensor(
                    [(self.cfg.vision_emb_num + 2) * [self.cfg.pad_id]],
                    dtype=torch.int64,
                )
            )

            positions += [
                {'type': 'SOI', 'position': prompt_len},
                {'type': 'IMG', 'position': (prompt_len + 1, prompt_len + 1 + self.cfg.vision_emb_num)},
                {'type': 'EOI', 'position': prompt_len + 1 + self.cfg.vision_emb_num}
            ]

            mask += (2 + self.cfg.vision_emb_num) * [False]

        for conversation in data_item['conversations']:
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
            if conversation['from'] == 'human':
                text = text.replace("<image>", self._sample_question()[0])
            else:
                text += self.tokenizer.eos_token

            text_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            tokens.append(text_tokens)
            if conversation['from'] == 'human':
                mask += text_tokens.shape[-1] * [False]
            else:
                mask += text_tokens.shape[-1] * [True]

        tokens = torch.cat(tokens, dim=-1)[0]
        mask = torch.tensor(mask, dtype=bool)
        return image, tokens, mask, positions


def get_dataset(cfg, tokenizer, image_processor):
    data = load_llava_recap_558k()['train']
    return ImageCaptioning(cfg, data, tokenizer, image_processor)


def get_collate_function(cfg):
    def colate_fn(data):
        images, tokens, masks, positions = zip(*data)

        images_mask = torch.tensor([True if image is not None else False for image in images], dtype=bool)
        if images_mask.sum() > 0:
            images = torch.stack([image for image in images if image is not None])
        else:
            images = None
        tokens = list(tokens)
        masks = list(masks)
        positions = list(positions)
        max_len = max([token.shape[-1] for token in tokens])
        for i in range(len(tokens)):
            pad_len = max_len - tokens[i].shape[-1]
            masks[i] = torch.cat([masks[i], torch.tensor(pad_len*[False], dtype=bool)], dim=0)
            tokens[i] = torch.cat([tokens[i], torch.tensor(pad_len*[cfg.pad_id], dtype=int)], dim=0)
        masks = torch.stack(masks)
        tokens = torch.stack(tokens)
        return images, images_mask, tokens, masks, positions
    return colate_fn


if __name__ == "__main__":
    data = load_llava_recap_558k()
    print(data)
