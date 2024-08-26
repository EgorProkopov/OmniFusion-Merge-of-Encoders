from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as albu
import transformers

from datasets import load_dataset


def load_llava_recap_558k():
    return load_dataset("lmms-lab/LLaVA-ReCap-558K")


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

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, item):
        data_item = self.data[item]
        tokens = []
        masks = []
        positions = []

        prompt_tokens = self.tokenizer.encode(f"{self.cfg.prompt}", add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_tokens.shape[-1]
        tokens.append(prompt_tokens)
        mask = prompt_len * [False]

        image = None
        if 'image' in data_item.keys():
            image_path = f"{self.cfg.image_folder}/{data_item['image']}"
            image = self.image_processor((Image.open(image_path)), return_tensors='pt')['pixel_values'][0]

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
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            tokens.append(text_tokens)
            if conversation['from'] == 'human':
                mask += text_tokens.shape[-1] * [False]
            else:
                mask += text_tokens.shape[-1] * [True]

        tokens = torch.cat(tokens.append(self.tokenizer.eos_token), dim=-1)[0]
        mask = torch.tensor(mask, dtype=bool)
        return image, tokens, mask, positions


if __name__ == "__main__":
    data = load_llava_recap_558k()
    print(data)
