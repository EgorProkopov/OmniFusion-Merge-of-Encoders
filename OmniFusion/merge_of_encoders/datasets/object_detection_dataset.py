import numpy as np

import torch
from torch.utils.data import Dataset

from datasets import load_dataset


def download_coco():
    return load_dataset("detection-datasets/coco")


class COCOODImageCaptioning(Dataset):
    def __init__(
            self, cfg, data, tokenizer,
            clip_image_processor, encoder_image_processor, max_length=64,
    ):
        super().__init__()
        self.cfg = cfg
        self.data = data

        self.tokenizer = tokenizer
        self.clip_image_processor = clip_image_processor
        self.encoder_image_processor = encoder_image_processor

        self.categories_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]
        self.questions_pool = [
            "Detect objects on this image.",
            "Which objects are depicted on this image and where are they located?",
            "Locate objects on this image and find their area.",
            "Find objects on this image and calculate their area.",
            "Which objects can you find on this image?",
            "Call objects on this image and find their area.",
            "What can you tell about objects on this image",
            "Find area of this objects.",
            "Which object in this picture has the largest area.",
            "Name the objects on this image."
        ]

        self.max_length = max_length

    def get_category_name_by_id(self, id):
        return self.categories_names[id]

    def get_text_captioning(self, bboxes_list, categories_list, areas_list, width_scale, height_scale):
        num_objects = len(categories_list)
        text_captioning = f"This image contains the following {num_objects} objects: "
        if len(bboxes_list) == 0:
            text_captioning = f"There is no objects on this image"
            return text_captioning

        biggest_area = 0
        biggest_object = None
        for bbox, category, area in zip(bboxes_list, categories_list, areas_list):
            category_name = self.get_category_name_by_id(category)

            x1, y1, x2, y2 = bbox

            x1, x2 = x1 / width_scale, x2 / width_scale
            y1, y2 = y1 / height_scale, y2 / height_scale
            area = area / (width_scale * height_scale)

            object_description = (f"the {category_name} is in a part of an image starting "
                                  f"from the coordinates at the lower-left corner ({int(x1)}, {int(y1)}) "
                                  f"and ending at the upper-right corner ({int(x2)}, {int(y2)}), the area is {round(area, 2)}, ")
            if area > biggest_area:
                biggest_area = area
                biggest_object = category_name

            text_captioning += object_description

        biggest_object_text = f"the biggest object is {biggest_object} with the area of {round(biggest_area, 2)}."
        text_captioning += biggest_object_text

        return text_captioning

    def process_coco_sample_to_image_captioning(self, data_sample, width_scale, height_scale):
        image = data_sample['image']

        objects = data_sample['objects']
        categories_list = objects['category']
        bboxes_list = objects['bbox']
        areas_list = objects['area']

        text_captioning = self.get_text_captioning(bboxes_list, categories_list, areas_list, width_scale, height_scale)
        return image, text_captioning

    def _sample_question(self):
        question = np.random.choice(self.questions_pool, size=1, replace=False)
        return question

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, item):
        data_item = self.data[item]
        # TODO: check scale
        image, text_captioning = self.process_coco_sample_to_image_captioning(data_item, width_scale=1.0, height_scale=1.0)
        question = self._sample_question()

        conversations = [{'from': 'human', 'value': question}, {'from':'synt_captioning', 'value': text_captioning}]

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

        # image processing
        clip_embs = self.clip_image_processor(image, return_tensors='pt')['pixel_values'][0]
        encoder_embs = self.encoder_image_processor(image, return_tensors='pt')['pixel_values'][0]

        tokens.append(
            torch.tensor(
                [(self.cfg.vision_emb_num + self.cfg.encoder_emb_num + 2) * [self.cfg.pad_id]],
                dtype=torch.int64,
            )
        )

        positions += [
            {'type': 'SOI', 'position': prompt_len},
            {'type': 'IMG', 'position': (prompt_len + 1, prompt_len + 1 + self.cfg.vision_emb_num + self.cfg.encoder_emb_num)},
            {'type': 'EOI', 'position': prompt_len + 1 + self.cfg.vision_emb_num + self.cfg.encoder_emb_num}
        ]

        mask += (self.cfg.vision_emb_num + self.cfg.encoder_emb_num + 2) * [False]

        # text processing
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

            text_tokens = self.tokenizer.encode(
                text, padding="max_length",
                max_length=self.max_length, truncation=True,
                add_special_tokens=False, return_tensors="pt"
            )
            tokens.append(text_tokens)
            if conversation['from'] == 'human':
                mask += text_tokens.shape[-1] * [False]
            else:
                mask += text_tokens.shape[-1] * [True]

        tokens = torch.cat(tokens, dim=-1)[0]
        mask = torch.tensor(mask, dtype=bool)

        return clip_embs, encoder_embs, tokens, mask, positions


def get_dataset(cfg, tokenizer, image_processor, encoder_image_processor, max_len=64):
    data = download_coco()['train']
    return COCOODImageCaptioning(
        cfg, data,
        tokenizer, image_processor, encoder_image_processor, max_length=max_len
    )


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
    data = download_coco()


