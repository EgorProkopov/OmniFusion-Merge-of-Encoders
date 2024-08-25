import albumentations as albu
import torch
from torch.utils.data import Dataset
import transformers

from datasets import load_dataset


def load_llava_recap_558k():
    return load_dataset("lmms-lab/LLaVA-ReCap-558K")


class ImageCaptioning(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, item):
        data_sample = self.data[item]

        # process data to visual language model
        ...

if __name__ == "__main__":
    data = load_llava_recap_558k()
    print(data)
