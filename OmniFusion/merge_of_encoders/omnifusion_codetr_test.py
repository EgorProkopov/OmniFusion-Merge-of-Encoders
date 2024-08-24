import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.request import urlopen
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Loading some sources of the projection adapter and image encoder
# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="models.py", local_dir='./')
from models import CLIPVisionTower
from our_models import CoDETRVisionTower
from tower_mixer import MLPMixer

DEVICE = "cuda:0"
PROMPT = "This is a dialog with AI assistant.\n"

tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tokenizer", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("AIRI-Institute/OmniFusion", subfolder="OmniMistral-v1_1/tuned-model", torch_dtype=torch.float32, device_map=DEVICE)

hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='../')
hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/special_embeddings.pt", local_dir='../')
clip_projection = torch.load("../OmniMistral-v1_1/projection.pt", map_location=DEVICE)
clip_projection = clip_projection.to(dtype=torch.float32)
mlp_projection = MLPMixer(in_dim=256, out_dim=256)
mlp_projection = mlp_projection.to(DEVICE)
special_embs = torch.load("../OmniMistral-v1_1/special_embeddings.pt", map_location=DEVICE)

clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
clip.load_model()
clip = clip.to(device=DEVICE, dtype=torch.float32)

codetr = CoDETRVisionTower("microsoft/conditional-detr-resnet-50")
codetr.load_model()
codetr = codetr.to(device=DEVICE, dtype=torch.float32)


def gen_answer(model, tokenizer, clip, codetr, clip_projection, codetr_projection, query, special_embs, image=None):
    bad_words_ids = tokenizer(["\n", "</s>", ":"], add_special_tokens=False).input_ids + [[13]]
    gen_params = {
        "do_sample": False,
        "max_new_tokens": 50,
        "early_stopping": True,
        "num_beams": 3,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "forced_eos_token_id": 2,
        "use_cache": True,
        "no_repeat_ngram_size": 4,
        "bad_words_ids": bad_words_ids,
        "num_return_sequences": 1,
    }
    with torch.no_grad():
        clip_image_features = clip.image_processor(image, return_tensors='pt')
        clip_image_embedding = clip(clip_image_features['pixel_values']).to(device=DEVICE, dtype=torch.float32)
        codetr_image_features = codetr.image_processor(image, return_tensors='pt')
        codetr_image_embedding = codetr(codetr_image_features['pixel_values']).to(device=DEVICE, dtype=torch.float32)

        clip_image_embedding = clip_projection(clip_image_embedding)
        codetr_image_embedding = codetr_projection(codetr_image_embedding)

        prompt_ids = tokenizer.encode(f"{PROMPT}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        question_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)

        prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.float32)
        question_embeddings = model.model.embed_tokens(question_ids).to(torch.float32)

        embeddings = torch.cat(
            [
                prompt_embeddings,
                special_embs['SOI'][None, None, ...],
                clip_image_embedding,
                special_embs['EOI'][None, None, ...],

                special_embs['SOI'][None, None, ...],
                codetr_image_embedding,
                special_embs['EOI'][None, None, ...],

                special_embs['USER'][None, None, ...],
                question_embeddings,
                special_embs['BOT'][None, None, ...]
            ],
            dim=1,
        ).to(dtype=torch.float32, device=DEVICE)
        out = model.generate(inputs_embeds=embeddings, **gen_params)
    out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts

img_url = "https://i.pinimg.com/originals/32/c7/81/32c78115cb47fd4825e6907a83b7afff.jpg"
question = "Describe this image?"
img = Image.open(urlopen(img_url))

answer = gen_answer(
    model,
    tokenizer,
    clip,
    codetr,
    clip_projection=clip_projection,
    codetr_projection=mlp_projection,
    query=question,
    special_embs=special_embs,
    image=img
)

img.show()
print(question)
print(answer)
