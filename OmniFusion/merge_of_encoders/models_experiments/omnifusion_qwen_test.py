'''
Qwen2 requires transformers>=4.37.0
Qwen2 embedding dim=896
'''
from urllib.request import urlopen

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from OmniFusion.merge_of_encoders.encoders.clip import CLIPVisionTower
from OmniFusion.merge_of_encoders.encoders.utils import initialize_special_embs
from OmniFusion.merge_of_encoders.adapters import VisualToGPTMapping, LDPNetV2Projector

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
EMB_DIM = 1536
PROMPT = "This is a dialog with AI assistant.\n"


def gen_answer(
        model, tokenizer, clip, clip_projection, query, special_embs, image=None
):
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
        clip_image_embedding = clip(clip_image_features['pixel_values']).to(device=DEVICE, dtype=DTYPE)
        projected_vision_embeddings = clip_projection(clip_image_embedding)

        prompt_ids = tokenizer.encode(f"{PROMPT}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        question_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)

        prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
        question_embeddings = model.model.embed_tokens(question_ids).to(torch.bfloat16)

        embeddings = torch.cat(
            [
                prompt_embeddings,
                special_embs['SOI'][None, None, ...],
                projected_vision_embeddings,
                special_embs['EOI'][None, None, ...],
                special_embs['USER'][None, None, ...],
                question_embeddings,
                special_embs['BOT'][None, None, ...]
            ],
            dim=1,
        ).to(dtype=DTYPE, device=DEVICE)
        out = model.generate(inputs_embeds=embeddings, **gen_params)
    out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]
    return generated_texts


if __name__ == "__main__":
    hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/special_embeddings.pt",
                    local_dir='../')

    clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    clip.load_model()
    clip = clip.to(device=DEVICE, dtype=DTYPE)

    # mlp_projector = VisualToGPTMapping(
    #     visual_emb_dim=1024,
    #     gpt_emb_dim=EMB_DIM,
    #     num_gpt_embs=576,
    #     num_heads=4
    # )
    # mlp_projector.load_state_dict(torch.load("./ckpts/qwen2-15B-pretrain/matvey_weights/projection.pt"))
    # mlp_projector = mlp_projector.to(device=DEVICE, dtype=DTYPE)
    ldp_projector = VisualToGPTMapping(
        visual_emb_dim=1024,
        gpt_emb_dim=EMB_DIM,
        num_tokens=576,
    )
    ldp_projector.load_state_dict(torch.load("./ckpts/qwen2-15B-pretrain/matvey_weights/projection_ldp.pth"))
    ldp_projector = ldp_projector.to(device=DEVICE, dtype=DTYPE)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", )
    special_embs = initialize_special_embs(emb_dim=EMB_DIM, dtype=DTYPE, device=DEVICE)
    special_embs.load_state_dict(torch.load("./ckpts/qwen2-15B-pretrain/matvey_weights/special_embeddings-2.pt"))
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen2-1.5B",
        torch_dtype=DTYPE,
        device_map=DEVICE
    )

    img_url = "https://i.pinimg.com/originals/32/c7/81/32c78115cb47fd4825e6907a83b7afff.jpg"
    question = "Describe this image"

    img = Image.open(urlopen(img_url))

    answer = gen_answer(
        model,
        tokenizer,
        clip,
        ldp_projector,
        query=question,
        special_embs=special_embs,
        image=img
    )

    img.show()
    print(question)
    print(answer)
