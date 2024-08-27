import os
from dotenv import load_dotenv


load_dotenv() 
wandb_api_key = os.getenv("WANDB_API_KEY")

if wandb_api_key:
    os.environ["WANDB_API_KEY"] = wandb_api_key
else:
    raise ValueError("WANDB_API_KEY не найден в .env файле")
