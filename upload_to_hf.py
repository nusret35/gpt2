import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
from torch.nn import functional as F
from utils import Block, CausalSelfAttention, MLP, GPTConfig, GPT


device_type = "cpu"
model = GPT(GPTConfig(vocab_size=50304))
checkpoint = torch.load('log/model_19072.pt')
model.load_state_dict(checkpoint['model'])


# push to the hub
model.push_to_hub("gpt2-model-19072")
