import torch
import tiktoken
from torch.nn import functional as F
from utils import Block, CausalSelfAttention, MLP, GPTConfig, GPT

if __name__ == "__main__":
    device = "cpu"
    device_type = "cpu"
    enc = tiktoken.get_encoding("gpt2")
    model = GPT(GPTConfig(vocab_size=50304))
    #checkpoint = torch.load('log/model_19072.pt')
    checkpoint = torch.load('log/model_19072.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    max_length = 64
    tokens = enc.encode("Real Madrid is the ")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(1, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    tokens = xgen[0, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"{decoded}")