import tiktoken
import torch
from model import Transformer
#device = torch.device("mps")
enc = tiktoken.encoding_for_model("gpt-4")


encodings = torch.tensor(enc.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1

transformer = Transformer(num_heads=8, d_model=512, d_ff=2048, vocab_size=enc.n_vocab)

output = transformer(encodings)

probabilities = torch.softmax(output, dim=-1)
first_sequence_probs = probabilities[0, 0, :]
print(enc.decode([torch.argmax(first_sequence_probs).item()]))
