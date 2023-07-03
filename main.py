import tiktoken
import torch
from model import Transformer
#device = torch.device("mps")
enc = tiktoken.encoding_for_model("gpt-4")


encodings = torch.tensor(enc.encode("hello my name is kittu what is your name and will it be amazing")).unsqueeze(0)  # Batch size 1

print(encodings.shape)
transformer = Transformer(num_heads=8, d_model=512, d_ff=2048, vocab_size=enc.n_vocab)

output = transformer(encodings)

probabilities = torch.softmax(output, dim=-1)
first_sequence_probs = probabilities[0, 0, :]
print(output.shape)
#print(torch.argmax(first_sequence_probs).item())
print(enc.decode([torch.argmax(first_sequence_probs).item()]))
