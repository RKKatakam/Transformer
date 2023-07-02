import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim

from model import Transformer

enc = tiktoken.encoding_for_model("gpt-4")
epochs = 10
batch_size = 32
learning_rate = 0.001
num_heads = 8
d_model = 512
d_ff = 2048
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = open("Book 1 - The Philosopher's Stone.txt", "r", encoding="utf-8")

model = Transformer(num_heads=num_heads, d_model=d_model, d_ff=d_ff, vocab_size=enc.n_vocab).to(device)

harry_potter_encodings = torch.tensor(enc.encode(file.read())).unsqueeze(0)

criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for i in range(800):
        optimizer.zero_grad()
        outputs = model(harry_potter_encodings[i])
        torch.argmax(first_sequence_probs).item()
