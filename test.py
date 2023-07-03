import torch
from torch import optim, nn
import tiktoken

from model import Transformer


def train(m, encodings, epochs):
    for epoch in range(epochs):
        for i in range(1, encodings.shape[1] - 1, 800):
            for j in range(1, 800):
                optimizer.zero_grad()
                output = m(encodings[:, i:i + j])
                probabilities = torch.softmax(output, dim=-1)
                first_sequence_probs = probabilities[0, 0, :]
                expected_output = torch.zeros(tiktoken.encoding_for_model("gpt-4").n_vocab)
                expected_output[encodings[:, i + j + 1].squeeze(0)] = 1

                loss = criterion(first_sequence_probs, expected_output)
                loss.backward()
                optimizer.step()
                print(f"Loss: {loss.item():.4f}")
        print(f"Epoch: {epoch + 1} / {epochs}")


if __name__ == "__main__":
    enc = tiktoken.encoding_for_model("gpt-4")
    harry_potter_encodings = torch.tensor(enc.encode(open("Book 1 - The Philosopher's Stone.txt").read())).unsqueeze(0)

    model = Transformer(num_heads=8, d_model=512, d_ff=2048, vocab_size=enc.n_vocab)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    encodings = torch.tensor(enc.encode("harry, ron and ")).unsqueeze(
        0)  # Batch size 1

    print(encodings.shape)

    output = model(encodings)

    probabilities = torch.softmax(output, dim=-1)
    first_sequence_probs = probabilities[0, 0, :]
    print(output.shape)
    # print(torch.argmax(first_sequence_probs).item())
    print(enc.decode([torch.argmax(first_sequence_probs).item()]))

    model.train()

    train(model, harry_potter_encodings, epochs=10)

    # save model
    torch.save(model.state_dict(), "model.pth")

    encodings = torch.tensor(enc.encode("harry, ron and ")).unsqueeze(
        0)  # Batch size 1

    print(encodings.shape)

    output = model(encodings)

    probabilities = torch.softmax(output, dim=-1)
    first_sequence_probs = probabilities[0, 0, :]
    print(output.shape)
    # print(torch.argmax(first_sequence_probs).item())
    print(enc.decode([torch.argmax(first_sequence_probs).item()]))
