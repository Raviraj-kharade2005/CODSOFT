import torch
from torch.utils.data import DataLoader
from model import EncoderCNN, DecoderTransformer
from utils import CaptionDataset, Vocabulary
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

vocab = Vocabulary(freq_threshold=1)
dataset = CaptionDataset("captions_dataset/images", "captions_dataset/captions.txt", vocab, transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: zip(*x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderCNN(embed_size=256).to(device)
decoder = DecoderTransformer(embed_size=256, vocab_size=len(vocab), num_heads=4, hidden_size=512, num_layers=2).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

for epoch in range(5):
    for imgs, caps in loader:
        imgs = torch.stack(imgs).to(device)
        caps = [cap.to(device) for cap in caps]
        caps = nn.utils.rnn.pad_sequence(caps, batch_first=True, padding_value=0)

        tgt_input = caps[:, :-1]
        tgt_output = caps[:, 1:]

        features = encoder(imgs)
        outputs = decoder(tgt_input, features)

        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt_output.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")
