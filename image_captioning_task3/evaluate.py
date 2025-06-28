from model import EncoderCNN, DecoderTransformer
from utils import Vocabulary
from PIL import Image
import torchvision.transforms as transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

vocab = Vocabulary(freq_threshold=1)
vocab.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "a", 5: "man", 6: "on", 7: "bike"}
vocab.stoi = {v: k for k, v in vocab.itos.items()}

encoder = EncoderCNN(embed_size=256).to(device)
decoder = DecoderTransformer(embed_size=256, vocab_size=len(vocab), num_heads=4, hidden_size=512, num_layers=2).to(device)

encoder.eval()
decoder.eval()

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    features = encoder(image)

    caption = [vocab.stoi["<SOS>"]]
    for _ in range(20):
        cap_tensor = torch.tensor(caption).unsqueeze(0).to(device)
        output = decoder(cap_tensor, features)
        predicted = output.argmax(2)[:, -1].item()
        caption.append(predicted)
        if predicted == vocab.stoi["<EOS>"]:
            break

    return " ".join([vocab.itos[i] for i in caption[1:-1]])

print(generate_caption("captions_dataset/images/sample1.jpg"))