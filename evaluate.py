#evaluate.py
import torch
from torch.utils.data import DataLoader
from src.dataset import TextCapsDataset
from model import OCRCaptioningModel
import torchvision.transforms as transforms

# Load Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OCRCaptioningModel().to(DEVICE)
model.load_state_dict(torch.load("ocr_captioning_model.pth", map_location=DEVICE))
model.eval()

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = TextCapsDataset(
    images_dir="path/to/test_images",
    annotations_file="path/to/test_annotations.json",
    ocr_tokens_file="path/to/test_ocr_tokens.json",
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate Model
for images, ocr_texts, captions in test_loader:
    images = images.to(DEVICE)
    
    with torch.no_grad():
        generated_caption = model(images, ocr_texts)
    
    print(f"Generated Caption: {generated_caption}")
    print(f"Ground Truth: {captions}")