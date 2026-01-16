import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import easyocr
import re
from spellchecker import SpellChecker  # Now works with pyspellchecker
from models import OCRCaptioningModel
from transformers import T5Tokenizer
import argparse
import warnings

warnings.filterwarnings("ignore")

class CaptionGenerator:
    def __init__(self, model_path="best_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spell = SpellChecker()
        
        # Initialize model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = OCRCaptioningModel(tokenizer=self.tokenizer).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        # Configure OCR
        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def _preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = ImageEnhance.Contrast(img).enhance(3.0)
        return img.convert('RGB')

    def _clean_ocr(self, text):
        words = [self.spell.correction(word) for word in text.split()]
        return ' '.join([w for w in words if w and len(w) > 2])

    def extract_text(self, image_path):
        try:
            img = np.array(self._preprocess_image(image_path))
            results = self.ocr_reader.readtext(img)
            return self._clean_ocr(' '.join([res[1] for res in results]))
        except:
            return ""

    def generate_caption(self, image_path):
        ocr_text = self.extract_text(image_path)
        inputs = self.tokenizer(
            f"Image shows: {ocr_text}"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs, max_length=60)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).capitalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", default="best_model.pt")
    args = parser.parse_args()
    
    generator = CaptionGenerator(args.model_path)
    print(f"Caption: {generator.generate_caption(args.image_path)}")
