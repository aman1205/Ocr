import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
from transformers import T5Tokenizer

class TextCapsDataset(Dataset):
    def __init__(self, images_dir, annotations_file, ocr_tokens_file, 
                 transform=None, max_caption_length=64, max_ocr_length=128, 
                 tokenizer=None):
        self.images_dir = images_dir
        self.transform = transform
        self.max_caption_length = max_caption_length
        self.max_ocr_length = max_ocr_length
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained("t5-small")
        
        # Load and validate data
        self.annotations = self._load_json(annotations_file)
        self.ocr_data = self._load_json(ocr_tokens_file)
        self.valid_samples = self._validate_samples()
        self._remove_invalid_samples()

    def _load_json(self, file_path):
        with open(file_path) as f:
            return {item['image_id']: item for item in json.load(f)['data']}

    def _validate_samples(self):
        valid = []
        for img_id in self.annotations:
            if img_id in self.ocr_data:
                path = os.path.join(self.images_dir, f"{img_id}.jpg")
                if os.path.exists(path):
                    valid.append(img_id)
        return valid

    def _remove_invalid_samples(self):
        valid = []
        for img_id in self.valid_samples:
            try:
                Image.open(os.path.join(self.images_dir, f"{img_id}.jpg")).convert("RGB")
                valid.append(img_id)
            except:
                continue
        self.valid_samples = valid

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_id = self.valid_samples[idx]
        try:
            # Load image
            img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            # Process text
            caption = self.annotations[img_id]['caption_str']
            ocr_text = " ".join(self.ocr_data[img_id].get('ocr_tokens', [])[:self.max_ocr_length])
            
            # Tokenize
            caption_inputs = self.tokenizer(
                caption,
                max_length=self.max_caption_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            ocr_inputs = self.tokenizer(
                ocr_text,
                max_length=self.max_ocr_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                'image': image,
                'ocr_input_ids': ocr_inputs.input_ids.squeeze(),
                'ocr_attention_mask': ocr_inputs.attention_mask.squeeze(),
                'caption_input_ids': caption_inputs.input_ids.squeeze(),
                'caption_attention_mask': caption_inputs.attention_mask.squeeze()
            }
        except:
            return None

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        return {
            'images': torch.stack([b['image'] for b in batch]),
            'ocr_input_ids': torch.stack([b['ocr_input_ids'] for b in batch]),
            'ocr_attention_mask': torch.stack([b['ocr_attention_mask'] for b in batch]),
            'caption_input_ids': torch.stack([b['caption_input_ids'] for b in batch]),
            'caption_attention_mask': torch.stack([b['caption_attention_mask'] for b in batch])
        }