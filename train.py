import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from models import OCRCaptioningModel
from dataset import TextCapsDataset
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')
        print(f"Available GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Hyperparameters
        self.batch_size = 8
        self.epochs = 35
        self.lr = 3e-5
        self.patience = 7
        
        self._setup_datasets()
        self._setup_model()
        self._setup_optimization()

    def _setup_datasets(self):
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = TextCapsDataset(
            images_dir="data/images/train",
            annotations_file="data/annotations/train.json",
            ocr_tokens_file="data/rosetta_ocr_tokens/train_ocr.json",
            transform=self.train_transform
        )
        
        self.val_dataset = TextCapsDataset(
            images_dir="data/images/val",
            annotations_file="data/annotations/val.json",
            ocr_tokens_file="data/rosetta_ocr_tokens/val_ocr.json",
            transform=self.val_transform
        )

    def _setup_model(self):
        self.model = OCRCaptioningModel(tokenizer=self.train_dataset.tokenizer).to(self.device)
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_token_id)

    def _setup_optimization(self):
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=0.01
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

    def train_epoch(self, epoch):
        self.model.train()
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=TextCapsDataset.collate_fn
        )
        
        total_loss = 0
        pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}")
        
        for batch in pbar:
            if batch is None: continue
            
            images = batch['images'].to(self.device, non_blocking=True)
            ocr_input_ids = batch['ocr_input_ids'].to(self.device)
            ocr_attention_mask = batch['ocr_attention_mask'].to(self.device)
            caption_input_ids = batch['caption_input_ids'].to(self.device)
            caption_attention_mask = batch['caption_attention_mask'].to(self.device)
            
            with autocast(enabled=self.device.type == 'cuda'):
                outputs = self.model(
                    images=images,
                    ocr_input_ids=ocr_input_ids,
                    ocr_attention_mask=ocr_attention_mask,
                    decoder_input_ids=caption_input_ids[:, :-1],
                    decoder_attention_mask=caption_attention_mask[:, :-1]
                )
                
                loss = self.criterion(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    caption_input_ids[:, 1:].reshape(-1)
                )
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(loader)

    def validate(self):
        self.model.eval()
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=TextCapsDataset.collate_fn
        )
        
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                if batch is None: continue
                
                images = batch['images'].to(self.device)
                ocr_input_ids = batch['ocr_input_ids'].to(self.device)
                ocr_attention_mask = batch['ocr_attention_mask'].to(self.device)
                caption_input_ids = batch['caption_input_ids'].to(self.device)
                caption_attention_mask = batch['caption_attention_mask'].to(self.device)
                
                outputs = self.model(
                    images=images,
                    ocr_input_ids=ocr_input_ids,
                    ocr_attention_mask=ocr_attention_mask,
                    decoder_input_ids=caption_input_ids[:, :-1],
                    decoder_attention_mask=caption_attention_mask[:, :-1]
                )
                
                loss = self.criterion(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    caption_input_ids[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
        
        return total_loss / len(loader)

    def train(self):
        os.makedirs("checkpoints", exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, "checkpoints/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                }, f"checkpoints/epoch_{epoch}.pt")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()