import torch
import torch.nn as nn
import torchvision.models as models
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class OCRCaptioningModel(nn.Module):
    def __init__(self, hidden_dim=512, freeze_backbone=True, tokenizer=None):
        super(OCRCaptioningModel, self).__init__()
        
        # Image Encoder
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection layers
        self.visual_proj = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.GELU()
        )
        self.encoder_proj = nn.Linear(hidden_dim, 512)  # Match T5's d_model
        
        # Text components
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained("t5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        # Freeze components
        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.t5_model.encoder.parameters():
                param.requires_grad = False

    def forward(self, images, ocr_input_ids, ocr_attention_mask, decoder_input_ids=None, decoder_attention_mask=None):
        # Process visual features
        vis_features = self.pool(self.cnn(images)).flatten(1)
        vis_features = self.visual_proj(vis_features)
        vis_features = self.encoder_proj(vis_features)
        
        
        # Combine features
        combined_features = encoder_outputs.last_hidden_state + vis_features.unsqueeze(1)
        
        # Generate captions
        decoder_outputs = self.t5_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=combined_features),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True
        )
        
        return decoder_outputs
