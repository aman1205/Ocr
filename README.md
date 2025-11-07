# OCR Captioning Model

## Model Architecture

The `OCRCaptioningModel` is a multi-modal neural network designed for generating captions based on images and OCR (Optical Character Recognition) text. Below is a detailed description of its components:

1. **Image Encoder (ResNet50)**:
   - A pre-trained ResNet50 model is used to extract visual features from input images.
   - The final layers of ResNet50 are removed, leaving a feature map of size `[B, C, H, W]`.

2. **Visual Projection**:
   - The extracted visual features are flattened and passed through a linear layer, followed by Layer Normalization and a GELU activation function.
   - This projects the visual features into a lower-dimensional space suitable for combining with text features.

3. **Text Encoder (T5)**:
   - A pre-trained T5 model is used to encode OCR text into a sequence of text embeddings.
   - The T5 encoder is frozen to prevent updates during training.

4. **Feature Fusion**:
   - The visual features are expanded and added to the text embeddings to create a combined feature representation.
   - This combined representation is used as input to the T5 decoder.

5. **Caption Generation**:
   - The T5 decoder generates captions based on the combined features.
   - During training, the decoder uses teacher forcing with input captions. During inference, it generates captions autoregressively.

6. **Loss Function**:
   - Cross-entropy loss is used, ignoring padding tokens.

## Model Architecture Diagram

![Model Architecture](model_architecture.png)

The diagram above illustrates the flow of data through the `OCRCaptioningModel`, highlighting the interaction between the image encoder, text encoder, and decoder.
