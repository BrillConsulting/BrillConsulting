# Zero-Shot Object Detection

Detect ANY object using natural language descriptions - no training required! Powered by CLIP and OWL-ViT.

## 🎯 Features
- Detect with text prompts: "a red car", "person wearing sunglasses"
- No training data needed
- Open-vocabulary detection
- CLIP and OWL-ViT models

## 🚀 Usage
```bash
pip install transformers torch opencv-python pillow

python zero_shot_detector.py --image street.jpg \
    --queries "a car" "a person" "a bicycle" \
    --output result.jpg
```

## 💡 Examples
```bash
# Detect specific objects
--queries "a dog" "a cat" "a bird"

# Detect with attributes
--queries "a red car" "a person wearing a hat" "a green tree"

# Complex descriptions
--queries "a laptop on a desk" "a coffee mug" "a smartphone"
```

## 📊 Applications
Custom object detection, retail, security, robotics, accessibility

**Brill Consulting** | Zero-Shot Vision AI
