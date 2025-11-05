"""
FastAI Deep Learning Models
High-level deep learning with FastAI library
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class FastAIModels:
    """Comprehensive FastAI model implementations"""

    def __init__(self):
        """Initialize FastAI models manager"""
        self.models = []
        self.learners = []

    def create_vision_learner(self, learner_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Vision Learner for image classification

        Args:
            learner_config: Learner configuration

        Returns:
            Learner details
        """
        learner = {
            'name': learner_config.get('name', 'VisionLearner'),
            'task': 'image_classification',
            'architecture': learner_config.get('architecture', 'resnet34'),
            'pretrained': learner_config.get('pretrained', True),
            'metrics': learner_config.get('metrics', ['accuracy', 'error_rate']),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from fastai.vision.all import *
import torch

# Create DataLoaders
path = Path('data/images')
dls = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='valid',
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(size=224, min_scale=0.75)
)

# Create Vision Learner
learn = vision_learner(
    dls,
    {learner['architecture']},
    metrics={learner['metrics']},
    pretrained={learner['pretrained']}
)

# Find optimal learning rate
learn.lr_find()

# Train with fine-tuning
learn.fine_tune(
    epochs=5,
    base_lr=1e-3,
    freeze_epochs=1
)

# Make predictions
img = PILImage.create('test.jpg')
pred_class, pred_idx, probs = learn.predict(img)
print(f'Predicted: {{pred_class}}, Probability: {{probs[pred_idx]:.4f}}')

# Export model
learn.export('model.pkl')
"""

        learner['code'] = code
        self.learners.append(learner)

        print(f"✓ Vision learner created: {learner['name']}")
        print(f"  Architecture: {learner['architecture']}, Pretrained: {learner['pretrained']}")
        return learner

    def create_text_learner(self, learner_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Text Learner for NLP tasks

        Args:
            learner_config: Learner configuration

        Returns:
            Learner details
        """
        learner = {
            'name': learner_config.get('name', 'TextLearner'),
            'task': 'text_classification',
            'architecture': learner_config.get('architecture', 'AWD_LSTM'),
            'vocab_size': learner_config.get('vocab_size', 60000),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from fastai.text.all import *

# Create DataLoaders
path = Path('data/text')
dls = TextDataLoaders.from_folder(
    path,
    train='train',
    valid='valid',
    text_vocab=None,
    bs=64
)

# Create Text Learner
learn = text_classifier_learner(
    dls,
    {learner['architecture']},
    metrics=accuracy,
    drop_mult=0.5
)

# Train language model first (optional but recommended)
learn_lm = language_model_learner(
    dls,
    {learner['architecture']},
    drop_mult=0.3
)
learn_lm.fine_tune(4, 1e-2)

# Fine-tune classifier
learn.fine_tune(4, 1e-2)

# Make predictions
text = "This is a test sentence"
pred_class, pred_idx, probs = learn.predict(text)
print(f'Predicted: {{pred_class}}, Probability: {{probs[pred_idx]:.4f}}')

# Export model
learn.export('text_model.pkl')
"""

        learner['code'] = code
        self.learners.append(learner)

        print(f"✓ Text learner created: {learner['name']}")
        print(f"  Architecture: {learner['architecture']}")
        return learner

    def create_tabular_learner(self, learner_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Tabular Learner for structured data

        Args:
            learner_config: Learner configuration

        Returns:
            Learner details
        """
        learner = {
            'name': learner_config.get('name', 'TabularLearner'),
            'task': 'tabular_prediction',
            'layers': learner_config.get('layers', [200, 100]),
            'cat_names': learner_config.get('cat_names', []),
            'cont_names': learner_config.get('cont_names', []),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from fastai.tabular.all import *
import pandas as pd

# Load data
df = pd.read_csv('data/tabular_data.csv')

# Define categorical and continuous variables
cat_names = ['category1', 'category2', 'category3']
cont_names = ['value1', 'value2', 'value3', 'value4']
procs = [Categorify, FillMissing, Normalize]

# Create DataLoaders
splits = RandomSplitter(valid_pct=0.2)(range_of(df))
to = TabularPandas(
    df,
    procs=procs,
    cat_names=cat_names,
    cont_names=cont_names,
    y_names='target',
    splits=splits
)

dls = to.dataloaders(bs=64)

# Create Tabular Learner
learn = tabular_learner(
    dls,
    layers={learner['layers']},
    metrics=accuracy
)

# Train
learn.fit_one_cycle(5, 1e-2)

# Make predictions
row = df.iloc[0]
pred, pred_idx, probs = learn.predict(row)
print(f'Predicted: {{pred}}, Probability: {{probs[pred_idx]:.4f}}')

# Feature importance
fi = learn.feature_importance()
fi.plot_importance(max_features=10)
"""

        learner['code'] = code
        self.learners.append(learner)

        print(f"✓ Tabular learner created: {learner['name']}")
        print(f"  Layers: {learner['layers']}")
        return learner

    def create_collab_learner(self, learner_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Collaborative Filtering Learner for recommendations

        Args:
            learner_config: Learner configuration

        Returns:
            Learner details
        """
        learner = {
            'name': learner_config.get('name', 'CollabLearner'),
            'task': 'collaborative_filtering',
            'n_factors': learner_config.get('n_factors', 50),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from fastai.collab import *
import pandas as pd

# Load ratings data
ratings = pd.read_csv('data/ratings.csv')

# Create DataLoaders
dls = CollabDataLoaders.from_df(
    ratings,
    user_name='user_id',
    item_name='item_id',
    rating_name='rating',
    bs=64
)

# Create Collaborative Filtering Learner
learn = collab_learner(
    dls,
    n_factors={learner['n_factors']},
    y_range=(0.5, 5.5)
)

# Train
learn.fit_one_cycle(5, 5e-3)

# Make predictions
user_id = 123
item_id = 456
predicted_rating = learn.predict(user_id, item_id)
print(f'Predicted rating: {{predicted_rating:.2f}}')

# Get recommendations for user
recommendations = learn.get_preds(dl=dls.valid)
"""

        learner['code'] = code
        self.learners.append(learner)

        print(f"✓ Collaborative filtering learner created: {learner['name']}")
        print(f"  Factors: {learner['n_factors']}")
        return learner

    def create_training_techniques(self) -> str:
        """Generate advanced FastAI training techniques"""

        code = """
from fastai.vision.all import *

# 1. Learning Rate Finder
learn.lr_find()

# 2. Progressive Resizing
# Train at small size first, then larger
learn = vision_learner(dls_small, resnet34, metrics=accuracy)
learn.fine_tune(3)

# Then train at larger size
learn = vision_learner(dls_large, resnet34, metrics=accuracy)
learn.fine_tune(3)

# 3. Discriminative Learning Rates
learn.fit_one_cycle(5, lr_max=slice(1e-5, 1e-3))

# 4. Mixup Data Augmentation
learn = vision_learner(dls, resnet34, metrics=accuracy, cbs=MixUp())
learn.fit_one_cycle(5)

# 5. Gradient Accumulation
learn = vision_learner(dls, resnet34, metrics=accuracy)
with learn.no_bar():
    learn.fit(5, cbs=GradientAccumulation(n_acc=8))

# 6. Mixed Precision Training
learn = vision_learner(dls, resnet34, metrics=accuracy).to_fp16()
learn.fit_one_cycle(5)

# 7. Model Interpretation
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9)

# 8. Learning Rate Annealing
learn.fit(5, lr=1e-3, wd=1e-4, cbs=ReduceLROnPlateau(monitor='valid_loss', patience=2))
"""

        print("✓ Training techniques code generated")
        return code

    def get_fastai_info(self) -> Dict[str, Any]:
        """Get FastAI manager information"""
        return {
            'learners_created': len(self.learners),
            'framework': 'FastAI',
            'version': '2.7.0',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate FastAI models"""

    print("=" * 60)
    print("FastAI Deep Learning Models Demo")
    print("=" * 60)

    fastai = FastAIModels()

    print("\n1. Creating Vision Learner...")
    vision = fastai.create_vision_learner({
        'name': 'ImageClassifier',
        'architecture': 'resnet50',
        'pretrained': True
    })
    print(vision['code'][:300] + "...\n")

    print("\n2. Creating Text Learner...")
    text = fastai.create_text_learner({
        'name': 'SentimentClassifier',
        'architecture': 'AWD_LSTM'
    })
    print(text['code'][:300] + "...\n")

    print("\n3. Creating Tabular Learner...")
    tabular = fastai.create_tabular_learner({
        'name': 'TabularPredictor',
        'layers': [200, 100, 50]
    })
    print(tabular['code'][:300] + "...\n")

    print("\n4. Creating Collaborative Filtering Learner...")
    collab = fastai.create_collab_learner({
        'name': 'RecommenderSystem',
        'n_factors': 50
    })
    print(collab['code'][:300] + "...\n")

    print("\n5. Generating training techniques...")
    techniques = fastai.create_training_techniques()
    print(techniques[:300] + "...\n")

    print("\n6. FastAI summary:")
    info = fastai.get_fastai_info()
    print(f"  Learners created: {info['learners_created']}")
    print(f"  Framework: {info['framework']} {info['version']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
