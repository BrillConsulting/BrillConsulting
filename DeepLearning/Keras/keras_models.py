"""
Keras High-Level Neural Networks API
Author: BrillConsulting
Description: Simple and powerful deep learning with Keras API - advanced implementations
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class KerasModelBuilder:
    """Advanced Keras model building and training"""

    def __init__(self):
        """Initialize Keras model builder"""
        self.models = []
        self.training_history = []

    def build_advanced_cnn(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build advanced CNN with residual connections

        Args:
            config: Model configuration

        Returns:
            Model details and code
        """
        print(f"\n{'='*60}")
        print("Building Advanced CNN")
        print(f"{'='*60}")

        model_config = {
            'name': config.get('name', 'AdvancedCNN'),
            'input_shape': config.get('input_shape', (224, 224, 3)),
            'num_classes': config.get('num_classes', 10),
            'filters': config.get('filters', [64, 128, 256]),
            'use_residual': config.get('use_residual', True)
        }

        code = """
from tensorflow import keras
from tensorflow.keras import layers

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1)(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def build_advanced_cnn(input_shape=(224, 224, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='AdvancedCNN')
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

model = build_advanced_cnn()
model.summary()
"""

        print(f"✓ Advanced CNN built: {model_config['name']}")
        print(f"  Input: {model_config['input_shape']}, Classes: {model_config['num_classes']}")
        print(f"  Residual connections: {model_config['use_residual']}")
        print(f"{'='*60}")

        result = {
            'config': model_config,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.models.append(result)
        return result

    def build_lstm_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build LSTM model for sequence tasks

        Args:
            config: Model configuration

        Returns:
            Model details
        """
        print(f"\n{'='*60}")
        print("Building LSTM Model")
        print(f"{'='*60}")

        model_config = {
            'name': config.get('name', 'BiLSTM'),
            'sequence_length': config.get('sequence_length', 100),
            'input_dim': config.get('input_dim', 128),
            'lstm_units': config.get('lstm_units', [256, 128]),
            'bidirectional': config.get('bidirectional', True)
        }

        code = """
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(sequence_length=100, input_dim=128, num_classes=10):
    inputs = keras.Input(shape=(sequence_length, input_dim))
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='BiLSTM')
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = build_lstm_model()
"""

        print(f"✓ LSTM Model built: {model_config['name']}")
        print(f"  Sequence length: {model_config['sequence_length']}")
        print(f"  LSTM units: {model_config['lstm_units']}")
        print(f"  Bidirectional: {model_config['bidirectional']}")
        print(f"{'='*60}")

        result = {
            'config': model_config,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.models.append(result)
        return result

    def create_custom_layer(self) -> str:
        """Generate custom layer code"""

        code = """
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class AttentionLayer(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.u = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_u'
        )
        
    def call(self, inputs):
        # Score calculation
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1),
            axis=1
        )
        
        # Apply attention
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector

# Usage in model
inputs = keras.Input(shape=(100, 128))
x = layers.LSTM(256, return_sequences=True)(inputs)
x = AttentionLayer(128)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
"""

        print("\n✓ Custom attention layer code generated")
        return code

    def create_learning_rate_scheduler(self) -> str:
        """Generate learning rate scheduler code"""

        code = """
from tensorflow import keras
import tensorflow as tf

# Cosine decay with warmup
def cosine_decay_with_warmup(global_step, learning_rate_base,
                             total_steps, warmup_steps):
    if global_step < warmup_steps:
        lr = learning_rate_base * global_step / warmup_steps
    else:
        cosine_decay = 0.5 * (1 + tf.cos(
            tf.constant(np.pi) * (global_step - warmup_steps) / 
            (total_steps - warmup_steps)
        ))
        lr = learning_rate_base * cosine_decay
    
    return lr

# Learning rate schedule callback
class WarmupCosineDecay(keras.callbacks.Callback):
    def __init__(self, learning_rate_base, total_steps, warmup_steps):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.global_step = 0
        
    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(
            self.global_step,
            self.learning_rate_base,
            self.total_steps,
            self.warmup_steps
        )
        keras.backend.set_value(self.model.optimizer.lr, lr)
        self.global_step += 1

# Cyclical learning rate
cyclic_lr = keras.callbacks.LearningRateScheduler(
    lambda epoch: 0.001 * (0.95 ** (epoch // 10))
)

# Step decay
step_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
"""

        print("\n✓ Learning rate scheduler code generated")
        return code

    def create_data_augmentation(self) -> str:
        """Generate data augmentation code"""

        code = """
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Data augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="data_augmentation")

# Advanced augmentation with MixUp
class MixupAugmentation(layers.Layer):
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def call(self, images, labels, training=None):
        if not training:
            return images, labels
            
        batch_size = tf.shape(images)[0]
        
        # Sample lambda
        lam = tf.random.uniform([batch_size, 1, 1, 1], 0, self.alpha)
        
        # Shuffle images and labels
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Mix
        mixed_images = lam * images + (1 - lam) * shuffled_images
        mixed_labels = lam[:, 0, 0, 0:1] * labels + \
                      (1 - lam[:, 0, 0, 0:1]) * shuffled_labels
        
        return mixed_images, mixed_labels

# CutMix augmentation
class CutMixAugmentation(layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def call(self, images, labels, training=None):
        if not training:
            return images, labels
            
        batch_size = tf.shape(images)[0]
        image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
        
        # Sample lambda
        lam = tf.random.uniform([], 0, self.alpha)
        
        # Random box
        cut_rat = tf.sqrt(1.0 - lam)
        cut_w = tf.cast(image_width * cut_rat, tf.int32)
        cut_h = tf.cast(image_height * cut_rat, tf.int32)
        
        cx = tf.random.uniform([], 0, image_width, dtype=tf.int32)
        cy = tf.random.uniform([], 0, image_height, dtype=tf.int32)
        
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_width)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_height)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_width)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_height)
        
        # Shuffle and mix
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Create mask and apply
        mask = tf.ones_like(images)
        mask = tf.tensor_scatter_nd_update(
            mask,
            [[i, slice(y1, y2), slice(x1, x2), slice(None)] 
             for i in range(batch_size)],
            [tf.zeros([y2-y1, x2-x1, 3])] * batch_size
        )
        
        mixed_images = images * mask + shuffled_images * (1 - mask)
        
        # Mix labels
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (image_width * image_height)
        mixed_labels = lam * labels + (1 - lam) * shuffled_labels
        
        return mixed_images, mixed_labels
"""

        print("\n✓ Data augmentation code generated")
        return code

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'models_created': len(self.models),
            'training_runs': len(self.training_history),
            'framework': 'Keras',
            'features': ['Sequential', 'Functional', 'Custom Layers', 'Callbacks'],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Keras"""
    print("=" * 60)
    print("Keras High-Level Neural Networks API Demo")
    print("=" * 60)

    builder = KerasModelBuilder()

    # Build advanced CNN
    print("\n1. Building Advanced CNN with residual connections...")
    cnn_result = builder.build_advanced_cnn({
        'input_shape': (224, 224, 3),
        'num_classes': 1000,
        'use_residual': True
    })

    # Build LSTM model
    print("\n2. Building Bidirectional LSTM...")
    lstm_result = builder.build_lstm_model({
        'sequence_length': 100,
        'input_dim': 128,
        'lstm_units': [256, 128],
        'bidirectional': True
    })

    # Custom layer
    print("\n3. Custom Attention Layer:")
    custom_code = builder.create_custom_layer()
    print(custom_code[:300] + "...\n")

    # Learning rate scheduler
    print("\n4. Learning Rate Schedulers:")
    lr_code = builder.create_learning_rate_scheduler()
    print(lr_code[:300] + "...\n")

    # Data augmentation
    print("\n5. Advanced Data Augmentation:")
    aug_code = builder.create_data_augmentation()
    print(aug_code[:300] + "...\n")

    # Manager info
    print("\n6. Manager summary:")
    info = builder.get_manager_info()
    print(f"  Models created: {info['models_created']}")
    print(f"  Framework: {info['framework']}")
    print(f"  Features: {', '.join(info['features'])}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
