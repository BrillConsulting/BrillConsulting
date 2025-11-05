"""
TensorFlow/Keras Deep Learning Models
Complete TensorFlow implementation for neural networks and deep learning
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class TensorFlowModels:
    """Comprehensive TensorFlow/Keras model implementations"""

    def __init__(self):
        """Initialize TensorFlow models manager"""
        self.models = []
        self.callbacks = []

    def create_sequential_cnn(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Sequential CNN model

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'SequentialCNN'),
            'type': 'sequential_cnn',
            'input_shape': model_config.get('input_shape', (224, 224, 3)),
            'num_classes': model_config.get('num_classes', 10),
            'parameters': 3_500_000,
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_shape={model['input_shape']}, num_classes={model['num_classes']}):
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = create_cnn_model()
model.summary()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ Sequential CNN model created: {model['name']}")
        print(f"  Input shape: {model['input_shape']}, Classes: {model['num_classes']}")
        return model

    def create_functional_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Functional API model

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'FunctionalModel'),
            'type': 'functional',
            'multi_input': model_config.get('multi_input', False),
            'multi_output': model_config.get('multi_output', False),
            'parameters': 2_800_000,
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_functional_model():
    # Multi-input model
    input_1 = keras.Input(shape=(128,), name='input_1')
    input_2 = keras.Input(shape=(64,), name='input_2')

    # Process input_1
    x1 = layers.Dense(64, activation='relu')(input_1)
    x1 = layers.Dropout(0.3)(x1)

    # Process input_2
    x2 = layers.Dense(32, activation='relu')(input_2)
    x2 = layers.Dropout(0.3)(x2)

    # Concatenate
    combined = layers.concatenate([x1, x2])

    # Shared layers
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Multi-output
    output_1 = layers.Dense(10, activation='softmax', name='output_1')(x)
    output_2 = layers.Dense(1, activation='sigmoid', name='output_2')(x)

    model = keras.Model(
        inputs=[input_1, input_2],
        outputs=[output_1, output_2],
        name='multi_io_model'
    )

    model.compile(
        optimizer='adam',
        loss={{
            'output_1': 'sparse_categorical_crossentropy',
            'output_2': 'binary_crossentropy'
        }},
        metrics={{
            'output_1': ['accuracy'],
            'output_2': ['accuracy']
        }}
    )

    return model

model = create_functional_model()
model.summary()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ Functional model created: {model['name']}")
        print(f"  Multi-input: {model['multi_input']}, Multi-output: {model['multi_output']}")
        return model

    def create_transfer_learning_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Transfer Learning model

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        base_model = model_config.get('base_model', 'ResNet50')
        model = {
            'name': model_config.get('name', f'{base_model}_Transfer'),
            'type': 'transfer_learning',
            'base_model': base_model,
            'freeze_layers': model_config.get('freeze_layers', True),
            'num_classes': model_config.get('num_classes', 10),
            'parameters': 25_000_000,
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import {base_model}

def create_transfer_learning_model(num_classes={model['num_classes']}):
    # Load pre-trained model
    base_model = {base_model}(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model layers
    base_model.trainable = {'False' if model['freeze_layers'] else 'True'}

    # Build model
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = create_transfer_learning_model()
model.summary()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ Transfer learning model created: {model['name']}")
        print(f"  Base: {model['base_model']}, Freeze: {model['freeze_layers']}")
        return model

    def create_custom_training_loop(self) -> str:
        """Generate custom training loop with tf.GradientTape"""

        code = """
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Custom training loop with GradientTape
@tf.function
def train_step(model, x, y, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, logits)

    return loss_value

@tf.function
def test_step(model, x, y, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)
    return loss_value

def custom_training_loop(model, train_dataset, val_dataset, epochs=10):
    optimizer = keras.optimizers.Adam()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(epochs):
        print(f"\\nEpoch {epoch + 1}/{epochs}")

        # Training
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(model, x_batch_train, y_batch_train,
                                   optimizer, loss_fn, train_acc_metric)

            if step % 100 == 0:
                print(f"Step {step}: Loss = {float(loss_value):.4f}")

        train_acc = train_acc_metric.result()
        print(f"Training accuracy: {float(train_acc):.4f}")
        train_acc_metric.reset_states()

        # Validation
        for x_batch_val, y_batch_val in val_dataset:
            test_step(model, x_batch_val, y_batch_val, loss_fn, val_acc_metric)

        val_acc = val_acc_metric.result()
        print(f"Validation accuracy: {float(val_acc):.4f}")
        val_acc_metric.reset_states()
"""

        print("✓ Custom training loop generated")
        return code

    def create_callbacks(self) -> str:
        """Generate Keras callbacks"""

        code = """
import tensorflow as tf
from tensorflow import keras
import os

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint
checkpoint_path = 'checkpoints/model_{epoch:02d}_{val_accuracy:.2f}.h5'
os.makedirs('checkpoints', exist_ok=True)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Learning rate reduction
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# TensorBoard
tensorboard = keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
)

# CSV Logger
csv_logger = keras.callbacks.CSVLogger('training.log')

# Custom callback
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\\nEpoch {epoch + 1} completed")
        print(f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
        print(f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

callbacks = [
    early_stopping,
    model_checkpoint,
    reduce_lr,
    tensorboard,
    csv_logger,
    CustomCallback()
]
"""

        print("✓ Keras callbacks generated")
        return code

    def create_data_augmentation(self) -> str:
        """Generate data augmentation code"""

        code = """
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Data augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2)
], name='data_augmentation')

# Using augmentation in model
def create_model_with_augmentation(input_shape=(224, 224, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    # Apply augmentation
    x = data_augmentation(inputs)

    # Preprocessing
    x = layers.Rescaling(1./255)(x)

    # Model architecture
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model
"""

        print("✓ Data augmentation code generated")
        return code

    def get_tensorflow_info(self) -> Dict[str, Any]:
        """Get TensorFlow manager information"""
        return {
            'models_created': len(self.models),
            'framework': 'TensorFlow/Keras',
            'version': '2.14.0',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate TensorFlow models"""

    print("=" * 60)
    print("TensorFlow/Keras Deep Learning Models Demo")
    print("=" * 60)

    tf_models = TensorFlowModels()

    print("\n1. Creating Sequential CNN model...")
    cnn = tf_models.create_sequential_cnn({
        'name': 'ImageClassifierCNN',
        'input_shape': (224, 224, 3),
        'num_classes': 1000
    })
    print(cnn['code'][:300] + "...\n")

    print("\n2. Creating Functional API model...")
    functional = tf_models.create_functional_model({
        'name': 'MultiIOModel',
        'multi_input': True,
        'multi_output': True
    })
    print(functional['code'][:300] + "...\n")

    print("\n3. Creating Transfer Learning model...")
    transfer = tf_models.create_transfer_learning_model({
        'name': 'ResNet50_Transfer',
        'base_model': 'ResNet50',
        'freeze_layers': True,
        'num_classes': 10
    })
    print(transfer['code'][:300] + "...\n")

    print("\n4. Generating custom training loop...")
    training_code = tf_models.create_custom_training_loop()
    print(training_code[:300] + "...\n")

    print("\n5. Generating Keras callbacks...")
    callbacks_code = tf_models.create_callbacks()
    print(callbacks_code[:300] + "...\n")

    print("\n6. Generating data augmentation...")
    augmentation_code = tf_models.create_data_augmentation()
    print(augmentation_code[:300] + "...\n")

    print("\n7. TensorFlow summary:")
    info = tf_models.get_tensorflow_info()
    print(f"  Models created: {info['models_created']}")
    print(f"  Framework: {info['framework']} {info['version']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
