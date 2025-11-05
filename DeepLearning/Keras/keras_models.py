"""
Keras High-Level Neural Networks API
Author: BrillConsulting
Description: Simple and powerful deep learning with Keras API
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class KerasModelBuilder:
    """Keras model building and training"""

    def __init__(self):
        self.models = []

    def build_sequential_model(self, config: Dict[str, Any]) -> str:
        """Build Sequential model"""
        code = '''from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint('best_model.h5')
    ]
)
'''
        print("✓ Sequential model created")
        return code

    def build_functional_model(self, config: Dict[str, Any]) -> str:
        """Build Functional API model"""
        code = '''from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''
        print("✓ Functional API model created")
        return code


def demo():
    """Demonstrate Keras"""
    print("=" * 60)
    print("Keras High-Level Neural Networks API Demo")
    print("=" * 60)

    builder = KerasModelBuilder()

    print("\n1. Building Sequential model...")
    print(builder.build_sequential_model({})[:200] + "...")

    print("\n2. Building Functional API model...")
    print(builder.build_functional_model({})[:200] + "...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
