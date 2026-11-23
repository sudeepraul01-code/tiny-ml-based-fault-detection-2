"""Small three-convolutional-layer 1D-CNN for the engineered sensor features."""

import tensorflow as tf

from data_sim.generator import CLASS_NAMES
from features.extract import FEATURE_COUNT


def build_classifier() -> tf.keras.Model:
    """Build the five-state classifier with an Adam optimizer."""
    inputs = tf.keras.Input(shape=(FEATURE_COUNT, 1), name="sensor_features")
    x = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(24, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(24, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="health_state")(x)
    model = tf.keras.Model(inputs, outputs, name="industrial_fault_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
