import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config import LR, EPOCHS, BATCH_SIZE, PATIENCE

def compile_model(model: tf.keras.Model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                 tf.keras.metrics.AUC(name="auc", curve="ROC")]
    )
    return model

def train_model(model: tf.keras.Model,
                X_train, y_train,
                X_val, y_val):
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, PATIENCE//3), min_lr=1e-6)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rl],
        verbose=2
    )
    return hist
