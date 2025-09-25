import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def make_classifier(input_dim: int,
                    hidden: int = 128,
                    dropout: float = 0.2,
                    lr: float = 1e-3,
                    n_classes: int = 3) -> keras.Model:
    """
    Simple MLP for tabular features. Uses softmax for 3-class direction.
    """
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(hidden, activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(hidden//2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
