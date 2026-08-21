from __future__ import annotations

LOOKBACK = 60
HORIZON = 14


def build_model():
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((LOOKBACK, 1)),
            tf.keras.layers.Conv1D(32, 5, padding="causal", activation="relu"),
            tf.keras.layers.LSTM(48, dropout=0.15),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(HORIZON),
        ],
        name="energy_forecaster",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mae",
        metrics=["mae"],
    )
    return model


def callbacks():
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-5,
        ),
    ]
