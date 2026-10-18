"""
Deep learning helpers: the tf.data input pipeline, model builders (Sequential and
Functional API), a macro-F1 callback, and one function that runs and logs a full
Keras experiment through the shared ExperimentTracker.
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# tf.data input pipeline
# ---------------------------------------------------------------------------
def make_dataset(X, y, class_weights=None, batch_size=512, shuffle=False, seed=42):
    """Build a tf.data pipeline: (optional per-row weights) -> cache -> shuffle -> batch -> prefetch.

    Class weights are turned into per-example sample weights inside the dataset.
    This works the same way for training and validation, so the validation loss is
    on the same scale as the (weighted) training loss and the two curves can be
    compared directly.
    """
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int32")
    if class_weights is None:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        lookup = np.array([class_weights[k] for k in sorted(class_weights)], dtype="float32")
        ds = tf.data.Dataset.from_tensor_slices((X, y, lookup[y]))
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _dense_block(x_or_model, units, dropout, batch_norm, l2, activation, functional):
    reg = keras.regularizers.l2(l2) if l2 else None
    layers = [keras.layers.Dense(units, kernel_regularizer=reg, use_bias=not batch_norm)]
    if batch_norm:
        layers.append(keras.layers.BatchNormalization())
    layers.append(keras.layers.Activation(activation))
    if dropout:
        layers.append(keras.layers.Dropout(dropout))
    if functional:
        x = x_or_model
        for layer in layers:
            x = layer(x)
        return x
    for layer in layers:
        x_or_model.add(layer)
    return x_or_model


def build_mlp(input_dim, n_classes=3, hidden=(64, 32), dropout=0.0, batch_norm=False,
              l2=0.0, activation="relu", name="mlp"):
    """Feed-forward network built with the Sequential API."""
    model = keras.Sequential(name=name)
    model.add(keras.Input(shape=(input_dim,)))
    for units in hidden:
        _dense_block(model, units, dropout, batch_norm, l2, activation, functional=False)
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    return model


def build_wide_and_deep(input_dim, n_classes=3, hidden=(256, 128, 64), dropout=0.3,
                        batch_norm=True, l2=1e-4, activation="relu", name="wide_and_deep"):
    """Wide & Deep network built with the Functional API.

    The 'wide' path is a single linear layer straight from the inputs to the class
    scores, which is essentially a logistic regression. The 'deep' path is a regular
    MLP. Their outputs are added before the softmax, so the network can keep the
    strong linear effects (blood pressure, BMI, general health) while the deep path
    learns the interactions on top.
    """
    inputs = keras.Input(shape=(input_dim,), name="survey_features")
    reg = keras.regularizers.l2(l2) if l2 else None
    wide = keras.layers.Dense(n_classes, kernel_regularizer=reg, name="wide_logits")(inputs)
    x = inputs
    for units in hidden:
        x = _dense_block(x, units, dropout, batch_norm, l2, activation, functional=True)
    deep = keras.layers.Dense(n_classes, name="deep_logits")(x)
    logits = keras.layers.Add(name="combined_logits")([wide, deep])
    outputs = keras.layers.Activation("softmax", name="probabilities")(logits)
    return keras.Model(inputs, outputs, name=name)


def make_optimizer(name="adam", lr=1e-3):
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class MacroF1Callback(keras.callbacks.Callback):
    """Computes training and validation macro-F1 after every epoch.

    Keras has no built-in multi-class macro-F1, and loss alone does not tell me how
    the minority classes are doing. The values are added to the epoch logs (so
    EarlyStopping and the history can use them) and written to TensorBoard.
    """

    def __init__(self, X_val, y_val, X_train_sample, y_train_sample, log_dir=None):
        super().__init__()
        self.X_val, self.y_val = X_val, np.asarray(y_val)
        self.X_tr, self.y_tr = X_train_sample, np.asarray(y_train_sample)
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, "macro_f1")) if log_dir else None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        val_pred = np.argmax(self.model.predict(self.X_val, batch_size=4096, verbose=0), axis=1)
        tr_pred = np.argmax(self.model.predict(self.X_tr, batch_size=4096, verbose=0), axis=1)
        logs["val_macro_f1"] = f1_score(self.y_val, val_pred, average="macro", zero_division=0)
        logs["train_macro_f1"] = f1_score(self.y_tr, tr_pred, average="macro", zero_division=0)
        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar("val_macro_f1", logs["val_macro_f1"], step=epoch)
                tf.summary.scalar("train_macro_f1", logs["train_macro_f1"], step=epoch)
                self.writer.flush()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_keras_experiment(tracker, exp_id, build_fn, X_train, y_train, X_val, y_val, *,
                         model_name, params, features="engineered", imbalance="none",
                         class_weights=None, batch_size=512, epochs=60, lr=1e-3,
                         optimizer="adam", reduce_lr=False, patience=8,
                         tb_root="logs/tensorboard", seed=42, notes="", verbose=0):
    """Build, train and log one Keras experiment.

    Early stopping watches validation macro-F1 (the project's selection metric) and
    restores the best epoch's weights.
    """
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model.compile(optimizer=make_optimizer(optimizer, lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train_ds = make_dataset(X_train, y_train, class_weights, batch_size, shuffle=True, seed=seed)
    val_ds = make_dataset(X_val, y_val, class_weights, batch_size=4096)

    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(len(X_train), size=min(30_000, len(X_train)), replace=False)
    log_dir = os.path.join(tb_root, exp_id)

    callbacks = [
        MacroF1Callback(X_val, y_val, X_train[tr_idx], np.asarray(y_train)[tr_idx], log_dir),
        keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
    ]
    if reduce_lr:
        callbacks.insert(1, keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5))

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=verbose)
    train_time = time.time() - start
    hist = history.history
    best_epoch = int(np.argmax(hist["val_macro_f1"])) + 1

    y_proba = model.predict(X_val, batch_size=4096, verbose=0)
    full_params = {**params, "optimizer": optimizer, "learning_rate": lr,
                   "batch_size": batch_size, "max_epochs": epochs, "patience": patience,
                   "reduce_lr_on_plateau": reduce_lr, "trainable_params": model.count_params()}
    metrics = tracker.log(exp_id, "Deep Learning", model_name, full_params, features, imbalance,
                          y_val, y_proba, train_time=train_time,
                          train_macro_f1=float(hist["train_macro_f1"][best_epoch - 1]), notes=notes,
                          extra={"epochs_run": len(hist["loss"]), "best_epoch": best_epoch})
    print(f"{exp_id} {model_name}: val macro-F1 {metrics['macro_f1']:.4f} | "
          f"train macro-F1 {hist['train_macro_f1'][best_epoch - 1]:.4f} | "
          f"ROC-AUC {metrics['roc_auc_macro']:.4f} | recall prediab {metrics['recall_prediab']:.3f}, "
          f"diab {metrics['recall_diab']:.3f} | best epoch {best_epoch}/{len(hist['loss'])} | "
          f"{train_time:.0f}s")
    return model, hist, metrics
