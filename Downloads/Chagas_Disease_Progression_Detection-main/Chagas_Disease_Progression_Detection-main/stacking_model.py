import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm_model(splits):
    X_train, y_train = splits["train"]
    X_validation, y_validation = splits["validation"]
    X_test, y_test = splits["test"]

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = {
        int(label): float(weight) for label, weight in zip(classes, class_weights)
    }
    model = build_lstm_model(tuple(X_train.shape[1:]))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_validation, y_validation),
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=2,
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print("\nTraining Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Test Loss:", test_loss)

    return model