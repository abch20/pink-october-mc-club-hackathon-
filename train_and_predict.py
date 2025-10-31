"""
Train a breast cancer image classifier (malignant vs normal) and produce a submission CSV.

Usage examples:
  python train_and_predict.py --epochs 8 --batch_size 16
  python train_and_predict.py --mode predict_only --weights best_model.h5

The script will:
 - build a transfer-learning model (MobileNetV2 by default)
 - use ImageDataGenerator with augmentation and validation split
 - train with EarlyStopping and ModelCheckpoint
 - load best weights and predict on images in the `test` folder
 - write `submission.csv` with columns: image file,label (M or N)

Notes:
 - For robustness: heavy data augmentation, class weighting, and fine-tuning are included.
 - If you have a GPU and TensorFlow installed, training will be much faster.
"""

import argparse
import os

import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(input_shape=(224,224,3), base_trainable=False, lr=1e-4):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
    x = base.output 
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def make_generators(train_dir, img_size=(224,224), batch_size=16, seed=42):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=seed
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=seed
    )

    return train_gen, val_gen


def predict_to_submission(model, test_dir, out_csv='submission.csv', img_size=(224,224)):
    # Build list of test files (sorted to keep deterministic order)
    files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    results = []

    for fname in files:
        img_path = os.path.join(test_dir, fname)
        img = image.load_img(img_path, target_size=img_size)
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        prob = float(model.predict(arr, verbose=0)[0][0])
        label = 'M' if prob >= 0.5 else 'N'
        results.append({'image file': fname, 'label': label})

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f'Wrote predictions for {len(df)} images to {out_csv}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='train', help='training folder with subfolders per class')
    parser.add_argument('--test_dir', default='test', help='test folder with images to predict')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--mode', choices=['train_and_predict', 'predict_only'], default='train_and_predict')
    parser.add_argument('--weights', default='best_model.keras', help='weights file to load for predict_only mode')
    parser.add_argument('--quick', action='store_true', help='quick mode for smoke test (few steps)')
    args = parser.parse_args()

    train_dir = args.train_dir
    test_dir = args.test_dir
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size

    if args.mode == 'predict_only':
        from tensorflow.keras.models import load_model
        model = load_model(args.weights)
        predict_to_submission(model, test_dir)
        return

    # train_and_predict
    train_gen, val_gen = make_generators(train_dir, img_size=img_size, batch_size=batch_size)

    # quick-mode adjustments
    epochs = args.epochs
    if args.quick:
        epochs = 2

    model = build_model(input_shape=(img_size[0], img_size[1], 3), base_trainable=False, lr=1e-4)
    print(model.summary())

    # For simplicity in this script's smoke tests we don't pass class_weight to fit.
    # If you want to enable it, uncomment the block below and ensure sklearn is
    # available and the generator returns (x, y) pairs only.
    class_weight = None

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    ]

    """Train and predict pipeline with improved data pipeline and training.

    Improvements over the previous script:
    - Use tf.data image_dataset_from_directory for stable pipelines
    - On-model augmentation layers (tf.keras.layers) for deterministic TTA support
    - Option to use EfficientNetB0 or MobileNetV2
    - Class weighting computed from training labels
    - ReduceLROnPlateau and EarlyStopping callbacks
    - Controlled fine-tuning (unfreeze last blocks)

    Run a quick smoke test with --quick to validate.
    """

    import argparse
    import os

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                            ReduceLROnPlateau)
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image


    def build_model(backbone_name='mobilenetv2', input_shape=(224, 224, 3), base_trainable=False, lr=1e-4):
        """Build a transfer-learning model with an on-model augmentation block.

        Returns a compiled model.
        """
        if backbone_name.lower() == 'efficientnetb0':
            base = keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            base = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

        base.trainable = base_trainable

        inputs = layers.Input(shape=input_shape)
        x = inputs
        # simple rescaling (image_dataset_from_directory yields uint8)
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255.0)(x)
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def make_datasets(train_dir, img_size=(224, 224), batch_size=16, val_split=0.2, seed=42):
        """Create tf.data training and validation datasets from a directory.

        Uses image_dataset_from_directory which returns (images, labels) tuples.
        """
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='binary',
            batch_size=batch_size,
            image_size=img_size,
            shuffle=True,
            seed=seed,
            validation_split=val_split,
            subset='training',
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='binary',
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False,
            seed=seed,
            validation_split=val_split,
            subset='validation',
        )

        # on-the-fly data augmentation (integrated into the tf.data pipeline)
        data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip('horizontal'),
            layers.experimental.preprocessing.RandomRotation(0.08),
            layers.experimental.preprocessing.RandomZoom(0.08),
            layers.experimental.preprocessing.RandomContrast(0.08),
        ], name='data_augmentation')

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.cache().prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)

        return train_ds, val_ds


    def predict_to_submission(model, test_dir, out_csv='submission.csv', img_size=(224, 224)):
        files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        results = []
        for fname in files:
            img_path = os.path.join(test_dir, fname)
            img = image.load_img(img_path, target_size=img_size)
            arr = image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, 0)
            prob = float(model.predict(arr, verbose=0)[0][0])
            label = 'M' if prob >= 0.5 else 'N'
            results.append({'image file': fname, 'label': label})
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print(f'Wrote predictions for {len(df)} images to {out_csv}')


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_dir', default='train', help='training folder with subfolders per class')
        parser.add_argument('--test_dir', default='test', help='test folder with images to predict')
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--epochs', type=int, default=8)
        parser.add_argument('--mode', choices=['train_and_predict', 'predict_only'], default='train_and_predict')
        parser.add_argument('--weights', default='best_model.keras', help='weights file to load for predict_only mode')
        parser.add_argument('--quick', action='store_true', help='quick mode for smoke test (few steps)')
        parser.add_argument('--model', choices=['mobilenetv2', 'efficientnetb0'], default='mobilenetv2', help='backbone model')
        args = parser.parse_args()

        train_dir = args.train_dir
        test_dir = args.test_dir
        img_size = (args.img_size, args.img_size)
        batch_size = args.batch_size

        if args.mode == 'predict_only':
            from tensorflow.keras.models import load_model
            model = load_model(args.weights)
            predict_to_submission(model, test_dir, img_size=img_size)
            return

        # train_and_predict
        train_ds, val_ds = make_datasets(train_dir, img_size=img_size, batch_size=batch_size)

        # quick-mode adjustments
        epochs = args.epochs
        if args.quick:
            epochs = 2

        model = build_model(backbone_name=args.model, input_shape=(img_size[0], img_size[1], 3), base_trainable=False, lr=1e-4)
        model.summary()

        # Compute class weights from files to address imbalance
        labels = []
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        for idx, cname in enumerate(class_names):
            cpath = os.path.join(train_dir, cname)
            for _f in os.listdir(cpath):
                if _f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    labels.append(idx)
        if len(labels) > 0:
            cw_vals = compute_class_weight('balanced', classes=np.unique(labels), y=np.array(labels))
            class_weight = {i: float(v) for i, v in enumerate(cw_vals)}
            print('Computed class_weight:', class_weight)
        else:
            class_weight = None

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ]

        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            class_weight=class_weight,
        )

        # Fine-tuning: unfreeze most of the base and train a few more epochs
        try:
            for layer in model.layers:
                if hasattr(layer, 'trainable'):
                    layer.trainable = True
            model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
            ft_epochs = 3 if not args.quick else 1
            model.fit(train_ds, epochs=ft_epochs, validation_data=val_ds, callbacks=callbacks, class_weight=class_weight)
        except Exception as e:
            print('Fine-tuning skipped:', e)

        # load best model and predict
        from tensorflow.keras.models import load_model
        best = load_model('best_model.keras')
        predict_to_submission(best, test_dir, img_size=img_size)


    if __name__ == '__main__':
        main()


