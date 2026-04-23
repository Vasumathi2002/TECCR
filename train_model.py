"""
TECCR Emotion Prediction Model Training Script
Trains a BiLSTM model on Tamil emotion dataset
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = "dataset.csv"
MODEL_DIR = "models"
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128
EPOCHS = 30
BATCH_SIZE = 32

# Create models directory
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Load dataset and prepare for training"""
    print("📂 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"📊 Columns: {df.columns.tolist()}")
    
    # Display emotion distribution
    print("\n📈 Primary Emotion Distribution:")
    print(df['primary_emotion'].value_counts())
    
    return df

def prepare_text_data(df):
    """Tokenize and pad text sequences"""
    print("\n🔤 Preparing text data...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    print(f"✅ Vocabulary size: {len(tokenizer.word_index)}")
    print(f"✅ Sequence shape: {padded_sequences.shape}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"💾 Tokenizer saved to {tokenizer_path}")
    
    return padded_sequences, tokenizer

def prepare_labels(df):
    """Encode labels for multi-output model"""
    print("\n🏷️ Encoding labels...")
    
    # Encode primary emotion
    primary_encoder = LabelEncoder()
    primary_labels = primary_encoder.fit_transform(df['primary_emotion'])
    primary_labels_cat = to_categorical(primary_labels)
    
    # Encode secondary emotion
    secondary_encoder = LabelEncoder()
    secondary_labels = secondary_encoder.fit_transform(df['secondary_emotion'])
    secondary_labels_cat = to_categorical(secondary_labels)
    
    print(f"✅ Primary emotions: {primary_encoder.classes_}")
    print(f"✅ Secondary emotions: {secondary_encoder.classes_}")
    
    # Save encoders
    primary_encoder_path = os.path.join(MODEL_DIR, "primary_encoder.pkl")
    secondary_encoder_path = os.path.join(MODEL_DIR, "secondary_encoder.pkl")
    
    with open(primary_encoder_path, 'wb') as f:
        pickle.dump(primary_encoder, f)
    with open(secondary_encoder_path, 'wb') as f:
        pickle.dump(secondary_encoder, f)
    
    print(f"💾 Encoders saved")
    
    return primary_labels_cat, secondary_labels_cat, primary_encoder, secondary_encoder

def build_teccr_model(vocab_size, num_primary_classes, num_secondary_classes):
    print("\n🏗️ Building TECCR Multi-Output Model...")

    input_layer = Input(shape=(MAX_LEN,))

    x = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(input_layer)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)

    shared_dense = Dense(128, activation='relu')(x)
    shared_dense = Dropout(0.4)(shared_dense)

    # 🔹 Primary emotion output
    primary_output = Dense(
        num_primary_classes,
        activation='softmax',
        name='primary_output'
    )(shared_dense)

    # 🔹 Secondary emotion output
    secondary_output = Dense(
        num_secondary_classes,
        activation='softmax',
        name='secondary_output'
    )(shared_dense)

    model = Model(
        inputs=input_layer,
        outputs=[primary_output, secondary_output]
    )

    model.compile(
        optimizer='adam',
        loss={
            'primary_output': 'categorical_crossentropy',
            'secondary_output': 'categorical_crossentropy'
        },
        metrics={
            'primary_output': 'accuracy',
            'secondary_output': 'accuracy'
        }
    )

    model.summary()
    return model

def train_model(model, X_train, yp_train, ys_train, X_val, yp_val, ys_val):
    """Train the model with callbacks"""
    print("\n🚀 Starting training...")
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_primary_output_accuracy',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "teccr_model_best.h5"),
    monitor='val_primary_output_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

    
    # Train
    history = model.fit(
    X_train,
    {
        'primary_output': yp_train,
        'secondary_output': ys_train
    },
    validation_data=(
        X_val,
        {
            'primary_output': yp_val,
            'secondary_output': ys_val
        }
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

    
    return history

def plot_training_history(history):
    """Plot training metrics"""
    print("\n📊 Generating training plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['primary_output_accuracy'], label='Primary Train Acc')
    ax1.plot(history.history['val_primary_output_accuracy'], label='Primary Val Acc')

    ax1.plot(history.history['secondary_output_accuracy'], label='Secondary Train Acc')
    ax1.plot(history.history['val_secondary_output_accuracy'], label='Secondary Val Acc')

    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(plot_path)
    print(f"💾 Training plot saved to {plot_path}")

def save_model_metadata(primary_encoder, secondary_encoder, history):
    """Save model metadata and training info"""
    metadata = {
        'max_words': MAX_WORDS,
        'max_len': MAX_LEN,
        'embedding_dim': EMBEDDING_DIM,
        'primary_classes': primary_encoder.classes_.tolist(),
        'secondary_classes': secondary_encoder.classes_.tolist(),
        'num_primary_classes': len(primary_encoder.classes_),
        'num_secondary_classes': len(secondary_encoder.classes_),
        'final_primary_accuracy': float(history.history['primary_output_accuracy'][-1]),
        'final_primary_val_accuracy': float(history.history['val_primary_output_accuracy'][-1]),
        'final_secondary_accuracy': float(history.history['secondary_output_accuracy'][-1]),
        'final_secondary_val_accuracy': float(history.history['val_secondary_output_accuracy'][-1])

    }
    
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Model metadata saved to {metadata_path}")
    return metadata

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🧠 TECCR EMOTION PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_and_preprocess_data()
    
    # Prepare text data
    X, tokenizer = prepare_text_data(df)
    
    # Prepare labels
    y_primary, y_secondary, primary_encoder, secondary_encoder = prepare_labels(df)
    
    # Split data
    X_train, X_val, yp_train, yp_val, ys_train, ys_val = train_test_split(
        X, y_primary, y_secondary,
        test_size=0.2,
        random_state=42
)

    
    print(f"\n📊 Train samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")
    
    # Build model
    model = build_teccr_model(
        vocab_size=min(MAX_WORDS, len(tokenizer.word_index) + 1),
        num_primary_classes=y_primary.shape[1],
        num_secondary_classes=y_secondary.shape[1]
    )
    
    # Train model
    history = train_model(
    model,
    X_train, yp_train, ys_train,
    X_val, yp_val, ys_val
)

    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "teccr_model.h5")
    model.save(final_model_path)
    print(f"\n💾 Final model saved to {final_model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save metadata
    metadata = save_model_metadata(primary_encoder, secondary_encoder, history)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"📈 Final Primary Train Accuracy: {metadata['final_primary_accuracy']:.4f}")
    print(f"📈 Final Primary Val Accuracy  : {metadata['final_primary_val_accuracy']:.4f}")
    print(f"📈 Final Secondary Train Accuracy: {metadata['final_secondary_accuracy']:.4f}")
    print(f"📈 Final Secondary Val Accuracy  : {metadata['final_secondary_val_accuracy']:.4f}")
    print(f"\n📂 Model files saved in: {MODEL_DIR}/")
    print("   - teccr_model.h5 (trained model)")
    print("   - tokenizer.pkl (text tokenizer)")
    print("   - primary_encoder.pkl (label encoder)")
    print("   - model_metadata.pkl (model info)")
    print("=" * 60)

if __name__ == "__main__":
    main()