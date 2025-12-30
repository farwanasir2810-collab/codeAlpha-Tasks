"""
=================================================================
EMOTION RECOGNITION FROM SPEECH - DEEP LEARNING PROJECT
=================================================================
Author: ML Student Project
Objective: Recognize human emotions from speech audio
Approach: Deep Learning + Audio Signal Processing
Features: MFCCs, Chroma, Mel Spectrogram, Zero Crossing Rate
Models: CNN-LSTM Hybrid Architecture
Datasets: RAVDESS, TESS (can be extended to EMO-DB)
=================================================================
"""

# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import librosa.display
from scipy.io import wavfile

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print("EMOTION RECOGNITION FROM SPEECH - DEEP LEARNING PROJECT")
print("=" * 80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Librosa Version: {librosa.__version__}")

# ============================================================
# STEP 2: EMOTION LABELS AND DATASET INFORMATION
# ============================================================
print("\n[STEP 1] Setting Up Emotion Labels...")

# Define emotion categories
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# For this demo, we'll focus on 4 main emotions
FOCUSED_EMOTIONS = ['happy', 'sad', 'angry', 'neutral']

print(f"  Total Emotions: {len(EMOTIONS)}")
print(f"  Emotions: {list(EMOTIONS.values())}")
print(f"\n  Focused Emotions for Classification: {FOCUSED_EMOTIONS}")

# ============================================================
# STEP 3: AUDIO FEATURE EXTRACTION FUNCTIONS
# ============================================================
print("\n[STEP 2] Defining Audio Feature Extraction Functions...")

def extract_features(file_path, max_pad_len=174):
    """
    Extract audio features from speech file
    
    Features extracted:
    1. MFCCs (Mel-Frequency Cepstral Coefficients) - 40 coefficients
    2. Chroma - 12 values
    3. Mel Spectrogram - Mean
    4. Zero Crossing Rate
    5. Root Mean Square Energy
    
    Parameters:
    - file_path: path to audio file
    - max_pad_len: maximum length for padding/truncating
    
    Returns:
    - feature_vector: numpy array of extracted features
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050)
        
        # Feature 1: MFCCs (most important for speech)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Feature 2: Chroma (pitch class profiles)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        # Feature 3: Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_scaled = np.mean(mel.T, axis=0)
        
        # Feature 4: Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_scaled = np.mean(zcr.T, axis=0)
        
        # Feature 5: Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        rms_scaled = np.mean(rms.T, axis=0)
        
        # Concatenate all features
        features = np.concatenate([mfccs_scaled, chroma_scaled, mel_scaled, 
                                   zcr_scaled, rms_scaled])
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def visualize_audio_features(file_path, emotion):
    """
    Visualize various audio features
    """
    audio, sample_rate = librosa.load(file_path, duration=2.5)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f'Audio Features - Emotion: {emotion.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Waveform
    librosa.display.waveshow(audio, sr=sample_rate, ax=axes[0])
    axes[0].set_title('Waveform')
    axes[0].set_ylabel('Amplitude')
    
    # 2. MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    img = librosa.display.specshow(mfccs, x_axis='time', ax=axes[1], cmap='coolwarm')
    axes[1].set_title('MFCCs (Mel-Frequency Cepstral Coefficients)')
    axes[1].set_ylabel('MFCC')
    plt.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # 3. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', 
                                    ax=axes[2], cmap='viridis')
    axes[2].set_title('Mel Spectrogram')
    axes[2].set_ylabel('Frequency (Hz)')
    plt.colorbar(img, ax=axes[2], format='%+2.0f dB')
    
    # 4. Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', 
                                    ax=axes[3], cmap='plasma')
    axes[3].set_title('Chroma Features')
    axes[3].set_ylabel('Pitch Class')
    plt.colorbar(img, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(f'audio_features_{emotion}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Audio features visualization saved for '{emotion}' emotion")

print("  ✓ Feature extraction functions defined")
print("    Features: MFCCs, Chroma, Mel Spectrogram, ZCR, RMS")

# ============================================================
# STEP 4: CREATE SYNTHETIC DATASET (DEMO VERSION)
# ============================================================
print("\n[STEP 3] Creating Synthetic Audio Dataset for Demo...")
print("  Note: In real project, use RAVDESS/TESS dataset")

def generate_synthetic_audio_features(num_samples=800):
    """
    Generate synthetic audio features for demonstration
    In real project, replace with actual audio file processing
    """
    features_list = []
    labels_list = []
    
    # Feature dimensions: 40 (MFCCs) + 12 (Chroma) + 128 (Mel) + 1 (ZCR) + 1 (RMS) = 182
    feature_dim = 182
    
    for emotion_idx, emotion in enumerate(FOCUSED_EMOTIONS):
        samples_per_emotion = num_samples // len(FOCUSED_EMOTIONS)
        
        for i in range(samples_per_emotion):
            # Generate features with emotion-specific characteristics
            if emotion == 'happy':
                # Happy: Higher energy, faster rhythm
                features = np.random.randn(feature_dim) * 0.8 + np.random.rand(feature_dim) * 0.5
            elif emotion == 'sad':
                # Sad: Lower energy, slower rhythm
                features = np.random.randn(feature_dim) * 0.5 - np.random.rand(feature_dim) * 0.3
            elif emotion == 'angry':
                # Angry: High energy, sharp peaks
                features = np.random.randn(feature_dim) * 1.2 + np.random.rand(feature_dim) * 0.8
            else:  # neutral
                # Neutral: Moderate energy, balanced
                features = np.random.randn(feature_dim) * 0.7
            
            features_list.append(features)
            labels_list.append(emotion)
    
    return np.array(features_list), np.array(labels_list)

# Generate synthetic dataset
X_features, y_labels = generate_synthetic_audio_features(num_samples=1000)

print(f"  ✓ Dataset created:")
print(f"    Total samples: {len(X_features)}")
print(f"    Feature dimension: {X_features.shape[1]}")
print(f"    Emotions: {FOCUSED_EMOTIONS}")

# Show distribution
unique, counts = np.unique(y_labels, return_counts=True)
print(f"\n  Emotion Distribution:")
for emotion, count in zip(unique, counts):
    print(f"    {emotion.capitalize()}: {count} samples ({count/len(y_labels)*100:.1f}%)")

# ============================================================
# STEP 5: DATA PREPROCESSING
# ============================================================
print("\n[STEP 4] Preprocessing Data...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)
y_categorical = to_categorical(y_encoded, num_classes=len(FOCUSED_EMOTIONS))

print(f"  ✓ Labels encoded:")
print(f"    Example: '{y_labels[0]}' → {y_encoded[0]} → {y_categorical[0]}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

print(f"\n  ✓ Features standardized:")
print(f"    Original range: [{X_features.min():.2f}, {X_features.max():.2f}]")
print(f"    Scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Further split for validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

print(f"\n  ✓ Data split:")
print(f"    Training: {len(X_train_final)} samples ({len(X_train_final)/len(X_scaled)*100:.1f}%)")
print(f"    Validation: {len(X_val)} samples ({len(X_val)/len(X_scaled)*100:.1f}%)")
print(f"    Test: {len(X_test)} samples ({len(X_test)/len(X_scaled)*100:.1f}%)")

# Reshape for CNN-LSTM (add time dimension)
X_train_reshaped = X_train_final.reshape(X_train_final.shape[0], X_train_final.shape[1], 1)
X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"\n  ✓ Reshaped for CNN-LSTM:")
print(f"    Training shape: {X_train_reshaped.shape}")
print(f"    Validation shape: {X_val_reshaped.shape}")
print(f"    Test shape: {X_test_reshaped.shape}")

# ============================================================
# STEP 6: BUILD CNN-LSTM HYBRID MODEL
# ============================================================
print("\n[STEP 5] Building CNN-LSTM Hybrid Model...")

def create_emotion_model(input_shape, num_classes):
    """
    Create CNN-LSTM hybrid model for emotion recognition
    
    Architecture:
    - 1D Convolutional layers for feature extraction
    - LSTM layers for temporal pattern learning
    - Dense layers for classification
    """
    
    model = models.Sequential([
        # CNN Block 1
        layers.Conv1D(64, kernel_size=5, activation='relu', 
                     input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2, name='pool1'),
        layers.Dropout(0.3),
        
        # CNN Block 2
        layers.Conv1D(128, kernel_size=5, activation='relu', name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2, name='pool2'),
        layers.Dropout(0.3),
        
        # CNN Block 3
        layers.Conv1D(256, kernel_size=3, activation='relu', name='conv3'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2, name='pool3'),
        layers.Dropout(0.3),
        
        # LSTM Layers for temporal learning
        layers.LSTM(128, return_sequences=True, name='lstm1'),
        layers.Dropout(0.4),
        layers.LSTM(64, return_sequences=False, name='lstm2'),
        layers.Dropout(0.4),
        
        # Dense Layers
        layers.Dense(128, activation='relu', name='dense1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', name='dense2'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

# Create model
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
model = create_emotion_model(input_shape, num_classes=len(FOCUSED_EMOTIONS))

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("  ✓ CNN-LSTM Model Created Successfully!")
print("\n  Model Architecture:")
model.summary()

# Count parameters
trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
print(f"\n  Total Trainable Parameters: {trainable_params:,}")

# ============================================================
# STEP 7: TRAIN THE MODEL
# ============================================================
print("\n[STEP 6] Training Emotion Recognition Model...")
print("-" * 80)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_emotion_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\n  Starting training...")
print("  Tip: Watch accuracy improve over epochs!\n")

history = model.fit(
    X_train_reshaped, y_train_final,
    batch_size=32,
    epochs=50,
    validation_data=(X_val_reshaped, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

print("\n" + "=" * 80)
print("  ✓ Training Completed!")
print("=" * 80)

# ============================================================
# STEP 8: EVALUATE MODEL
# ============================================================
print("\n[STEP 7] Evaluating Model Performance...")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)

print(f"\n  📊 Test Results:")
print(f"  ├─ Test Loss: {test_loss:.4f}")
print(f"  └─ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions
y_pred_probs = model.predict(X_test_reshaped, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Convert to emotion labels
y_pred_emotions = label_encoder.inverse_transform(y_pred_classes)
y_true_emotions = label_encoder.inverse_transform(y_true_classes)

# Calculate metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"\n  📈 Detailed Metrics:")
print(f"  ├─ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ├─ Precision: {precision:.4f}")
print(f"  ├─ Recall:    {recall:.4f}")
print(f"  └─ F1-Score:  {f1:.4f}")

# Classification report
print("\n  📋 Classification Report (Per Emotion):")
print(classification_report(y_true_emotions, y_pred_emotions, 
                          target_names=FOCUSED_EMOTIONS))

# ============================================================
# STEP 9: CONFUSION MATRIX
# ============================================================
print("\n[STEP 8] Generating Confusion Matrix...")

cm = confusion_matrix(y_true_classes, y_pred_classes)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=FOCUSED_EMOTIONS, yticklabels=FOCUSED_EMOTIONS)
ax.set_title('Confusion Matrix - Emotion Recognition', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('True Emotion', fontsize=12)
ax.set_xlabel('Predicted Emotion', fontsize=12)

plt.tight_layout()
plt.savefig('emotion_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ Confusion matrix saved as 'emotion_confusion_matrix.png'")

# ============================================================
# STEP 10: TRAINING HISTORY VISUALIZATION
# ============================================================
print("\n[STEP 9] Visualizing Training History...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('emotion_training_history.png', dpi=300, bbox_inches='tight')
print("  ✓ Training history saved as 'emotion_training_history.png'")

# ============================================================
# STEP 11: PREDICTION FUNCTION
# ============================================================
print("\n[STEP 10] Creating Prediction Function...")

def predict_emotion(audio_file_path):
    """
    Predict emotion from audio file
    
    Parameters:
    - audio_file_path: path to audio file (.wav format)
    
    Returns:
    - predicted_emotion: string (happy, sad, angry, neutral)
    - confidence: float (0-1)
    - all_probabilities: dict with all emotion probabilities
    """
    # Extract features
    features = extract_features(audio_file_path)
    
    if features is None:
        return None, None, None
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)
    
    # Predict
    probabilities = model.predict(features_reshaped, verbose=0)[0]
    predicted_class = np.argmax(probabilities)
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    confidence = probabilities[predicted_class]
    
    # Create probability dictionary
    prob_dict = {emotion: float(prob) for emotion, prob in 
                 zip(FOCUSED_EMOTIONS, probabilities)}
    
    return predicted_emotion, confidence, prob_dict

print("  ✓ Prediction function created")
print("\n  Usage example:")
print("    emotion, conf, probs = predict_emotion('audio.wav')")
print("    print(f'Emotion: {emotion}, Confidence: {conf:.2%}')")

# ============================================================
# STEP 12: EMOTION-WISE PERFORMANCE ANALYSIS
# ============================================================
print("\n[STEP 11] Analyzing Per-Emotion Performance...")

# Calculate per-emotion accuracy
emotion_performance = {}
for emotion in FOCUSED_EMOTIONS:
    emotion_idx = label_encoder.transform([emotion])[0]
    mask = y_true_classes == emotion_idx
    if mask.sum() > 0:
        emotion_acc = (y_pred_classes[mask] == y_true_classes[mask]).mean()
        emotion_performance[emotion] = emotion_acc

print("\n  Per-Emotion Accuracy:")
for emotion, acc in emotion_performance.items():
    print(f"    {emotion.capitalize()}: {acc*100:.2f}%")

# Visualize emotion performance
fig, ax = plt.subplots(figsize=(10, 6))
emotions = list(emotion_performance.keys())
accuracies = list(emotion_performance.values())

bars = ax.bar(emotions, accuracies, color=['#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3'])
ax.set_title('Per-Emotion Recognition Accuracy', fontsize=16, fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height*100:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('emotion_accuracy_by_class.png', dpi=300, bbox_inches='tight')
print("\n  ✓ Per-emotion accuracy saved as 'emotion_accuracy_by_class.png'")

# ============================================================
# STEP 13: SAVE MODEL
# ============================================================
print("\n[STEP 12] Saving Model...")

# Save complete model
model.save('emotion_recognition_model.h5')
print("  ✓ Model saved as 'emotion_recognition_model.h5'")

# Save model architecture as JSON
model_json = model.to_json()
with open('emotion_model_architecture.json', 'w') as json_file:
    json_file.write(model_json)
print("  ✓ Model architecture saved as 'emotion_model_architecture.json'")

# Save label encoder
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("  ✓ Label encoder saved as 'label_encoder.pkl'")

# Save scaler
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  ✓ Feature scaler saved as 'feature_scaler.pkl'")

# ============================================================
# STEP 14: PROJECT SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)

print(f"""
✓ Task: Emotion Recognition from Speech
✓ Total Samples: {len(X_features):,}
✓ Training Samples: {len(X_train_final):,}
✓ Validation Samples: {len(X_val):,}
✓ Test Samples: {len(X_test):,}
✓ Emotions: {', '.join(FOCUSED_EMOTIONS)}
✓ Model: CNN-LSTM Hybrid
✓ Total Parameters: {trainable_params:,}

📊 Final Performance:
  ├─ Test Accuracy: {test_accuracy*100:.2f}%
  ├─ Precision: {precision:.4f}
  ├─ Recall: {recall:.4f}
  └─ F1-Score: {f1:.4f}

🎯 Audio Features Extracted:
  ├─ MFCCs (40 coefficients)
  ├─ Chroma (12 values)
  ├─ Mel Spectrogram (128 bins)
  ├─ Zero Crossing Rate
  └─ Root Mean Square Energy

📈 Training Information:
  ├─ Total Epochs: {len(history.history['accuracy'])}
  ├─ Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%
  ├─ Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%
  └─ Batch Size: 32

🎙️ Model Capabilities:
  ✓ Recognizes 4 emotions: happy, sad, angry, neutral
  ✓ Processes 2.5-second audio clips
  ✓ Real-time emotion detection capable
  ✓ Confidence scores for predictions

📁 Generated Files:
  ✓ emotion_confusion_matrix.png
  ✓ emotion_training_history.png
  ✓ emotion_accuracy_by_class.png
  ✓ emotion_recognition_model.h5
  ✓ label_encoder.pkl
  ✓ feature_scaler.pkl
""")

print("=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\n💡 Next Steps:")
print("  1. Test with real RAVDESS/TESS dataset")
print("  2. Implement real-time emotion detection")
print("  3. Add more emotions (fearful, surprised, disgust)")
print("  4. Create web interface for audio upload")
print("  5. Deploy as voice assistant emotion analyzer")
print("\n📚 To use with real audio files:")
print("  Download RAVDESS dataset from: https://zenodo.org/record/1188976")
print("  Replace synthetic data generation with actual file processing")