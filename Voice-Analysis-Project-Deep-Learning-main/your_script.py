import os
import numpy as np
import librosa
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# Veri işleme ve özellik çıkarımı

def load_data(data_directory, target_shape=(128, 128)):
    """
    Veri kümesini yükler, mel-spektrogram çıkarır ve özellikleri hazırlar.
    """
    data = []  # Veriler için liste
    labels = []  # Etiketler için liste
    classes = os.listdir(data_directory)  # Sınıfları listele
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_directory, class_name)
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                mel_spectrogram_resized = cv2.resize(mel_spectrogram_db, target_shape)
                data.append(mel_spectrogram_resized)
                labels.append(idx)

                # Mel-spektrogram görselleştirme
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
                plt.colorbar(label="dB")
                plt.title(f"Mel-Spektrogram: {file}")
                plt.show()

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels, classes


def preprocess_data(data, labels):
    """
    Veriyi ön işler: kanal boyutu ekler ve etiketleri one-hot kodlar.
    """
    data = np.expand_dims(data, axis=-1)  # Kanal boyutunu ekle
    labels = to_categorical(labels)  # One-hot kodlama
    return data, labels


def create_model(input_shape, num_classes):
    """
    Mel-spektrogramları sınıflandırmak için CNN modeli oluşturur.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    optimizer= Adam(learning_rate=0.0001)
    
    model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def predict_top_artists(model, file_path, classes, target_shape=(128, 128), top_n=3):
    """
    Verilen bir ses dosyasının en olası 'top_n' sanatçılarını tahmin eder.
    """
    try:
        # Ses dosyasını yükle ve mel-spektrogram çıkar
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_resized = cv2.resize(mel_spectrogram_db, target_shape)

        # Veri biçimlendirme
        mel_spectrogram_resized = (mel_spectrogram_resized - np.min(mel_spectrogram_resized)) / (np.max(mel_spectrogram_resized) - np.min(mel_spectrogram_resized))

        mel_spectrogram_resized = np.expand_dims(mel_spectrogram_resized, axis=-1)
        mel_spectrogram_resized = np.expand_dims(mel_spectrogram_resized, axis=0)

        # Modelden tahmin al
        predictions = model.predict(mel_spectrogram_resized)[0]

        # En yüksek olasılıklı 'top_n' sınıfları sıralar
        top_indices = predictions.argsort()[-top_n:][::-1]  # Büyükten küçüğe sıralama
        top_artists = [(classes[i], predictions[i]) for i in top_indices]

        return top_artists
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
    
def predict_artist(model, file_path, classes, target_shape=(128, 128)):
    """
    Verilen bir ses dosyasının sanatçısını tahmin eder.
    """
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_resized = cv2.resize(mel_spectrogram_db, target_shape)

        mel_spectrogram_resized = (mel_spectrogram_resized - np.min(mel_spectrogram_resized)) / (np.max(mel_spectrogram_resized) - np.min(mel_spectrogram_resized))
        mel_spectrogram_resized = np.expand_dims(mel_spectrogram_resized, axis=-1)
        mel_spectrogram_resized = np.expand_dims(mel_spectrogram_resized, axis=0)

        predictions = model.predict(mel_spectrogram_resized)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]
        return classes[predicted_class], confidence
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

# Ana script
if __name__ == "__main__":
    data_directory = "C:\\Users\\Sila Kara韩\\OneDrive\\Masaüstü\\derinogrenme2\\traning_data"
    test_file_path = "C:\\Users\\Sila Kara韩\\OneDrive\\Masaüstü\\derinogrenme2\\test_audio\\example.wav"
    data, labels, classes = load_data(data_directory)
    print(f"Loaded {len(data)} samples from {len(classes)} classes.")

    data, labels = preprocess_data(data, labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    input_shape = X_train[0].shape
    num_classes = len(classes)
    model = create_model(input_shape, num_classes)

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
  
    top_artists = predict_top_artists(model, test_file_path, classes, top_n=3)
    predicted_artist, confidence = predict_artist(model, test_file_path, classes)
    if top_artists:
        print("Top artists and their probabilities:")
        for artist, confidence in top_artists:
            print(f"{artist}: {confidence:.3f}")
    else:
        print("Error in predicting the artist.")

    if predicted_artist:
        print(f"Predicted artist: {predicted_artist}")
        print(f"Confidence: {confidence:.1f}")
    else:
        print("Error in predicting the artist.")   