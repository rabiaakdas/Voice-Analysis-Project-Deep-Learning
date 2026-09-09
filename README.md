# Voice Analysis Project - Deep Learning

Ses dosyalarından mel-spektrogram özelliklerini çıkartarak Convolutional Neural Network (CNN) modeli ile sanatçı sınıflandırması yapan bir derin öğrenme projesidir.

## 📋 Proje Açıklaması

Bu proje, ses dosyalarını analiz ederek hangi sanatçıya ait olduğunu tahmin etmektedir. Mel-spektrogram özellik çıkarımı ve CNN tabanlı derin öğrenme modeli kullanılarak ses sınıflandırması gerçekleştirilmektedir.

### Temel Özellikler

- 🎵 **Ses İşleme**: Librosa kütüphanesi ile ses dosyalarını yükleme ve işleme
- 📊 **Özellik Çıkarımı**: Mel-spektrogram oluşturma ve normalize etme
- 🤖 **Derin Öğrenme Modeli**: 3 katmanlı CNN modeli ile sınıflandırma
- 🎯 **Tahmin Fonksiyonları**: Tek tahmin ve top-N tahmin desteği
- 📈 **Görselleştirme**: Mel-spektrogram görselleştirmesi

## 🏗️ Proje Yapısı

```
Voice-Analysis-Project-Deep-Learning/
├── your_script.py          # Ana Python script
├── README.md               # Bu dosya
├── training_data/          # Eğitim veri seti (sınıflara göre klasörler)
│   ├── artist1/
│   ├── artist2/
│   └── ...
└── test_audio/             # Test edilecek ses dosyaları
    └── example.wav
```

## 🔧 Gereksinimler

### Kütüphaneler

- Python 3.7+
- numpy
- librosa (ses işleme)
- opencv-python (görüntü işleme)
- matplotlib (görselleştirme)
- scikit-learn (veri bölünmesi)
- tensorflow/keras (derin öğrenme modeli)

### Kurulum

```bash
pip install numpy librosa opencv-python matplotlib scikit-learn tensorflow
```

## 📊 Model Mimarisi

Model, aşağıdaki katmanlardan oluşmaktadır:

```
Input (128x128x1)
    ↓
Conv2D(32, 3x3) + ReLU + MaxPooling(2x2) + Dropout(0.25)
    ↓
Conv2D(64, 3x3) + ReLU + MaxPooling(2x2) + Dropout(0.25)
    ↓
Conv2D(128, 3x3) + ReLU + MaxPooling(2x2) + Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256, ReLU) + Dropout(0.5)
    ↓
Dense(num_classes, Softmax)
```

**Optimizer**: Adam (learning_rate=0.0001)
**Loss Function**: Categorical Crossentropy
**Metrics**: Accuracy

## 📁 Veri Seti Hazırlama

Veri seti aşağıdaki klasör yapısında organize edilmelidir:

```
training_data/
├── artist1/
│   ├── song1.wav
│   ├── song2.wav
│   └── ...
├── artist2/
│   ├── song1.wav
│   ├── song2.wav
│   └── ...
└── artist_n/
    └── ...
```

Her sanatçı için ayrı bir klasör oluşturun ve ses dosyalarını içinde yerleştirin.

## 🚀 Kullanım

### 1. Veri Yükleme ve Eğitim

```python
from your_script import load_data, preprocess_data, create_model
from sklearn.model_selection import train_test_split

# Veri yükleme
data_directory = "path/to/training_data"
data, labels, classes = load_data(data_directory)

# Veri ön işleme
data, labels = preprocess_data(data, labels)

# Eğitim-test bölme
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Model oluşturma
input_shape = X_train[0].shape
num_classes = len(classes)
model = create_model(input_shape, num_classes)

# Eğitim
model.fit(X_train, y_train, epochs=30, batch_size=32,
          validation_data=(X_test, y_test))

# Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### 2. Tek Bir Ses Dosyasını Sınıflandırma

```python
from your_script import predict_artist

file_path = "path/to/audio/file.wav"
predicted_artist, confidence = predict_artist(model, file_path, classes)

print(f"Predicted Artist: {predicted_artist}")
print(f"Confidence: {confidence:.4f}")
```

### 3. Top N Tahminleri Almak

```python
from your_script import predict_top_artists

file_path = "path/to/audio/file.wav"
top_artists = predict_top_artists(model, file_path, classes, top_n=3)

print("Top 3 Artists:")
for artist, confidence in top_artists:
    print(f"  {artist}: {confidence:.4f}")
```

### Ana Script Çalıştırma

```bash
python your_script.py
```

**Not**: `your_script.py` içinde veri ve test dosya yollarını kendi ortamınıza göre güncelleyin:

```python
data_directory = "C:\\path\\to\\training_data"
test_file_path = "C:\\path\\to\\test_audio\\example.wav"
```

## 📈 Eğitim Parametreleri

- **Epochs**: 30
- **Batch Size**: 32
- **Test Split**: 0.2 (20%)
- **Learning Rate**: 0.0001
- **Input Shape**: 128x128 (mel-spektrogram boyutu)

Bu parametreleri ihtiyaç doğrultusunda ayarlayabilirsiniz.

## 🎯 Mel-Spektrogram Hakkında

Mel-spektrogram, ses işlemede kullanılan bir özellik çıkarım yöntemidir:

- Ses sinyalini frekans alanına dönüştürür
- İnsan kulağının algılama biçimini taklit eden Mel ölçeğini kullanır
- Çıkılan spektrogram görüntü olarak işlenir (CNN uyumlu)

## 📊 Fonksiyon Açıklamaları

### `load_data(data_directory, target_shape=(128, 128))`
Eğitim veri setini yükler ve mel-spektrogram özelliklerini çıkarır.

### `preprocess_data(data, labels)`
Veriyi normalize eder ve etiketleri one-hot kodlamasına dönüştürür.

### `create_model(input_shape, num_classes)`
CNN mimarisini oluşturur ve derler.

### `predict_artist(model, file_path, classes, target_shape=(128, 128))`
Tek bir ses dosyasının sanatçısını tahmin eder.

### `predict_top_artists(model, file_path, classes, target_shape=(128, 128), top_n=3)`
Bir ses dosyası için en olası top N sanatçıyı döndürür.

## 💡 İpuçları

1. **Daha Iyi Sonuçlar İçin**:
   - Daha fazla eğitim verisi kullanın
   - Epoch sayısını artırın
   - Farklı learning rate değerleri deneyin
   - Model katmanlarını ayarlayın

2. **Ses Dosyası Formatı**:
   - WAV, MP3, FLAC gibi yaygın formatlar desteklenir
   - Mono veya stereo olabilir
   - Minimum 1-2 saniye ölçüsünde sesler önerilir

3. **Performans İyileştirmesi**:
   - GPU desteği eklemek için CUDA/CuDNN kullanın
   - Batch normalization katmanları ekleyin
   - Veri artırma (data augmentation) uygulamaları yapın

## ⚠️ Bilinen Sınırlamalar

- Matplotlib görselleştirmesi eğitim sırasında pencereler açar (headless ortamda sorunlar yaşanabilir)
- Çok büyük veri setlerinde bellek kullanımı yüksek olabilir
- Model, eğitim sırasında gördüğü sanatçılar üzerinde optimize edilmiştir

## 🔄 Gelecek Geliştirmeler

- [ ] Eğitim parametrelerinin configuration dosyasından okunması
- [ ] Model kaydetme/yükleme (model persistence)
- [ ] Batch işleme desteği
- [ ] Web API interfacesi
- [ ] Veri artırma (data augmentation) fonksiyonları
- [ ] Daha gelişmiş model mimarileri (ResNet, MobileNet)

## 📝 Lisans

MIT License - Detaylar için LICENSE dosyasını kontrol edin.

## 👤 Yazar

Bu proje ses analizi ve sanatçı sınıflandırması için geliştirilmiştir.

## 📞 İletişim ve Destek

Herhangi bir sorun veya soru için lütfen bir issue açınız.

---

**Son Güncelleme**: 2024
