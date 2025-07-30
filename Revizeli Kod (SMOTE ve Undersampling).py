import os
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from imblearn.over_sampling import SMOTE

# Yerel dizinleri belirtin
image_dir_part_1 = r'C:\Users\Cihat\Desktop\DERSLER\4.1\görüntü\cilt\HAM10000_images_part_1'
image_dir_part_2 = r'C:\Users\Cihat\Desktop\DERSLER\4.1\görüntü\cilt\HAM10000_images_part_2'
csv_dir = r'C:\Users\Cihat\Desktop\DERSLER\4.1\görüntü\cilt'  # CSV dosyasının bulunduğu dizin

# HAM10000_metadata.csv dosyasını yükleyin
skin_df = pd.read_csv(os.path.join(csv_dir, 'HAM10000_metadata.csv'))

# Görselleri yüklemek ve boyutlarını ayarlamak için fonksiyon
def load_and_preprocess_images(image_dir, size=(64, 64)):  # Boyutu 64x64 yapalım
    image_paths = glob(os.path.join(image_dir, '*.jpg'))  # Resim dosyalarını bul
    images = []
    for img_path in image_paths:
        # Resimleri aç ve yeniden boyutlandır
        img = Image.open(img_path).convert('RGB')
        img = img.resize(size)
        img_array = np.array(img)
        images.append(img_array)
    return np.array(images)

# Görsellerin boyutunu küçültüyoruz
SIZE = (64, 64)  # 64x64 boyutunda
images_part_1 = load_and_preprocess_images(image_dir_part_1, size=SIZE)
images_part_2 = load_and_preprocess_images(image_dir_part_2, size=SIZE)

# Verileri birleştir
images = np.concatenate([images_part_1, images_part_2], axis=0)

# Label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['dx'])
skin_df['label'] = le.transform(skin_df['dx'])

# Görsellerin yollarını eşleştirme
image_path = {os.path.splitext(os.path.basename(x))[0].lower(): x
              for x in glob(os.path.join(image_dir_part_1, '.jpg')) + glob(os.path.join(image_dir_part_2, '.jpg'))}

skin_df['path'] = skin_df['image_id'].map(lambda x: image_path.get(x.lower(), None))

# Görselleri yükleyip yeniden boyutlandıralım
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE[0], SIZE[1]))) if x else None)

# Eksik görselleri kontrol edelim
missing_images = skin_df[skin_df['image'].isnull()]
print(f"Görselleri yüklemekte sorun yaşanan satırlar: {len(missing_images)}")



# Hastalık dağılımını elde et
hastalik_sayilari = skin_df['dx'].value_counts().reset_index()
hastalik_sayilari.columns = ['Hastalık', 'Sayı']

# Grafik çizimi
plt.figure(figsize=(10, 6))
sns.barplot(data=hastalik_sayilari, x='Hastalık', y='Sayı', color='red')
plt.title("Hastalık Dağılımı")
plt.xlabel("Hastalık")
plt.ylabel("Sayı")
plt.show()

#teşhis türü dağılımı
teshis_sayilari = skin_df['dx_type'].value_counts().reset_index()
teshis_sayilari.columns = ['Teşhis Türü', 'Sayı']

plt.figure(figsize=(10, 6))
sns.barplot(data=teshis_sayilari, x='Teşhis Türü', y='Sayı', color='red')
plt.title("Teşhis Dağılımı")
plt.xlabel("Teşhis Türü")
plt.ylabel("Sayı")
plt.show()

#HASTALIK BÖLGESİ GRAFİĞİ
bölge_sayilari = skin_df['localization'].value_counts().reset_index()
bölge_sayilari.columns = ['Bölge', 'Sayı']

plt.figure(figsize=(10, 6))
sns.barplot(data=bölge_sayilari, x='Bölge', y='Sayı', color='red')
plt.title("Bölge Dağılımı")
plt.xlabel("Bölge")
plt.ylabel("Sayı")
plt.show()

# YAŞ DAĞILIM GRAFİĞİ
yas_sayilari = skin_df['age'].value_counts().reset_index()
yas_sayilari.columns = ['Yaş', 'Sayı']

plt.figure(figsize=(10, 6))
sns.histplot(data=yas_sayilari, x='Yaş', y='Sayı', color='red', kde=True, bins=40)
plt.title("Yaş Dağılımı")
plt.xlabel("Yaş")
plt.ylabel("Sayı")
plt.show()

# CİNSİYET DAĞILIMI
cinsiyet_sayilari = skin_df['sex'].value_counts().reset_index()
cinsiyet_sayilari.columns = ['Cinsiyet', 'Sayı']

plt.figure(figsize=(10, 6))
sns.barplot(data=cinsiyet_sayilari, x='Cinsiyet', y='Sayı', color='red')
plt.title("Cinsiyet Dağılımı")
plt.xlabel("Cinsiyet")
plt.ylabel("Sayı")
plt.show()

import matplotlib.pyplot as plt

# Sınıf sayısını ve eşsiz sınıfları kontrol edin
unique_classes = skin_df['dx'].unique()
n_classes = len(unique_classes)

n_samples = 5
fig, axes = plt.subplots(n_classes, n_samples, figsize=(4 * n_samples, 4 * n_classes))

# Görselleri gruplara göre yerleştirme
for row_idx, (type_name, type_rows) in enumerate(skin_df.groupby('dx')):
    axes[row_idx, 0].set_title(type_name)  # Sınıf başlığını ekle
    samples = type_rows.sample(n_samples, replace=True, random_state=1234)  # Eksik örnekler için replace=True
    for col_idx, (_, c_row) in enumerate(samples.iterrows()):
        ax = axes[row_idx, col_idx]
        ax.imshow(c_row['image'], cmap='gray')  # Görseli göster
        ax.axis('off')  # Eksenleri kapat

# Grafik düzeni ve kaydetme
fig.tight_layout()
fig.savefig('category_samples.png', dpi=300)
plt.show()


# X ve Y veri setlerini oluşturma
X = np.asarray(skin_df['image'].tolist())
X = X / 255.  # Normalizasyon
Y = skin_df['label']  # Etiketleri al
Y_cat = to_categorical(Y, num_classes=7)  # Etiketlerin one-hot encoding'i

# SMOTE ile azınlık sınıflarını artırma
smote = SMOTE(sampling_strategy='auto', random_state=42)

# X'i 2D'ye dönüştürerek SMOTE işlemi yapalım
X_reshaped = X.reshape(X.shape[0], -1)  # (num_samples, num_pixels)
X_res, y_res = smote.fit_resample(X_reshaped, Y)

# Yeniden resimleri 4D'ye dönüştürme
X_res = X_res.reshape(X_res.shape[0], SIZE[0], SIZE[1], 3)

# Undersampling işlemi
class_counts = pd.Series(y_res).value_counts()
max_samples = 5000  # Çoğunluk sınıflarını bu kadar örnekle sınırlayalım

# Her sınıf için sınırlama uygulayalım
X_resampled, y_resampled = [], []

for label in np.unique(y_res):
    class_X = X_res[y_res == label]
    class_y = y_res[y_res == label]

    if class_X.shape[0] > max_samples:
        # Çoğunluk sınıfları için Undersampling
        class_X, class_y = resample(class_X, class_y, n_samples=max_samples, random_state=42)

    X_resampled.append(class_X)
    y_resampled.append(class_y)

# Yeniden birleştirelim
X_resampled = np.concatenate(X_resampled, axis=0)
y_resampled = np.concatenate(y_resampled, axis=0)

# Eğitim ve test verilerine ayırma
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

# One-hot encoding
y_train_cat = to_categorical(y_train, num_classes=7)  # Eğitim etiketlerinin one-hot encoding'i
y_test_cat = to_categorical(y_test, num_classes=7)  # Test etiketlerinin one-hot encoding'i

# Modeli tanımlama
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE[0], SIZE[1], 3)))
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ReduceLROnPlateau tanımlama
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Doğrulama kaybını izliyoruz
    factor=0.5,          # Öğrenme oranını yarıya düşür
    patience=3,          # Doğrulama kaybı 3 epoch boyunca iyileşmezse devreye gir
    min_lr=1e-6          # Öğrenme oranının düşebileceği minimum değer
)

# Modeli eğitme
history = model.fit(
    x_train, y_train_cat,
    epochs=10,  # Maksimum epoch sayısı
    batch_size=16,
    validation_data=(x_test, y_test_cat),
    callbacks=[reduce_lr]  # Callbacks'e ReduceLROnPlateau'yu ekleyin
)

# Eğitim ve doğrulama doğruluğunu görselleştirme
plt.plot(history.history['accuracy'], label='Eğitim doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama doğruluğu')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Modelin tahminlerini alalım
y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test_cat, axis=1)  # One-hot encoding kullanarak

# Classification report (F1, Precision, Recall)
print("Sınıflandırma Raporu:")
print(classification_report(y_test_class, y_pred_class, target_names=le.classes_))

# Confusion Matrix'i çizme
conf_matrix = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('Confusion Matrix')
plt.show()

# Sınıfların örnek sayıları
class_counts = pd.Series(y_resampled).value_counts()

# Sınıf isimlerini etiketlere dönüştürelim
class_labels = le.inverse_transform(class_counts.index)

# Sınıfların örnek sayıları
print("Sınıfların örnek sayıları (SMOTE ve Undersampling sonrası):")
for label, count in zip(class_labels, class_counts):
    print(f"Sınıf: {label} | Örnek Sayısı: {count}")
