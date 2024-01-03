# Impor Paket / Library yang diperlukan
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB3, EfficientNetB7
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Conv2D, DepthwiseConv2D, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time

# Menginisialisasi kecepatan pembelajaran awal , Jumlah periode yang akan dilatih dan Ukuran tumpukan
INIT_LR = 1e-4
EPOCHS = 60
# EPOCHS = 1
BS = 32
NETWORK = "ResNet50V2_Dense512_KFold10_TA"

dataset = r"C:\Users\asus\Downloads\TA\Dataset";
model_path = NETWORK + "_" + str(EPOCHS) + ".model"

# Ambil daftar gambar di direktori kumpulan data yang sudah dibuat lalu Inisialisasi
# daftar data (mis., Gambar) dan gambar kelas
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))  # Folder Training berisi gambar Meningioma, Glioma, Pituitary, dan Normal
data = []
labels = []

# Masukkan data diatas dalam Path Gambar
for imagePath in imagePaths:
    # ekstrak label kelas dari nama file
    label = imagePath.split(os.path.sep)[-2]

    # Masukkan / Impor gambar dengan piksel (224x224) dan Lakukan prosesnya terlebih dahulu
    image = load_img(imagePath, target_size=(224, 224, 3))
    image = img_to_array(image)
    image = preprocess_input(image)

    # Perbarui daftar data dan label masing-masing
    data.append(image)
    labels.append(label)

# Konversikan data dan label ke array NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# lakukan Pengkodean satu-hot pada label
# lb = LabelBinarizer()
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print(lb.classes_)

# Menginisialisasi objek KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Lakukan cross validation
fold_no = 1
for train_index, test_index in kf.split(data):
    # Mempersiapkan data train dan data test untuk tiap fold
    trainX, testX = data[train_index], data[test_index]
    trainY, testY = labels[train_index], labels[test_index]

# Mempartisi data menjadi pemisahan pelatihan dan pengujian menggunakan :
# 80% dari data untuk pelatihan dan 20% sisanya untuk pengujian
# (trainX, testX, trainY, testY) = train_test_split(data, labels,
#                                                   test_size=0.20, stratify=labels, random_state=42)

print(len(trainX))
print(len(testX))

# Membangun generator gambar pelatihan untuk augmentasi data
aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.15,
    horizontal_flip=True,
    # brightness_range=(0.5,2.0),
    fill_mode="nearest")

starttime = time.time()
# # Memuat jaringan MobileNetV2, memastikan adanya kumpulan lapisan
# baseModel = MobileNetV2(weights="imagenet", include_top=False,
#                         input_tensor=Input(shape=(224, 224, 3)))

baseModel = ResNet50V2(weights="imagenet", include_top=False,
 	input_tensor=Input(shape=(224, 224, 3)))

# baseModel = InceptionV3(weights="imagenet", include_top=False,
#  	input_tensor=Input(shape=(224, 224, 3)))  #globalpoolnya 5x5

# baseModel = InceptionResNetV2(weights="imagenet", include_top=False,
#  	input_tensor=Input(shape=(224, 224, 3))) #globalpoolnya 5x5

# baseModel = VGG16(weights="imagenet", include_top=False,
#  	input_tensor=Input(shape=(224, 224, 3)))

# baseModel = VGG19(weights="imagenet", include_top=False,
#  	input_tensor=Input(shape=(224, 224, 3)))

# baseModel = EfficientNetB0(weights="imagenet", include_top=False,
#  	input_tensor=Input(shape=(224, 224, 3)))

# baseModel = EfficientNetB3(weights="imagenet", include_top=False,
#  	input_tensor=Input(shape=(224, 224, 3)))

# baseModel = EfficientNetB7(weights="imagenet", include_top=False,
#  	input_tensor=Input(shape=(224, 224, 3)))

baseModel.trainable = False
# baseModel.trainable = True

# # Let's take a look to see how many layers are in the base model
# print("Number of layers in the base model: ", len(baseModel.layers))

# # Fine-tune from this layer onwards
# fine_tune_at = 100  # sekitar block 12

# # Freeze all the layers before the `fine_tune_at` layer
# for layer in baseModel.layers[:fine_tune_at]:
#   layer.trainable = False

# Membangun kepala model yang akan ditempatkan di atas Model dasar
headModel = baseModel.output  # outshape = 7 x 7 x 1280 channel
print(headModel.shape)

# headModel = DepthwiseConv2D(kernel_size=3, strides=(1,1), padding='same')(headModel)
# headModel = SeparableConv2D(filters=320, kernel_size=3, strides=(1,1), padding='same')(headModel)
# headModel = Conv2D(filters=320, kernel_size=3, strides=(1,1), padding='same')(headModel)
# print(headModel.shape)

# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = GlobalAveragePooling2D()(headModel)
# print(headModel.shape)
headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(1024, activation="relu")(headModel)
headModel = Dense(512, activation="relu")(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.1)(headModel)
headModel = Dense(4, activation="softmax")(headModel)

# tempatkan kepala model di atas model dasar ini akan menjadi model sebenarnya yang akan dilatih
model = Model(inputs=baseModel.input, outputs=headModel)
# model.summary()

# Ulangi Semua lapisan dalam model dasar dan bekukan sehingga mereka bisa melakukannya
# dan * tidak * diperbarui selama proses pelatihan pertama
# for layer in baseModel.layers:
# 	layer.trainable = False

# Lakukan Kompilasi model ( Mengukur accuracy )
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# Lakukan pelatihan untuk kepala jaringan
lala = aug.flow(trainX, trainY)
print(len(lala))
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    # steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    # validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Buat prediksi pada set pengujian
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Untuk setiap gambar dalam set pengujian kita perlu menemukan Indeks File label
# dengan probabilitas prediksi terbesar yang sesuai
predIdxs = np.argmax(predIdxs, axis=1)

endtime = time.time()
# Tampilkan Laporan Klasifikasi yang diformat dengan baik
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# Tampilkan running time
print("Running time: {} s".format(endtime - starttime))

# Lakukan Penyimpanan model ke disk
print("[INFO] saving brain tumor detector model...")
model.save(model_path, save_format="h5")  # save ke format .h5

# Munculkan Hasil Klasifikasi , Kompilasi , Pelatihan data diatas dan amatilah akurasi pelatihan
plot_name = NETWORK + "_" + str(EPOCHS)

train_acc = H.history["accuracy"]
val_acc = H.history["val_accuracy"]

train_loss = H.history["loss"]
val_loss = H.history["val_loss"]

epochs_range = range(EPOCHS)

plt.style.use("ggplot")
plt.figure()
plt.plot(epochs_range, train_acc, label="train_acc")
plt.plot(epochs_range, val_acc, label="val_acc")
plt.legend(loc='lower left')
plt.title('Training and Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(epochs_range, train_loss, label="train_loss")
plt.plot(epochs_range, val_loss, label="val_loss")
plt.legend(loc="lower left")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()