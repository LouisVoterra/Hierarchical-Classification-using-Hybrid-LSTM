import pandas as pd
import numpy as np
import re
import pickle
import os

# Libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# --- KONFIGURASI ---
MAX_LEN = 200 #setiap kalimat dipotong atau klo kurang dikasi padding sampai 200 kata
EMBEDDING_DIM = 100  #setiap kata diubah jadi deretan 100 angka
VOCAB_SIZE = 20000 #koskata paling banyak muncul, selebihnya diabaikan
EPOCHS = 10 
BATCH_SIZE = 64 

if not os.path.exists('models'): #buat folder models kalo belum ada
    os.makedirs('models')

print("1. MEMUAT DATASET...")
df = pd.read_csv('dataset/DBP_wiki_data.csv')  #baca dataset dari file CSV
df = df[['text', 'l1', 'l2', 'l3']].dropna() #ambil kolom text, l1,l2,l3 dan hapus baris kalau ada nilai NaN


df = df.sample(n=100000, random_state=42) #menggambil 50k sample data acak dari dataset 
print(f"Total Data (Sampled): {len(df)}")

#Preprocessing
print("2. CLEANING DATA...")

#method untuk membersikan teks dengan parameter input nya adalah teks
def clean_text(text):
    text = str(text).lower() #ubah semua huruf jadi kecil
    text = re.sub(r'[^a-z0-9\s]', '', text) #hapus karakter selain huruf, angka dan spasi
    return text #mengembalikan teks yang sudah dibersihkan

df['clean_text'] = df['text'].apply(clean_text) #membuat variabel baru 'clean_text' dengan menerapkan fungsi clean_text pada kolom 'text'

# Label Encoding 
print("3. Label Encoder")

#Level 1
le_l1 = LabelEncoder() #var le_l1 instance object dari LabelEncoder 
y1 = le_l1.fit_transform(df['l1'])
y1_cat = to_categorical(y1)
with open('models/le_l1.pkl', 'wb') as f: pickle.dump(le_l1, f)

#Level 2 
le_l2 = LabelEncoder() #var le_l2 instance object dari LabelEncoder 
y2 = le_l2.fit_transform(df['l2'])
y2_cat = to_categorical(y2)
with open('models/le_l2.pkl', 'wb') as f: pickle.dump(le_l2, f) 

#Level 3
le_l3 = LabelEncoder() #var le_l3 instance object dari LabelEncoder 
y3 = le_l3.fit_transform(df['l3'])
y3_cat = to_categorical(y3)
with open('models/le_l3.pkl', 'wb') as f: pickle.dump(le_l3, f) 

#Tokenization
print("4. Tokenisasi ")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text']) #mengubah teks bersih menjadi urutan angka
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

with open('models/tokenizer.pkl', 'wb') as f: pickle.dump(tokenizer, f)

# Split Data
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    X, y1_cat, y2_cat, y3_cat, test_size=0.2, random_state=42
)

#Word2Vec 
print("5. Training Word2Vec ")
sentences = [text.split() for text in df['clean_text']]
w2v_model = Word2Vec(sentences, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=4) #CBOW

word_index = tokenizer.word_index
num_words = min(VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i < VOCAB_SIZE:
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))

#Model Building
print("6. Membangun Model")

input_layer = Input(shape=(MAX_LEN,))

# Embedding
embedding = Embedding(input_dim=num_words, 
                      output_dim=EMBEDDING_DIM, 
                      weights=[embedding_matrix], 
                      trainable=False)(input_layer)

#CNN
cnn = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding)
cnn = MaxPooling1D(pool_size=2)(cnn)
cnn = Dropout(0.2)(cnn)

#LSTM
lstm = LSTM(128, return_sequences=False)(cnn)
lstm = Dropout(0.2)(lstm)

#CABANG OUTPUT
# Output 1: Level 1
dense1 = Dense(64, activation='relu')(lstm)
out1 = Dense(len(le_l1.classes_), activation='softmax', name='output_l1')(dense1)

# Output 2: Level 2
dense2 = Dense(64, activation='relu')(lstm)
out2 = Dense(len(le_l2.classes_), activation='softmax', name='output_l2')(dense2)

# Output 3: Level 3 (INI YANG TADI HILANG)
dense3 = Dense(64, activation='relu')(lstm)
out3 = Dense(len(le_l3.classes_), activation='softmax', name='output_l3')(dense3)

# Build Model with 3 Outputs
model = Model(inputs=input_layer, outputs=[out1, out2, out3])

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics={'output_l1': 'accuracy', 'output_l2': 'accuracy', 'output_l3': 'accuracy'})
model.summary()

#Training
print("7. Training...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    ModelCheckpoint('models/2_projectnlp_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, 
    
    {'output_l1': y1_train, 'output_l2': y2_train, 'output_l3': y3_train}, 
    validation_data=(X_test, {'output_l1': y1_test, 'output_l2': y2_test, 'output_l3': y3_test}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print("Training Selesai.")