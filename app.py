from flask import Flask, render_template, request
import numpy as np
import pickle
import re
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

MAX_LEN = 200

# Path ke file
MODEL_PATH = 'models/2_projectnlp_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'
LE1_PATH = 'models/le_l1.pkl'
LE2_PATH = 'models/le_l2.pkl'
LE3_PATH = 'models/le_l3.pkl'

print("Loading assets...")
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f: tokenizer = pickle.load(f)
        with open(LE1_PATH, 'rb') as f: le1 = pickle.load(f)
        with open(LE2_PATH, 'rb') as f: le2 = pickle.load(f)
        with open(LE3_PATH, 'rb') as f: le3 = pickle.load(f)
        print(">>> SUCCESS: System Ready!")
    else:
        print(">>> ERROR: Model file not found. Please train first.")
except Exception as e:
    print(f">>> ERROR Loading Model: {e}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    original_text = ""
    
    if request.method == 'POST':
        original_text = request.form['text_input']
        
        if model:
            # Preprocess & Predict
            cleaned = clean_text(original_text)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            
            # Predict (Returns list of 3 numpy arrays)
            preds = model.predict(padded)
            
            # Extract Results
            # Output model urutannya: [out1, out2, out3]
            res_l1, res_l2, res_l3 = preds[0][0], preds[1][0], preds[2][0]
            
            # Get Max Probability
            idx1, idx2, idx3 = np.argmax(res_l1), np.argmax(res_l2), np.argmax(res_l3)
            
            prediction = {
                'l1': le1.inverse_transform([idx1])[0],
                'l1_conf': round(float(res_l1[idx1]) * 100, 2),
                
                'l2': le2.inverse_transform([idx2])[0],
                'l2_conf': round(float(res_l2[idx2]) * 100, 2),
                
                'l3': le3.inverse_transform([idx3])[0],
                'l3_conf': round(float(res_l3[idx3]) * 100, 2)
            }
            
    return render_template('index.html', prediction=prediction, original_text=original_text)

if __name__ == '__main__':
    app.run(debug=True)