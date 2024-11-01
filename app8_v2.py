from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Charger le modèle
model = load_model('model_efficientnet.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Prétraitement de l'image
    img = Image.open(file.stream)
    img = img.resize((256, 256))  # Redimensionner selon ton modèle
    img_array = np.array(img) / 255.0  # Normaliser
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension du lot

    # Prédiction
    preds = model.predict(img_array)
    mask = np.argmax(preds, axis=-1).squeeze()  # Obtenir la classe prédite

    return jsonify({'mask': mask.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
