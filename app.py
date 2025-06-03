from flask import Flask, render_template, request, send_file
from PIL import Image
import os
import torch
from realesrgan import RealESRGAN

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RealESRGAN(device, scale=4)
model.load_weights('RealESRGAN_x4.pth')  # Make sure you have this weights file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    img = Image.open(img_path).convert('RGB')
    sr_img = model.predict(img)

    enhanced_path = os.path.join(UPLOAD_FOLDER, 'enhanced_' + file.filename)
    sr_img.save(enhanced_path)

    return send_file(enhanced_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)