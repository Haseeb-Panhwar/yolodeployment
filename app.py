import os
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load YOLO model
model = YOLO('yolo11x.pt')

# Upload folder
UPLOAD_FOLDER = 'uploads'
PRED_FOLDER = 'predictions'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'
        
        file = request.files['image']
        
        if file.filename == '':
            return 'No selected file'

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Run YOLO inference
            results = model.predict(source=filepath, save=False, conf=0.5)

            # Get annotated image
            result_image = results[0].plot()

            # Save result image
            pred_path = os.path.join(PRED_FOLDER, f"pred_{file.filename}")
            Image.fromarray(result_image).save(pred_path)

            return render_template('result.html', pred_image=f"pred_{file.filename}")

    return render_template('upload.html')

@app.route('/predictions/<filename>')
def send_pred(filename):
    return send_file(os.path.join(PRED_FOLDER, filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
