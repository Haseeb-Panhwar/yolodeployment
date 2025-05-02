

import os
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
from PIL import Image
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

azure_storage_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
UPLOAD_FOLDER = 'uploads'
PRED_FOLDER = 'predictions'
MODEL_PATH = 'yolo11x.pt'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)


def initialize_model():
    """Initialize the YOLO model once"""
    global model
    if not os.path.exists(MODEL_PATH):
        download_large_file(
            connection_string=azure_storage_key,
            container_name="videoasblob",
            blob_name="yolo11x.pt",
            download_path=MODEL_PATH,
            chunk_size=5 * 1024 * 1024  # 5MB chunks
        )
    model = YOLO(MODEL_PATH)
    
    
# Global model variable
model = None
initialize_model()


def download_large_file(connection_string, container_name, blob_name, download_path, chunk_size=4 * 1024 * 1024):
    """Download a large file from Azure Blob Storage in chunks"""
    if os.path.exists(download_path):
        print(f"Model already exists at {download_path}. Skipping download.")
        return

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    blob_props = blob_client.get_blob_properties()
    blob_size = blob_props.size
    print(f"Downloading model: {blob_size / (1024 * 1024):.2f} MB")

    with open(download_path, "wb") as file:
        offset = 0
        while offset < blob_size:
            stream = blob_client.download_blob(offset=offset, length=chunk_size)
            data = stream.readall()
            file.write(data)
            offset += chunk_size
            print(f"Downloaded {offset / (1024 * 1024):.2f} MB")

    print("Model download complete.")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global model

    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'

        file = request.files['image']

        if file.filename == '':
            return 'No selected file'

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Inference using YOLO
            results = model.predict(source=filepath, save=False, conf=0.5)
            result_image = results[0].plot()

            # Save prediction
            pred_path = os.path.join(PRED_FOLDER, f"pred_{file.filename}")
            Image.fromarray(result_image).save(pred_path)

            return render_template('result.html', pred_image=f"pred_{file.filename}")

    return render_template('upload.html')


@app.route('/predictions/<filename>')
def send_pred(filename):
    return send_file(os.path.join(PRED_FOLDER, filename), mimetype='image/jpeg')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
