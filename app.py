import os
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
from PIL import Image
import io
from azure.storage.blob import BlobServiceClient, BlobBlock
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Azure Storage Account Key from the environment variable
azure_storage_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')

app = Flask(__name__)


# Upload folder
UPLOAD_FOLDER = 'uploads'
PRED_FOLDER = 'predictions'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)

MODEL_PATH = 'yolo11x.pt'  # Local path for the YOLO model file

def download_large_file(connection_string, container_name, blob_name, download_path, chunk_size=4 * 1024 * 1024):
    """Download the model only if not already downloaded"""
    if os.path.exists(download_path):  # If the file already exists locally, no need to download it again
        print(f"Model already exists at {download_path}. Skipping download.")
        return

    # Connect to the blob service
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get the blob client
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Get blob size from properties
    blob_props = blob_client.get_blob_properties()
    blob_size = blob_props.size

    print(f"Blob size: {blob_size / (1024 * 1024):.2f} MB")

    # Open local file for writing
    with open(download_path, "wb") as file:
        offset = 0

        while offset < blob_size:
            # Download this chunk
            stream = blob_client.download_blob(offset=offset, length=chunk_size)
            data = stream.readall()

            # Write to local file
            file.write(data)

            offset += chunk_size
            print(f"Downloaded {offset / (1024 * 1024):.2f} MB")

    print("Download complete.")

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
    if not os.path.exists(MODEL_PATH):  # Check if model exists locally
        download_large_file(
            connection_string=azure_storage_key,
            container_name="videoasblob",
            blob_name="yolo11x.pt",
            download_path=MODEL_PATH,
            chunk_size= 5 * 1024 * 1024  # 5MB chunks
        )
    
    # Load the model once after checking or downloading it
    
    model = YOLO(MODEL_PATH)
    
    app.run(debug=True)
