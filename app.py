from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import matplotlib.pyplot as plt
import torch
from io import BytesIO
import base64
from flask_cors import CORS
from load_model import load_the_model
from utils import preprocess_image
import os

app = Flask(__name__)
CORS(app)
# Load the trained model
trained_model_path = r'G:\People_Crowd_Counting\models\first_model.pth'
model = load_the_model(trained_model_path)
model.eval()

# Helper function to convert image to base64
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

# Helper function to convert density map tensor to an image
def density_map_to_image(density_tensor: torch.Tensor) -> Image.Image:
    density_map = density_tensor.squeeze().cpu().numpy()  # Convert to 2D array
    plt.figure(figsize=(5, 5))
    plt.imshow(density_map, cmap='jet')  # Use a colormap like 'jet'
    plt.axis('off')

    # Save the density map as an image in memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    plt.close()

    # Return the density map as a PIL image
    return Image.open(buffer)

# Route to serve the HTML page
@app.route("/")
def get_homepage():
    return send_from_directory('static', 'index.html')

# Define a route to handle predictions
@app.route("/predict/", methods=['POST'])
def predict_image():
    # Load and preprocess the image
    file = request.files['file']
    image = Image.open(file).convert('RGB')
    image_tensor = preprocess_image(image)

    # Get the model predictions (assuming the model output is a density map)
    with torch.no_grad():
        output = model(image_tensor)

    # Get the count by summing the density map
    no_of_people_count = torch.sum(output).item()

    # Convert density map tensor to a heatmap image
    density_map_image = density_map_to_image(output)

    # Convert both images (original and density map) to base64
    original_image_base64 = image_to_base64(image)
    density_map_base64 = image_to_base64(density_map_image)

    # Return both images as base64 strings along with the predicted count
    return jsonify({
        "people_count": no_of_people_count,
        "original_image": original_image_base64,
        "density_map_image": density_map_base64
    })

# Serve static files (CSS, JavaScript, HTML)
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
