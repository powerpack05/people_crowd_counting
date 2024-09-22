import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

st.title("People Counting with Density Map")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make POST request to Flask API
    files = {'file': uploaded_file.getvalue()}
    response = requests.post('https://predict/', files=files)

    if response.status_code == 200:
        data = response.json()
        st.write(f"Predicted People Count: {data['people_count']}")

        # Show original image and density map
        original_image = base64.b64decode(data['original_image'])
        density_map = base64.b64decode(data['density_map_image'])

        st.image(Image.open(BytesIO(original_image)), caption='Original Image', use_column_width=True)
        st.image(Image.open(BytesIO(density_map)), caption='Density Map', use_column_width=True)
    else:
        st.error("Error occurred during prediction.")
