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
    try:
        response = requests.post('https://peoplecrowdcounting.streamlit.app/predict/', files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            try:
                # Try to parse the JSON response
                data = response.json()
                st.write(f"Predicted People Count: {data['people_count']}")

                # Show original image and density map
                original_image = base64.b64decode(data['original_image'])
                density_map = base64.b64decode(data['density_map_image'])

                st.image(Image.open(BytesIO(original_image)), caption='Original Image', use_column_width=True)
                st.image(Image.open(BytesIO(density_map)), caption='Density Map', use_column_width=True)

            except ValueError:
                # Handle JSON parsing error
                st.error("Invalid JSON response from the server.")
                st.write(f"Response content: {response.content.decode()}")
        else:
            # If the request wasn't successful, display the status code and response content
            st.error(f"API request failed with status code {response.status_code}")
            st.write(f"Response content: {response.content.decode()}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred during the API request: {str(e)}")
