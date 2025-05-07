import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Load model
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
model = load_model()

# Preprocess image
def preprocess_image(image):
    image = image.convert("RGB")
    img = np.array(image)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (256, 256))
    img = img[tf.newaxis, :]
    return img

# Convert tensor to PIL
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# App Title
st.title("🎨 Neural Style Transfer")
st.markdown("Upload a photo and apply your own artistic style or select one from the gallery.")

# Upload content image
content_file = st.file_uploader("📷 Upload your content image", type=["jpg", "jpeg", "png"])

# Choose style source
style_option = st.radio("🎨 Choose style source:", ["Use built-in style", "Upload your own style"])

style_image = None
style_dict = {
    "Starry Night (Van Gogh)": "style_starry_night.jpg",
    "The Great Wave (Hokusai)": "style_wave.jpg",
    "Mosaic": "style_mosaic.jpg",
    "Ghibli": "style_ghibli.jpg",
    "Cubism": "style_cubism.jpg",
    "The Scream (Munch)": "style_the_scream.jpg",
    "Monet Water Lilies": "style_monet_water_lilies.jpg",
    "Pointillism": "style_pointillism.jpg",
    "Ink Wash": "style_ink_wash.jpg",
    "Graffiti": "style_graffiti.jpg"
}

if style_option == "Use built-in style":
    selected_style = st.selectbox("🎭 Choose a style", list(style_dict.keys()))
    style_path = os.path.join("sample_images", style_dict[selected_style])
    style_image = Image.open(style_path)
    st.image(style_image, caption=f"Style: {selected_style}", use_column_width=True)

elif style_option == "Upload your own style":
    style_upload = st.file_uploader("🎨 Upload a style image", type=["jpg", "jpeg", "png"])
    if style_upload:
        style_image = Image.open(style_upload)
        st.image(style_image, caption="Your uploaded style image", use_column_width=True)

# Run style transfer
if content_file and style_image:
    content_image = Image.open(content_file)
    st.image(content_image, caption="Your Content Image", use_column_width=True)

    if st.button("✨ Apply Style"):
        content_tensor = preprocess_image(content_image)
        style_tensor = preprocess_image(style_image)

        stylized_output = model(tf.constant(content_tensor), tf.constant(style_tensor))[0]
        output_image = tensor_to_image(stylized_output)

        st.image(output_image, caption="Stylized Output", use_column_width=True)

        img_bytes = io.BytesIO()
        output_image.save(img_bytes, format="PNG")
        st.download_button("⬇️ Download Stylized Image", data=img_bytes.getvalue(), file_name="stylized_output.png", mime="image/png")

elif not content_file:
    st.warning("Please upload a content image to continue.")
