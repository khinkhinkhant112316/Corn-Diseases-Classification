import os
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# App title
st.title('Corn Leaf Diseases Identifier Web App')

# App info
st.write('''
## About

The plant diseases compose a threat to global food security and smallholder farmers whose livelihoods depend mainly on agriculture and healthy crops. 
In developing countries, smallholder farmers produce more than 80% of the agricultural production, 
and reports indicate that more than fifty percent loss in crop due to pests and diseases. 
The world population is expected to grow to more than 9.7 billion by 2050, making food security a major concern in the upcoming years. Hence, rapid and accurate methods of identifying plant diseases are needed to take appropriate measures.

**This Streamlit App utilizes a Deep Learning model to detect diseases (Northern Leaf Blight, Common Rust, Gray Leaf Spot) that attack corn leaves, based on digital images.**

''')

# Sidebar file uploader
st.sidebar.write("# File Required")
uploaded_image = st.sidebar.file_uploader('', type=['jpg', 'png', 'jpeg'])

# Class mapping
map_class = {
    0: 'Northern Leaf Blight',
    1: 'Common Rust',
    2: 'Gray Leaf Spot',
    3: 'Healthy'
}

# DataFrame to store results
df_results = pd.DataFrame({
    'Corn Leaf Condition': ['Northern Leaf Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy'],
    'Confidence': [0, 0, 0, 0]
})

# Function to update DataFrame with predictions
def predictions(preds):
    for i, pred in enumerate(preds[0]):
        df_results.loc[i, 'Confidence'] = pred
    return df_results

# Custom DepthwiseConv2D to handle 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Load the model with caching to prevent reloading on every run
@st.cache_resource
def get_model():
    model = tf.keras.models.load_model("model_mobnetv2.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    return model

if __name__ == '__main__':
    # Load model
    model = get_model()

    # Image preprocessing and display
    if not uploaded_image:
        st.sidebar.write('Please upload an image before proceeding!')
        st.stop()
    else:
        # Decode and resize image
        img_as_bytes = uploaded_image.read()
        st.write("## Corn Leaf Image")
        st.image(img_as_bytes, use_column_width=True)  # Display the image
        img = tf.io.decode_image(img_as_bytes, channels=3)
        img = tf.image.resize(img, (224, 224))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = tf.expand_dims(img_arr, 0)  # Create a batch

    # Preprocess image
    img = preprocess_input(img_arr)

    # Button to trigger prediction
    generate_pred = st.button("Detect Result")

    if generate_pred:
        st.subheader('Probabilities by Class')
        preds = model.predict(img)
        preds_class = preds.argmax()

        # Display prediction results
        st.dataframe(predictions(preds))

        # Display disease information
        if map_class[preds_class] in ["Northern Leaf Blight", "Common Rust", "Gray Leaf Spot"]:
            st.subheader(f"The Corn Leaf is infected by {map_class[preds_class]} disease")
        else:
            st.subheader(f"The Corn Leaf is {map_class[preds_class]}")

