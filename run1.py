import streamlit as st
import torch
from torchvision import transforms, models
from torch import nn
import numpy as np
from PIL import Image
import io
import cv2
import tempfile
import base64
import os
import importlib.util


class ActionClassifier(nn.Module):
    def __init__(self, ntargets):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(resnet.fc.in_features),
            nn.Dropout(0.2),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, ntargets)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

ind2cat = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 
          'fighting', 'hugging', 'laughing', 'listening_to_music', 'running', 
          'sitting', 'sleeping', 'texting', 'using_laptop']
config_path = os.path.join(os.path.dirname(__file__), '.config', 'utils.py')
spec = importlib.util.spec_from_file_location("utils", config_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0)
    return image

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActionClassifier(len(ind2cat))
    model.load_state_dict(torch.load('classifier_weights.pth', map_location=device))
    model.eval()
    return model.to(device)

def predict(image, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        prediction = torch.argmax(probabilities).item()
    return ind2cat[prediction], probabilities.cpu().numpy()



def capture_image():
    st.write("**Capture an image using your webcam**")
    captured_image = st.camera_input("Open Webcam and Capture")

    if captured_image is not None:
        # Convert to PIL.Image
        image = Image.open(captured_image)
        st.success("Image Captured!")
        return image

    return None



def main():
    st.title("Human Action Recognition")
    st.write("Upload an image or capture one using the webcam")

    # Option to upload or capture image
    option = st.radio("Select Input Method", ("Upload Image", "Capture from Webcam"))

    image = None
    model_prediction = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

    elif option == "Capture from Webcam":
        image = capture_image()
        if image is not None:
            
            with st.spinner("Model Predicting..."):
                model=load_model()
                model_prediction = utils.predict_class(image)

    if image is not None:
        st.image(image, caption="Selected Image", use_column_width=True)

        # Load model
        with st.spinner("Loading model..."):
            model = load_model()
            model_prediction = utils.predict_class(image)

        if model_prediction is not None:
            st.write(f"**Model Prediction:** {model_prediction}")       

if __name__ == "__main__":
    main()
