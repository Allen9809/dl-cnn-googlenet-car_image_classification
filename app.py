import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

os.environ['TORCH_USE_NNPACK'] = '0'

# Load car classes
with open('car_classes.txt', 'r') as f:
    car_classes = f.read().splitlines()

# Load model artifact
model_path = './car_model.pth'
model = torch.load(model_path, map_location='cpu')
model.eval()

# Define input preprocessing
preprocess = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def model_predict(img, model):
    img = preprocess(img)  # Apply preprocessing
    img = img.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        preds = model(img)
    return preds

# Streamlit application
st.title("CNN Image Classification using Pretrained GoogleNet")

st.header("Recognized Car Brands")
st.markdown("""
    Explore a comprehensive array of globally recognized car brands, meticulously curated and integrated into our system for precise identification.
    
    Brands: Acura, AM, Aston Martin, Audi, Bentley, BMW, Bugatti, Buick, Cadillac, Chevrolet, Chrysler, Daewoo, Dodge, Eagle, Ferrari, Fisker, Ford, Geo, GMC, HUMMER, Honda, Hyundai, Infiniti, Isuzu, Jaguar, Jeep, Lamborghini, Land Rover, Lincoln, Mazda, McLaren, Mercedes-Benz, MINI, Mitsubishi, Nissan, Plymouth, Porsche, Ram, Rolls-Royce, Scion, smart, Spyker, Suzuki, Tesla, Toyota, Volkswagen, Volvo.
""")

st.header("Recognized Years")
st.markdown("""
    Access a curated range of significant automotive years, meticulously integrated into our system for accurate historical recognition.
    
    Years: 1991-2002, 2006-2012.
""")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    preds = model_predict(img, model)
    result = torch.argmax(preds, dim=1).item()
    st.write(f"Result: {car_classes[result]}")

# streamlit run app.py --server.enableXsrfProtection false


# Displaying common image types and their descriptions
st.header("Common Image Types and Descriptions")