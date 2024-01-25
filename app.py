import streamlit as st
import os
import sys
import cv2
from PIL import Image
import numpy as np
import torch 
from torchvision.ops import box_convert
import supervision as sv
import base64
import requests

# Add the GroundingDINO module to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
groundingdino_path = os.path.join(current_dir, "GroundingDINO")
sys.path.append(groundingdino_path)


from GroundingDINO.groundingdino.util.inference import load_model, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T

# Define relative paths for weights and config
WEIGHTS_PATH = os.path.join(current_dir, "weights", "groundingdino_swint_ogc.pth")
CONFIG_PATH = os.path.join(groundingdino_path, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")

openai_token = os.getenv('OPENAI_API_TOKEN')

# Function to encode the file-like object
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

def load_image_from_ui(uploaded_file):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(uploaded_file).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def annotate(image_source, boxes, logits, phrases):
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase}"
        for phrase
        in phrases
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame




st.title('Image Processing App')
# box_threshold = st.slider("Box Threshold", 0.0, 1.0, 0.03)
box_threshold = 0.03
# text_threshold = st.slider("Text Threshold", 0.0, 1.0, 0.03)
text_threshold = 0.03

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])


if uploaded_file is not None:
    # Getting the base64 string
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    base64_image = encode_image(uploaded_file)

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    print("Model loaded!")

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_token}"
    }
    print(openai_token)

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Your primary function is to analyze images of grocery shelves uploaded by users and extract one-word item names. Focus on very generic names. For instance, if the image contains fruits, list them simply as 'banana', 'orange', 'lime'. Provide only names without any additional comments, acting as a classifier. Return only one-word names; for example, 'Pringles' should be classified as 'chips'. Use simple, generic terms like 'spray', 'soda', 'chips'. You will receive one image, and your response should be a list of words, limited to the seven most visible items, without any comments. If you don't see anything, just respond 'product'. Give responce splitted by ','."
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json()['choices'][0]['message']['content'])
    TEXT_PROMPT = response.json()['choices'][0]['message']['content']
    # st.text_input(keywords)
    st.text(TEXT_PROMPT)

# Button to run the processing
# if st.button('Run'):
# if uploaded_file is not None:
    # Convert the file to an image
    image_source, image = load_image_from_ui(uploaded_file)

    # Your existing prediction and annotation code
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold,
        device="cpu"
    )

    # Convert back to numpy array and change channel order from RGB to BGR
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display the processed image
    st.image(annotated_frame, caption='Processed Image', use_column_width=True)
