import streamlit as a
import cv2 as b
import numpy as c
import pandas as d
from yolov8 import e

f = d.read_csv("dataset.csv")

html_style = """
<style>
.container {    padding: 20px;    background-color: #f9f9f9;    border-radius: 10px;    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);}
.title {    color: #ff69b4;    font-size: 36px;    text-align: center;    margin-bottom: 30px;}
.subheader {    color: #ff69b4;    font-size: 24px;    margin-top: 20px;}
.image-container {    margin-top: 20px;    text-align: center;}
</style>
"""
a.markdown(html_style, unsafe_allow_html=True)

a.markdown("<h1 class='title'>AI Skin Analyzer</h1>", unsafe_allow_html=True)

# Skin Analysis Question
g = a.text_input("Describe your skin concern:", 
                                       value="Is there anything unusual I should be aware of?")

# Additional questions
h = a.text_input("Have you experienced any skin irritation recently?", 
                                       value="Yes/No")

i = a.text_input("Are you currently using any skincare products?", 
                                       value="Yes/No")

j = a.text_input("Do you have any known allergies?", 
                                       value="Yes/No")

k = a.text_input("Do you smoke or consume alcohol regularly?", 
                                       value="Yes/No")

l = a.text_input("How many hours of sleep do you get per night?", 
                                       value="7")

m = a.text_input("How many glasses of water do you drink per day?", 
                                       value="8")

# Image Upload 
n = a.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if n is not None:  
    o = b.imdecode(c.fromstring(n.read(), c.uint8), 1)    
    a.image(o, caption="Uploaded Image", use_column_width=True) 

    a.markdown("<h2 class='subheader'>Model Predictions:</h2>", unsafe_allow_html=True)

    # Load YOLOv8 models
    p = []
    p.append(e('yolov8n.pt'))
    p.append(e('yolov8s.pt'))
    p.append(e('yolov8m.pt'))
    p.append(e('yolov8l.pt'))

    # Perform object detection with each model
    for q, r in enumerate(p):
        a.markdown(f"<h3 class='subheader'>Model {q+1}</h3>", unsafe_allow_html=True)
        s = r(o)
        s.render()
        a.image(s.imgs[0], use_column_width=True)

    # Recommendations Based on Question
    # Helper function (needs improvement for accurate matching)
def u(x):
    y = ["acne", "spots", "dry"] # Example keywords 
    z = []
    for aa in y:
        if aa in x.lower():
            z.append(aa.capitalize())
    return z


    a.markdown("<h2 class='subheader'>Recommendations:</h2>", unsafe_allow_html=True)
    for v in t:
        w = f[f['Condition'] == v]
        if not w.empty:
            a.image(w['Image'].values[0], caption=v, use_column_width=True)
            a.write(w['Advice'].values[0])

# Helper function (needs improvement for accurate matching)
def u(x):
    y = ["acne", "spots", "dry"] # Example keywords 
    z = []
    for aa in y:
        if aa in x.lower():
            z.append(aa.capitalize())
    return z
