import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import gradio as gr
import pickle as pk
from keras import utils
def pred(image):
    cnn=tf.keras.models.load_model("C:/code/GRADIO/2.cnn/cnn.h5")
    test_image=tf.image.resize(image, (64,64))
    test_image=utils.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    predictions=cnn.predict(test_image)

    # You can also get the class labels if you have a mapping, for example, '0' as 'cat' and '1' as 'dog'
    class_labels = ['cat', 'dog']
    if(predictions[0][0]==0):
        return("CAT")
    else:
        return("DOG")
    

ui=gr.Interface(
    fn=pred,
    inputs=[
        gr.inputs.Image()
        ],
    outputs=gr.outputs.Textbox(label="RESULT"),
    title="DOG V/s CAT CLASSIFIER")
ui.launch()