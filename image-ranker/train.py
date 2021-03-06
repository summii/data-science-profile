import os

import random

import numpy as np
from PIL import Image
from keras import backend as K
from keras import optimizers
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, Flatten, Concatenate
from tqdm import tqdm

from utils import get_image_file_paths


img_heigth, img_width = 224, 224

def preprocess_input(np_img):
    np_img = np_img/127.5
    
    np_img = np_img/1.
    
    return np_img
    
def generate_examples(num_examples):
    x1_data = []
    
    x2_data = []
    
    y_data = []
    
    for i in tqdm(range(num_examples), desc="Preparing data"):
        
        image1_file_path, image2_file_path = random.sample(image_file_paths, 2)
        
        # Load the images and convert them to numpy arrays
        
        img1 = np.array(Image.open(image1_file_path).convert("RGB"))
        img1 = preprocess_input(img1)
        x1_data.append(img1)
        
        img2 = np.array(Image.open(image2_file_path).convert("RGB"))
        
        img2 = preprocess_input(img2)
        x2_data.append(img2)
        
        is_2nd_number_greater = image2_file_path > image1_file_path
        output_vector = [0,1] if is_2nd_number_greater else [1,0]
        y_data.append(output_vector)
        
    x1_data = np.array(x1_data)
    x2_data = np.array(x2_data)
    y_data = np.array(y_data)
    
    return [x1_data, x2_data], y_data
        
        
    
    
if __name__ == "__main__":
    random.seed(42)
    
    image_file_paths = get_image_file_paths()
    
    submodel_inputs = []
    submodel_outputs = []
    num_submodels = 2
    
    for i in range(num_submodels):
        
        submodel = MobileNet(alpha=0.75, weights="imagenet", include_top=False, input_shape=(img_heigth, img_width, 3),)
        
        for layer in submodel.layers:
            
            layer.name = "submodel{}_{}".format(i, layer.name)
            layer.trainable = True
            
        x = submodel.output
        x = Flatten()(x)
        x = Dense(100, activation="relu")(x)
        submodel_inputs.append(submodel.input)
        submodel_outputs.append(x)
        
        
    merged_output = Concatenate()(submodel_outputs)
    merged_output = Dense(100, activation="relu")(merged_output)
    merged_output = Dense(2, activation="softmax")(merged_output)
        
    final_model = Model(inputs=submodel_inputs, outputs=merged_output)
        
    final_model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(momentum=0.9), metrics=["accuracy"])
        
    num_example_per_epoch = 1024
    num_epochs = 20
        
    for i in range(num_epochs):
        x_data, y_data = generate_examples(num_example_per_epoch)
            
        new_lr = 0.002/ (2+i)
        print('Learning Rate: {0:.6f}'.format(new_lr))
        K.set_value(final_model.optimizer.lr, new_lr)
            
        final_model.fit(x_data, y_data, batch_size=16, shuffle=False)
            
    final_model.save(os.path.join(os.path.dirname(__file__), "models", "final.h5"))