from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

from PIL import Image
import glob
for filename in glob.glob('E:\\COCO-QA Dataset\\Testing Images/*.jpg'): 
    img_path = filename
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    block4_pool_features = model.predict(x)
    
    V = np.full((512,49),0.0)
    c=0
    for i in range(7):
        for j in range(7):
            V[:,c] = block4_pool_features[0,i,j,:]
            c = c + 1
    import csv
    with open("E:\\COCO-QA Dataset\\Testing Images Features\\" + filename[36:-4] + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(V)

