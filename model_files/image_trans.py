import io
import base64
import numpy as np
from PIL import Image
import pickle
from keras.preprocessing.image import img_to_array


def img_convert(string64):

    default_image_size = tuple((256, 256))
    # with open(r"leaf smut.JPG", "rb") as img_file:
    #     b64_string = base64.b64encode(img_file.read())

    image = Image.open(io.BytesIO(base64.b64decode(string64)))
    if image is not None:
        image = image.resize(default_image_size, Image.ANTIALIAS)
        img_array = img_to_array(image)
        endimg = np.expand_dims(img_array, axis=0)
        
        return endimg
    
    else:
        return None, "Error loading image file"

    