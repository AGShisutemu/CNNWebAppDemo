import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO
import io
import os


def base64_to_pil(img_base64):
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def base64_string_to_pillow_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))


def get_base_image_url():
    # Get the current host dynamically from the request object in Flask
    current_host = request.host_url

    # Assuming your images will be served from a specific directory like '/images/'
    base_image_url = urljoin(current_host, "/images/")
    return base_image_url


def save_base64_to_image(base64_string, output_filename):
    try:
        # Decode the Base64 string to binary data
        raw_string = re.sub('^data:image/.+;base64,', '', base64_string)
        image_data = base64.b64decode(raw_string)

        output_file_path = os.path.join("uploads", output_filename)
        # Write the binary data to the output file
        with open(output_file_path, "wb") as image_file:
            image_file.write(image_data)
        
        print("Image saved successfully as:", output_file_path)
    except Exception as e:
        print("Error saving the image:", str(e))
