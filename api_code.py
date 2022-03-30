from fastapi import FastAPI, Request

import base64
from PIL import Image
from io import BytesIO

from mask_generator import img_to_maskpath

app = FastAPI()

@app.post("/get_mask")
async def receive_img(data: Request):
    # assume data contains your decoded image
    data = await data.json()
    print(data.keys())
    img = Image.open(BytesIO(base64.b64decode(data['img_str'].encode('utf-8'))))

    mask_path = img_to_maskpath(img)
    with open(mask_path, "rb") as image_file:
        data = base64.b64encode(image_file.read())

    return {'img_str': data.decode('utf-8')}