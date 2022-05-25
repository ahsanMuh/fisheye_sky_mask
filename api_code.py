from fastapi import FastAPI, Request

import base64
from PIL import Image
from io import BytesIO
import json

from mask_generator import img_to_maskpath
from complete_pipeline_api import img_to_shading_factor

app = FastAPI()

@app.post("/get_mask")
async def receive_img(data: Request):
    # assume data contains your decoded image
    data = await data.json()
    #print(data.keys())
    
    img = Image.open(BytesIO(base64.b64decode(data['img_str'].encode('utf-8'))))

    _ = img_to_maskpath(img)

    with open('mask.png', "rb") as image_file:
        data = base64.b64encode(image_file.read())

    return {'img_str': data.decode('utf-8')}


@app.post("/get_shadding_factor")
async def priduce_shadding(data: Request):
    # assume data contains your decoded image
    data = await data.json()
    #print(data.keys())
    
    img = Image.open(BytesIO(base64.b64decode(data['img_str'].encode('utf-8'))))
    dLatitude = float(data['dLatitude'])
    dLongitude = float(data['dLongitude'])

    res_df = img_to_shading_factor(img, dLatitude, dLongitude)


    return json.dumps(list(res_df.T.to_dict().values()))