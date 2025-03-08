from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import base64
from service.image_generate import generate_image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/txt_generate")
def generate(prompt: str, mode: str = "normal", count: int = 4):
    images = [generate_image(prompt, mode) for _ in range(count)]

    # 1개
    # img_io = BytesIO()
    # images[0].save(img_io, format="PNG")
    # img_io.seek(0)
    #
    # return Response(content=img_io.getvalue(), media_type="image/png")

    # 4개
    encoded_images = []
    for img in images:
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        encoded_images.append(base64.b64encode(img_bytes.getvalue()).decode())

    return JSONResponse(content={"images": encoded_images})

@app.post("/img_generate")
def img2img(prompt: str ="", mode: str = "normal", file: UploadFile = File(...), count: int = 4, strength: float = Query(0.10)):
    image = Image.open(BytesIO(file.file.read()))

    result_images = [ generate_image(prompt, mode, image, strength) for _ in range(count)]

    # 1개
    # img_io = BytesIO()
    # result_image.save(img_io, format="PNG")
    # img_io.seek(0)
    #
    # return Response(content=img_io.getvalue(), media_type="image/png")

    # 4개
    encoded_images = []
    for img in result_images:
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        encoded_images.append(base64.b64encode(img_bytes.getvalue()).decode())

    return JSONResponse(content={"images": encoded_images})
