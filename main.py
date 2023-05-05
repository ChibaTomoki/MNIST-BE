from fastapi import FastAPI
from pydantic import BaseModel
from base64 import b64decode
from io import BytesIO
from PIL import Image as PILImage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Image(BaseModel):
    image_base64: str


@app.get("/sample/")
async def sample():
    return "sample"


@app.post("/images/")
async def create_image(image: Image):
    image_data = b64decode(image.image_base64.split(",")[1])

    image_file = BytesIO(image_data)
    img = PILImage.open(image_file)
    img.save("./posted_png_sample.png")

    return {"message": "画像が正常に受信されました。"}
