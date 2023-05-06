from fastapi import FastAPI
from pydantic import BaseModel
from base64 import b64decode
from io import BytesIO
from PIL import Image as PILImage, ImageOps as PILImageOps
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import nn
from torchvision.transforms import ToTensor

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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


nn_model = NeuralNetwork().to("cpu")
trained_model = torch.load("model.pth")
nn_model.load_state_dict(trained_model)


@app.get("/sample/")
async def sample():
    return "sample"


@app.post("/analyze-images/")
async def create_image(image: Image):
    img_bytes = b64decode(image.image_base64.split(",")[1])
    img_bytes_io = BytesIO(img_bytes)
    img = PILImage.open(img_bytes_io)
    img_gray_scale_inverted = img.convert("L")
    img_gray_scale = PILImageOps.invert(img_gray_scale_inverted)
    img_resized = img_gray_scale.resize((28, 28))
    img_tensor = ToTensor()(img_resized)
    img_tensor_for_nn = img_tensor.unsqueeze(0)

    output = nn_model(img_tensor_for_nn)
    prediction = torch.argmax(output, dim=1)

    return prediction.item()
