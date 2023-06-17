from fastapi import FastAPI
from pydantic import BaseModel
from base64 import b64decode
from io import BytesIO
from PIL import Image as PILImage, ImageOps as PILImageOps
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import nn
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
from os import getenv, remove
from pymongo import MongoClient
from typing import Any, Dict

load_dotenv()

mongo_url = getenv("MONGO_URL")

client: MongoClient[Dict[str, Any]] = MongoClient(mongo_url)
db = client["mydatabase"]
collection = db["mymodels"]

model_data = collection.find_one()["model"]

with open("temp_model.pth", "wb") as f:
    f.write(model_data)

app = FastAPI()
origins = [getenv("FE_URL")]
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
        self.lenet5 = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5)),  # -> 6*28*28
            nn.ReLU(),
            nn.AvgPool2d((2, 2), 2),  # -> 6*14*14
            nn.ReLU(),
            nn.Conv2d(6, 16, (5, 5)),  # -> 16*10*10
            nn.ReLU(),
            nn.AvgPool2d((2, 2), 2),  # -> 16*5*5
            nn.ReLU(),
            nn.Conv2d(16, 120, (5, 5)),  # -> 120*1*1
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.lenet5(x)


nn_model = NeuralNetwork().to("cpu")
trained_model = torch.load("temp_model.pth")
remove("temp_model.pth")
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
    img_resized = img_gray_scale.resize((32, 32))
    img_tensor = ToTensor()(img_resized)
    img_tensor_for_nn = img_tensor.unsqueeze(0)

    output = nn_model(img_tensor_for_nn)
    prediction = torch.argmax(output, dim=1)
    prob_distribution = torch.softmax(output, 1)
    max_prob = torch.gather(prob_distribution, 1, prediction.view(-1, 1))
    print(max_prob.item())

    return {"num": prediction.item(), "prob": max_prob.item()}
