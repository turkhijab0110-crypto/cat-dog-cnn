import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path


# =========================
# 1. Model path
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "simple_binary_cnn.pth"


# =========================
# 2. Image path
# =========================

IMAGE_PATH = Path(
    r"C:\Users\turkh\Downloads\test5.jpg"
)


# =========================
# 3. Define CNN model
# =========================

class SimpleBinaryCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, 3, padding=1
        )

        self.conv2 = nn.Conv2d(
            16, 32, 3, padding=1
        )

        self.conv3 = nn.Conv2d(
            32, 64, 3, padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            64 * 8 * 8,
            128
        )

        self.fc2 = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        x = self.pool(
            torch.relu(self.conv1(x))
        )

        x = self.pool(
            torch.relu(self.conv2(x))
        )

        x = self.pool(
            torch.relu(self.conv3(x))
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = torch.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x


# =========================
# 4. Create model
# =========================

model = SimpleBinaryCNN()


# =========================
# 5. Load trained model
# =========================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# =========================
# 6. Image transformation
# =========================

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# =========================
# 7. Load image
# =========================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# =========================
# 8. Transform image
# =========================

image_tensor = transform(
    image
).unsqueeze(0)


# =========================
# 9. Make prediction
# =========================

with torch.no_grad():

    output = model(
        image_tensor
    )

    probability = torch.sigmoid(
        output
    ).item()


# =========================
# 10. Determine class
# =========================

if probability >= 0.5:

    prediction = "dog"
    confidence = probability

else:

    prediction = "cat"
    confidence = 1 - probability


# =========================
# 11. Display result
# =========================

print(
    f"Prediction: {prediction}"
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)