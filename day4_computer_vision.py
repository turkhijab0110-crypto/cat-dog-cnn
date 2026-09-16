import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path


# =========================
# 1. Project paths
# =========================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"

MODEL_PATH = BASE_DIR / "simple_binary_cnn.pth"


# =========================
# 2. Image transformations
# =========================

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# =========================
# 3. Load datasets
# =========================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


# =========================
# 4. Check dataset
# =========================

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# =========================
# 5. Create DataLoaders
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# =========================
# 6. Define CNN model
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
# 7. Create model
# =========================

model = SimpleBinaryCNN()

print(model)


# =========================
# 8. Select device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

model = model.to(device)


# =========================
# 9. Loss function
# =========================

criterion = nn.BCEWithLogitsLoss()


# =========================
# 10. Optimizer
# =========================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 11. Training
# =========================

num_epochs = 10

print("\nStarting training...\n")

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.float().to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Make prediction
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs.squeeze(1),
            labels
        )

        # Calculate gradients
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Convert output to 0 or 1
        predictions = (
            torch.sigmoid(
                outputs.squeeze(1)
            ) >= 0.5
        ).float()

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    accuracy = (
        100 * correct / total
    )

    average_loss = (
        running_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# =========================
# 12. Test the model
# =========================

print("\nTesting model...\n")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)

        predictions = (
            torch.sigmoid(
                outputs.squeeze(1)
            ) >= 0.5
        ).float()

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


test_accuracy = (
    100 * correct / total
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)


# =========================
# 13. Save trained model
# =========================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nModel saved successfully!")
print(f"File: {MODEL_PATH}")