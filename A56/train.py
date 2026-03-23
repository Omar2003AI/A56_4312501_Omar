import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class TinyImageNet_CNN(nn.Module):

    def __init__(self):
        super(TinyImageNet_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 1, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 384, 5, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 1, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adapt_pool = nn.AdaptiveAvgPool2d((3, 3))

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = self.adapt_pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout1(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(
        root=r"C:\Users\user\Downloads\tiny-imagenet-200 (1)\tiny-imagenet-200\train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=r"C:\Users\user\Downloads\tiny-imagenet-200 (1)\tiny-imagenet-200\val",
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = TinyImageNet_CNN().to(device)

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    epochs = 15
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print("Epoch:", epoch + 1, "Train Batch:", batch_idx)

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx % 25 == 0:
                    print("Epoch:", epoch + 1, "Val Batch:", batch_idx)

                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)

                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_accuracy = 100 * val_correct / val_total

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_tinyimagenet_model_15epochs.pth")

        end_time = time.time()
        epoch_time = end_time - start_time

        print("Epoch:", epoch + 1)
        print("Training Time:", epoch_time)
        print("Training Loss:", train_loss)
        print("Training Accuracy:", train_accuracy)
        print("Validation Accuracy:", val_accuracy)
        print("----------------------------------")

    print("Best Validation Accuracy:", best_val_accuracy)


if __name__ == "__main__":
    main()
