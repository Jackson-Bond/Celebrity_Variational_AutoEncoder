# Bond, Jackson
# 1002_032_254
# 2026_03_29
# Assignment_04_02

from pathlib import Path
import time
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from vae import VAE, vaeLoss


# class for loading CelebA images using the Kaggle CSV format
class CelebADataset(Dataset):
    splitMap = {"train": 0, "val": 1, "test": 2}

    def __init__(self, root, split="train", transform=None, imgSubdir="img_align_celeba", partitionCsv="list_eval_partition.csv"):

        self.imgDir = Path(root) / imgSubdir
        self.transform = transform

        partitionPath = Path(root) / partitionCsv
        df = pd.read_csv(partitionPath)
        df.columns = [c.strip() for c in df.columns]

        imgCol = None
        partCol = None
        for c in df.columns:
            if "image" in c.lower():
                imgCol = c
            if "partition" in c.lower() or "split" in c.lower():
                partCol = c

        splitCode = self.splitMap[split]
        self.filenames = df.loc[df[partCol] == splitCode, imgCol].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        imgPath = self.imgDir / self.filenames[idx]
        img     = Image.open(imgPath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def trainEpoch(model, dataLoader, optimizer, device, beta, epoch, totalEpochs):
    model.train()
    totalLoss = 0.0
    startTime = time.time()

    for batchIdx, (x, _) in enumerate(dataLoader):
        x = x.to(device)

        optimizer.zero_grad()
        xHat, mu, logvar = model(x)
        loss = vaeLoss(xHat, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

        # just nice to have to see progress during training
        if (batchIdx + 1) % 128 == 0:
            elapsed = time.time() - startTime
            print(f"  Epoch [{epoch}/{totalEpochs}]  Batch [{batchIdx+1}/{len(dataLoader)}] s"f"  Loss: {loss.item():.4f}  ({elapsed:.0f}s)")

    return totalLoss / len(dataLoader)



def train(dataDir="./celeba", epochs=20, batchSize=128, lr=1e-3, beta=4.0, save="Bond_04_04.pth", numWorkers=4):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device : {device}")
    print(f"Data   : {Path(dataDir).resolve()}")

    # build dataset and dataloader
    imageTransform = transforms.Compose([transforms.CenterCrop(178), transforms.Resize(64), transforms.ToTensor()])
    trainDataset = CelebADataset(root=dataDir, split="train", transform=imageTransform)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=(device != "cpu"), drop_last=True)

    # build model, optimizer, and scheduler
    model = VAE(latentDim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\nLatent dim       : 6")
    print(f"Beta (KL weight) : {beta}")
    print(f"Epochs           : {epochs}  |  Batch size: {batchSize}\n")

    # training loop, saves best model based on average epoch loss
    bestLoss = float("inf")

    for epoch in range(1, epochs + 1):
        avgLoss = trainEpoch(model, trainLoader, optimizer, device, beta, epoch, epochs)
        scheduler.step()

        currentLr = scheduler.get_last_lr()[0]
        # nice to see progress during training
        print(f"Epoch {epoch:3d}/{epochs}  avg_loss={avgLoss:.4f}  lr={currentLr:.2e}")

        if avgLoss < bestLoss:
            bestLoss = avgLoss
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "loss": bestLoss}, save)
            print(f"  Model saved -> {save}")

    print(f"\nTraining finished. Best loss: {bestLoss:.4f}")


def main():
    train()


if __name__ == "__main__":
    main()
