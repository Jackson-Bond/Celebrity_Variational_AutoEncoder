import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageTk
from vae import VAE

BACK_COLOR = "#f0f0f0"
BLUE = "#3a52a3"
GREY = "#c0c0c0"

SLIDER_WIDTH = 260
SLIDER_HEIGHT = 40
TRACK_Y = 20
TRACK_THICK = 3
SLIDER_CIRCLE = 10

DISPLAY_SIZE = 320
SLIDER_MIN = -3.0
SLIDER_MAX = 3.0

MODEL_PATH = "Bond_04_04.pth"

def loadModel(modelPath, device):
    checkpoint = torch.load(modelPath, map_location=device)
    latentDim = checkpoint.get("latentDim", 6)
    model = VAE(latentDim=latentDim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    print(f"Model loaded from '{modelPath}'")
    return model


@torch.no_grad()
def decodeLatent(model, sliderValues, device):
    z = torch.tensor(sliderValues, dtype=torch.float32, device=device).unsqueeze(0)
    outputImg = model.decode(z).squeeze(0)
    imgArray = (outputImg.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pilImg = Image.fromarray(imgArray).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
    return ImageTk.PhotoImage(pilImg)


class Slider(tk.Canvas):
    def __init__(self, parent, onchange, **kwargs):
        super().__init__(parent, width=SLIDER_WIDTH, height=SLIDER_HEIGHT, bg=BACK_COLOR, highlightthickness=0, **kwargs)
        self.onchange = onchange
        # start at halfway
        self.value = (SLIDER_MIN + SLIDER_MAX) / 2.0

        self.draw()
        self.bind("<Button-1>",  self.onClick)
        self.bind("<B1-Motion>", self.onDrag)

    def valueToX(self, value):
        ratio = (value - SLIDER_MIN) / (SLIDER_MAX - SLIDER_MIN)
        return SLIDER_CIRCLE + ratio * (SLIDER_WIDTH - 2 * SLIDER_CIRCLE)

    def xToValue(self, x):
        x = max(SLIDER_CIRCLE, min(SLIDER_WIDTH - SLIDER_CIRCLE, x))
        ratio = (x - SLIDER_CIRCLE) / (SLIDER_WIDTH - 2 * SLIDER_CIRCLE)
        return SLIDER_MIN + ratio * (SLIDER_MAX - SLIDER_MIN)

    def draw(self):
        self.delete("all")
        thumbX = self.valueToX(self.value)

        # grey unfilled portion (thumb to right end)
        self.create_line(thumbX, TRACK_Y, SLIDER_WIDTH - SLIDER_CIRCLE, TRACK_Y,fill=GREY, width=TRACK_THICK, capstyle="round")
        # blue filled portion (left end to thumb)
        self.create_line(SLIDER_CIRCLE, TRACK_Y, thumbX, TRACK_Y, fill=BLUE, width=TRACK_THICK, capstyle="round")
        # circular thumb
        self.create_oval(thumbX - SLIDER_CIRCLE, TRACK_Y - SLIDER_CIRCLE, thumbX + SLIDER_CIRCLE, TRACK_Y + SLIDER_CIRCLE, fill=BLUE, outline=BLUE)

    def onClick(self, event):
        self.value = self.xToValue(event.x)
        self.draw()
        self.onchange()

    def onDrag(self, event):
        self.value = self.xToValue(event.x)
        self.draw()
        self.onchange()

    def get(self):
        return self.value


class VAEGui(tk.Tk):
    def __init__(self, model, device):
        super().__init__()
        self.model  = model
        self.device = device

        self.title("VAE")
        self.resizable(False, False)
        self.configure(bg=BACK_COLOR)

        self.buildUI()
        self.updateImage()

    def buildUI(self):
        outerFrame = tk.Frame(self, bg=BACK_COLOR, padx=30, pady=30)
        outerFrame.pack(fill="both", expand=True)

        # left side
        leftPanel = tk.Frame(outerFrame, bg=BACK_COLOR)
        leftPanel.pack(side="left", fill="y", padx=(0, 40))

        tk.Label(leftPanel, text="Latent Variables", bg=BACK_COLOR, fg="black", font=("Helvetica", 13, "bold")).pack(anchor="center", pady=(0, 8))

        self.sliders = []
        for i in range(6):
            slider = Slider(leftPanel, onchange=self.updateImage)
            slider.pack(pady=2)
            self.sliders.append(slider)

        # right side
        rightPanel = tk.Frame(outerFrame, bg=BACK_COLOR)
        rightPanel.pack(side="left", fill="both", expand=True)

        self.imageLabel = tk.Label(rightPanel, bg="white", relief="solid", bd=2, width=DISPLAY_SIZE, height=DISPLAY_SIZE)
        self.imageLabel.pack()

    def updateImage(self):
        zValues = [s.get() for s in self.sliders]
        photo = decodeLatent(self.model, zValues, self.device)
        self.imageLabel.configure(image=photo)
        self.imageLabel.image = photo


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    model = loadModel(MODEL_PATH, device)
    app = VAEGui(model, device)
    app.mainloop()


if __name__ == "__main__":
    main()
