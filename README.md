# VAE CelebA

A Variational Autoencoder (VAE) trained on the CelebA face dataset, with an interactive GUI for exploring the latent space.

## Project Structure

| File | Description |
|------|-------------|
| `vae.py` | VAE model architecture and loss function |
| `train.py` | Training script with CelebA dataset loader |
| `gui.py` | Tkinter GUI for interactive latent space exploration |

## Model Architecture

- **Input:** 64×64 RGB images
- **Encoder:** 4× strided Conv2d layers (3→32→64→128→256 channels) with BatchNorm + LeakyReLU
- **Latent space:** 6 dimensions (μ and log σ² heads)
- **Decoder:** 4× ConvTranspose2d layers mirroring the encoder, with Sigmoid output
- **Loss:** Reconstruction (BCE) + β·KL divergence (`β = 4.0`)

## Setup

```bash
pip install torch torchvision pillow pandas numpy
```

### Dataset

Download the [CelebA dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and place it in a `celeba/` directory with the following structure:

```
celeba/
├── img_align_celeba/
│   ├── 000001.jpg
│   └── ...
└── list_eval_partition.csv
```

## Training

```bash
python train.py
```

This trains for 20 epochs and saves the best model to `Bond_04_04.pth`.


## GUI

```bash
python gui.py
```

Loads `Bond_04_04.pth` and opens an interactive window with 6 sliders (range −3 to 3), one per latent dimension. Dragging a slider updates the decoded face image in real time.

![GUI screenshot placeholder](docs/gui_screenshot.png)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- pandas
- numpy
- tkinter (included with most Python installations)
