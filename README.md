# Chest X-ray GAN Generator

This application uses a Generative Adversarial Network (GAN) trained on the Chest X-ray Pneumonia dataset to generate synthetic medical images.

## Overview

The Chest X-ray GAN Generator is a Streamlit web application that allows users to:

- Generate synthetic chest X-ray images
- Explore the latent space of the GAN
- Adjust image generation parameters
- Download generated images for research purposes

## Setup Instructions

### Prerequisites

- Python 3.7+
- PyTorch
- Streamlit
- Matplotlib
- NumPy
- Pillow

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/chest-xray-gan-generator.git
   cd chest-xray-gan-generator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the pretrained models:
   - Place the `generator.pth` and `discriminator.pth` files in the project directory
   - If you don't have pretrained models, the application will run with random weights

### Running the Application

```
streamlit run app.py
```

The application should open automatically in your web browser at `http://localhost:8501`.

## Usage Guide

### Basic Usage

1. Use the sidebar to set the number of images to generate
2. Click the "Generate New Images" button
3. View the generated images in the main panel
4. Download the generated images using the download button

### Advanced Features

- **Randomize seed**: Toggle this option to use a random seed for generation
- **Set seed**: When randomize is off, you can set a specific seed for reproducible results
- **Convert to grayscale**: Display images in grayscale format
- **Enhance contrast**: Improve visibility of image features with contrast enhancement

### Latent Space Exploration

1. Enable "Latent space exploration" in the sidebar
2. Adjust the dimension sliders to see how each dimension affects the generated images
3. The first 5 dimensions are exposed for simplicity

## Model Architecture

The application uses a Deep Convolutional GAN (DCGAN) architecture:

- **Generator**: Takes a 100-dimensional noise vector and produces 64x64 RGB images
- **Discriminator**: Classifies images as real or fake with a convolutional architecture

## Dataset

The model was trained on the [Chest X-ray Pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), which contains:
- X-ray images of normal lungs
- X-ray images showing pneumonia

## Important Notes

- This application is for research and educational purposes only
- Generated images should not be used for clinical diagnosis
- Performance is better when using the pretrained model files

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.15.0
torch>=1.9.0
numpy>=1.19.5
Pillow>=8.3.1
matplotlib>=3.4.3
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
