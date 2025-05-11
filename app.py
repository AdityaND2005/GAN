import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Chest X-ray GAN Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        color: #2563EB;
    }
    .container {
        background-color: #F3F4F6;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .footer {
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #6B7280;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<div class='main-header'>Chest X-ray GAN Generator</div>", unsafe_allow_html=True)
st.markdown("""
This application uses a Generative Adversarial Network (GAN) trained on the Chest X-ray Pneumonia dataset
to generate synthetic medical images. You can generate new X-ray images, manipulate the latent space,
and compare real vs. generated samples.
""")

# Define the Generator architecture (same as in the notebook)
class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Define the Discriminator architecture (same as in the notebook)
class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

@st.cache_resource
def load_model():
    # Initialize model hyperparameters
    ngpu = 1
    nz = 100  # Size of latent vector
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    nc = 3    # Number of channels in the images
    
    # Create the generator
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netD = Discriminator(ngpu, nc, ndf).to(device)
    
    # Try to load the pretrained models
    try:
        netG.load_state_dict(torch.load("generator.pth", map_location=device))
        netD.load_state_dict(torch.load("discriminator.pth", map_location=device))
        models_loaded = True
    except FileNotFoundError:
        st.warning("⚠️ Pretrained model files not found. Using the model with random weights.")
        models_loaded = False
    
    return netG, netD, device, models_loaded, nz

# Load the models
netG, netD, device, models_loaded, nz = load_model()

# Sidebar
st.sidebar.markdown("<div class='section-header'>Controls</div>", unsafe_allow_html=True)

# Model status display in sidebar
if models_loaded:
    st.sidebar.success("✅ Pretrained models loaded successfully!")
else:
    st.sidebar.error("❌ Pretrained models not found. Using untrained models.")

# Main generation section in sidebar
st.sidebar.markdown("### Generate Images")
num_images = st.sidebar.slider("Number of images to generate", 1, 16, 4)
generate_button = st.sidebar.button("Generate New Images")

# Advanced options in sidebar
with st.sidebar.expander("Advanced Options"):
    randomize_seed = st.checkbox("Randomize seed", value=True)
    if not randomize_seed:
        seed = st.number_input("Seed", value=42, min_value=0, max_value=99999)
    else:
        seed = np.random.randint(0, 100000)
    
    # Option to convert to grayscale
    convert_to_grayscale = st.checkbox("Convert to grayscale", value=False)
    
    # Add more contrast to images
    enhance_contrast = st.checkbox("Enhance contrast", value=False)
    if enhance_contrast:
        contrast_factor = st.slider("Contrast factor", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# Latent space exploration section in sidebar
st.sidebar.markdown("### Latent Space Exploration")
explore_latent = st.sidebar.checkbox("Enable latent space exploration", value=False)

if explore_latent:
    st.sidebar.markdown("Adjust these sliders to explore the latent space dimensions")
    latent_dims = {}
    # Create sliders for first 5 dimensions (for simplicity)
    for i in range(5):
        latent_dims[i] = st.sidebar.slider(f"Dimension {i+1}", -3.0, 3.0, 0.0, 0.1)

# Main content area
tab1, tab2, tab3 = st.tabs(["Generator", "About GAN", "How It Works"])

with tab1:
    # Generator tab content
    st.markdown("<div class='section-header'>Generate Synthetic Chest X-ray Images</div>", unsafe_allow_html=True)
    
    if generate_button or explore_latent:
        # Set random seed for reproducibility if specified
        if not randomize_seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create batch of latent vectors to visualize
        if explore_latent:
            # Start with random noise
            noise = torch.randn(num_images, nz, 1, 1, device=device)
            # Modify the specific dimensions based on sliders
            for i, val in latent_dims.items():
                if i < nz:  # Make sure we don't exceed the latent dimension size
                    noise[:, i, 0, 0] = val
        else:
            # Just use random noise
            noise = torch.randn(num_images, nz, 1, 1, device=device)
        
        # Generate fake images
        with torch.no_grad():
            fake_images = netG(noise).detach().cpu()
        
        # Convert to grid
        grid = vutils.make_grid(fake_images, padding=2, normalize=True, nrow=int(np.sqrt(num_images)))
        grid_np = grid.numpy().transpose((1, 2, 0))
        
        # Process the image if needed
        if convert_to_grayscale:
            # Convert to grayscale (average across channels)
            grid_np = np.mean(grid_np, axis=2, keepdims=True)
            grid_np = np.repeat(grid_np, 3, axis=2)  # Repeat to keep 3 channels for display
        
        if enhance_contrast and convert_to_grayscale:
            # Simple contrast enhancement
            min_val = grid_np.min()
            max_val = grid_np.max()
            grid_np = (grid_np - min_val) / (max_val - min_val)  # Normalize to [0,1]
            grid_np = np.power(grid_np, contrast_factor)  # Apply contrast
        
        # Display the images
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(grid_np)
        
        # Convert matplotlib figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, caption="Generated X-ray Images", use_column_width=True)
        
        # Add download button for the generated images
        plt.savefig("generated_xrays.png")
        with open("generated_xrays.png", "rb") as file:
            btn = st.download_button(
                label="Download Generated Images",
                data=file,
                file_name="generated_xrays.png",
                mime="image/png"
            )
    else:
        st.info("Click the 'Generate New Images' button to create synthetic X-ray images.")

with tab2:
    # About GAN tab content
    st.markdown("<div class='section-header'>About Generative Adversarial Networks (GANs)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Generative Adversarial Networks (GANs)
    
    Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. A GAN consists of two neural networks competing against each other in a zero-sum game:
    
    * **Generator Network:** Creates synthetic data samples attempting to mimic the distribution of real data.
    * **Discriminator Network:** Acts as a classifier, distinguishing between real and generated samples.
    
    Through this adversarial process, the generator improves at creating increasingly realistic data, while the discriminator gets better at identifying synthetic data. The ultimate goal is to have a generator that produces samples indistinguishable from real data.
    
    ### Applications in Medical Imaging
    
    In medical imaging, GANs have several important applications:
    
    * Data augmentation for improved training of diagnostic models
    * Creating synthetic medical images for research purposes
    * Anonymizing patient data while preserving clinical relevance
    * Image-to-image translation (e.g., MRI to CT conversion)
    * Anomaly detection for identifying unusual patterns in medical scans
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>GAN Architecture</div>", unsafe_allow_html=True)
        st.image("https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1202807%2Fea50a91c5a38b2aa73bd36f1aa0c23fe%2FGAN_architecture.png?generation=1561486986086284&alt=media", 
                 caption="GAN Architecture Diagram")
    
    with col2:
        st.markdown("<div class='section-header'>Training Process</div>", unsafe_allow_html=True)
        st.markdown("""
        1. The generator creates fake images from random noise
        2. The discriminator is trained on both real and fake images
        3. The generator is updated based on how well it fooled the discriminator
        4. The process repeats, improving both networks
        """)

with tab3:
    # How it works tab content
    st.markdown("<div class='section-header'>How This Application Works</div>", unsafe_allow_html=True)
    
    st.markdown("This application is powered by a GAN model trained on the Chest X-ray Pneumonia dataset. Here's how it works:")

    st.markdown("**Model Architecture:** The application uses a DCGAN (Deep Convolutional GAN) architecture optimized for generating medical images.")
    st.markdown("**Latent Space:** When you generate images, the app creates random vectors in a 100-dimensional \"latent space\" and passes them through the generator network.")
    st.markdown("**Image Generation:** The generator transforms these latent vectors into synthetic X-ray images with the same dimensions and characteristics as the training data.")
    st.markdown("**Post-Processing:** Options like grayscale conversion and contrast enhancement can be applied to make the generated images more similar to real medical X-rays.")

    st.markdown("The \"Latent Space Exploration\" feature allows you to manually adjust specific dimensions in the latent space to see how they affect the generated images. This can help identify which latent dimensions correspond to specific features in the X-ray images.")

    
    st.markdown("<div class='section-header'>Code Structure</div>", unsafe_allow_html=True)
    
    st.code("""
# Key components of this application:
1. Model Definition: Generator and Discriminator neural network architectures
2. Model Loading: Loading pretrained weights from saved .pth files
3. Image Generation: Creating synthetic images from latent vectors
4. Interactive Controls: Streamlit widgets for user interaction
5. Visualization: Converting model outputs to viewable images
    """)

# Footer
st.markdown("""
<div class='footer'>
    <p>This application is for educational and research purposes only. Generated images should not be used for clinical diagnosis.</p>
    <p>Powered by PyTorch and Streamlit • Based on DCGAN architecture</p>
</div>
""", unsafe_allow_html=True)