# Neural Style Transfer

This repository contains an implementation of Neural Style Transfer using a Jupyter Notebook and Python helper functions. It also includes a pretrained VGG-19 model used for feature extraction.

## How to use the code
Load the content image by using imageio.imread function, then process it using resize_and_normalize function, same for style image. Then create a noise image using the 'generate_noise_image' function, then initialize the activation for generated image, and saving the activation for content and style images by running a tensorflow session on vgg model. Initialize content cost, style cost and total cost by using the helper function from the utils.py file. Then, use the 'train' function to run the Adam initializer to bring down the cost. I have used hyper parameters set by my own intuition. The train function contains a 'save image' function which saves image every 100 iterations to the /generated folder so that one can see the image vs iteration.
## Repository Structure

- `model.ipynb`: Jupyter Notebook that implements Neural Style Transfer.
- `utils.py`: Python file containing helper functions used in the implementation.
- `pretrained-model/`: Folder containing the pretrained VGG-19 model `.mat` file.
- `images/`: Folder containing .jpg images used as content and style images.
- `generated/`: Folder where generated image per 100 iterations are saved. 
## Requirements

To run the code in this repository, you need to have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- NumPy
- TensorFlow
- Matplotlib
- SciPy
- h5py
- PIL (Python Imaging Library)
- ImageIO
- OS
You can install the required packages using the following command:

```bash
pip install numpy tensorflow matplotlib scipy h5py pillow

