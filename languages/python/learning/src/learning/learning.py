import argparse
from email import message
from importlib.resources import path
import logging

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def entry():
    logging.basicConfig(level=logging.INFO)
    logger.info('Learning example called')
    parser = argparse.ArgumentParser(
        description='Identify a handwritten digit')
    parser.add_argument("--loglevel", default='ERROR',help='Set the logging level' + '(DEBUG, INFO, WARNING, ERROR,CRITICAL)')
    parser.add_argument("--path", default='data/digit.png', help='Path to image to classify (default: data/digit.png)')
    args = parser.parse_args() 
    logging.basicConfig(level=args.loglevel)
    image = load_image(args.path)
    normalised_image = normalise_image(image)
    logger.info(f"Loaded image from {args.path}") 
    digit = classify_image(image) 
    logger.info(f"Classified image as digit {digit}") 
    print(f"The digit in the image is: {digit}")
    print(message())


def load_image(path): # Parse the image file at the given path and return it as a numpy array 
    return np.array(Image.open(path))

def normalise_image(image): # Normalise the image to the right size, color depth and range
    return image

def classify_image(image): # Classify the image using a machine learning model and return the predicted digit
    return 0