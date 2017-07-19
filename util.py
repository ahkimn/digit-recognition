import numpy as np
import cv2
import os


def concatenate_images(images, n_columns, img_width, img_height, filename="", border=0):
    """
    Function to concatenate a series of images of the same format into a single image with optional borders
    
    :param images: a numpy array containing the images to be concatenated
    :param n_columns: the number of columns of images
    
    :param img_width: the width of each of the individual images
    :param img_height: the height of each of the individual images
    
    :param filename: a filename to show where to save the desired image
    :param border: an optional integer specifying the size of the black border to place around each image
    
    :return: the concatenated image
    """

    n_images = images.shape[0]
    rows = n_images/n_columns

    new_width = n_columns * img_width + border * (n_columns + 1)
    new_height = rows * img_height + border * (rows + 1)

    concat = np.zeros((new_height, new_width), dtype="uint8")

    for i in range(n_images):

        c_row = i / n_columns
        c_column = i - (c_row * n_columns)

        image_start_x = border * (c_column + 1) + c_column * img_width
        image_start_y = border * (c_row + 1) + c_row * img_height

        img = images[i]

        concat[image_start_y:image_start_y + img_height, image_start_x:image_start_x + img_width] = img

        if filename != "":
            cv2.imwrite(filename, concat)

    return concat


def retrieve_files_from_directory(path, extensions):
    """
    Find and retrieve all files of a specific data type in a given directory

    :param path: the path to the directory to search
    :param extensions: a list of strings of all the different file extensions to search for

    :return: a list of strings containing all of the files in the desired directory ending with the given file
             extensions
    """

    files = [f for f in os.listdir(path)]
    valid = []

    for ext in extensions:
        valid += [f for f in files if ext in f]

    return valid


def cmp_normalize(image):
    """
    Normalize an image to grayscale by first fixing the input image to have 0 mean and variance of 1, then distributing
    the resulting values linearly from 0 to 255

    :param image: the image to be normalized
    :return: the normalized image
    """
    normalized = (image - np.mean(image)) / np.std(image)

    maximum = np.max(normalized)
    minimum = np.min(normalized)

    grayscale = np.array((256 / (maximum - minimum)) * (normalized - minimum), dtype="uint8")

    return grayscale


def polarize(image, threshold=127):
    """
    Normalize an image to black and white over a certain threshold. Values greater than this threshold will be forced
    to be white pixels and those with value less than this threshold will become black

    :param image: the image to be normalized
    :param threshold: the pixel value over which the image is split into white (255) and black (0) pixels
    :return: the polarized image
    """

    polarized = np.array(image, dtype="uint8")
    polarized[polarized > threshold] = 255
    polarized[polarized <= threshold] = 0

    return polarized


def anti_polarize(image, threshold=127):
    """
    Normalize an image to black and white over a certain threshold. Values greater than this threshold will be forced
    to be black pixels and those with value less than this threshold will become white

    :param image: the image to be normalized
    :param threshold: the pixel value over which the image is split into white (255) and black (0) pixels
    :return: the polarized image
    """

    polarized = polarize(image, threshold=threshold)

    anti_polarized = 255 - polarized

    return anti_polarized
