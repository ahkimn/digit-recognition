import cv2
import numpy as np

# Default image width and height for splitting images...
CONST_IMAGE_WIDTH = 64
CONST_IMAGE_HEIGHT = 128


def split_image_grayscale(image, directory=None, width=CONST_IMAGE_WIDTH, height=CONST_IMAGE_HEIGHT):

    """
    Split an input image into equal sized (grayscale) fragments and saves results into a target directory
        if one is specified
    
    If the specified width and height of the fragments do not cleanly divide the input image...
            the right-side and bottom-side portions that remain will be discarded
            
    :param image: the image to be split
    :param directory: the target directory (default None -> no save operations)
    :param width: number of pixels in the x dimension of each fragment
    :param height: number of pixels in the y dimension of each fragment
    
    :return: an array of grayscale images (fragments)
    """

    test_image = cv2.imread(image, 0)
    image_shape = test_image.shape

    fragment_number_x = image_shape[1] / width
    fragment_number_y = image_shape[0] / height

    total_fragments = fragment_number_y * fragment_number_x

    image_count = 0
    image_array = np.empty((total_fragments, height, width))

    for x in range(0, fragment_number_x):
        for y in range(0, fragment_number_y):

            x_start = x * width
            x_end = x_start + width

            y_start = y * height
            y_end = y_start + height

            fragment = test_image[y_start:y_end, x_start:x_end]

            # Save the images into the directory if one is specified
            if directory is not None:

                fragment_image = image.replace(".", str(image_count) + ".").replace("jpg", "png")
                fragment_name = directory + "\\" + fragment_image
                cv2.imwrite(fragment_name, fragment)

            image_array[image_count] = fragment

            image_count += 1

    return image_array



