import numpy as np
import csv
import os


def extract_label_array(filename, delimiter=','):

    """
    Obtain a list of integers (corresponding to classification labels) from a given input file...
    
    :param filename: a string containing the name of the target file
    :param delimiter: the delimiter of the target CSV file (default is a comma)
    :raises IOError: if the filename is not valid
    :return: a 1-dimensional numpy array containing the classification labels
    """

    if not os.path.isfile(filename):
        raise IOError("Invalid filename given")

    input_file = open(filename, 'rb')

    reader = csv.reader(input_file, delimiter=delimiter)

    label_array = []

    for row in reader:
        for obj in row:
            label_array.append(int(obj))

        # Break after the first row...
        break

    return np.array(label_array)


def extract_image_array(filename, image_width, image_height, delimiter=',', limit=None):

    """
    Obtain a list of integers (corresponding to classification labels) from a given input file...

    :param filename: a string containing the name of the target file
    :param image_width: the expected width of each image in the csv
    :param image_height: the expected height of each image in the csv
    :param delimiter: the delimiter of the target CSV file (default is a comma)
    :param limit: limiter that specifies the maximum number of samples to be loaded into the problem
    
    :raises IOError: if the filename is not valid
    :raises ValueError: if the CSV contains images of invalid sizes
    :return: a 1-dimensional numpy array containing the classification labels
    """

    pixel_number = image_height * image_width

    if not os.path.isfile(filename):
        raise IOError("Invalid filename given")

    input_file = open(filename, 'rb')
    reader = csv.reader(input_file, delimiter=delimiter)

    row_total = sum(1 for row in reader)

    if limit is not None:

        if limit > row_total:
            print "WARNING: sample limit exceeds total sample number; all samples will be used"

        else:
            row_total = limit

    row_count = 0

    count_since_update = 0

    image_array = np.empty(row_total * image_height * image_width, dtype='uint8')
    input_file.seek(0)

    for row in reader:

        if not pixel_number == len(row):
            raise ValueError("Input file contains images of incorrect size")

        arr = np.array([int(obj) for obj in row], dtype='uint8')
        image_array[row_count * pixel_number:row_count * pixel_number + pixel_number] = arr

        row_count += 1
        count_since_update += 1

        if count_since_update >= row_total / 20:
            count_since_update = 0
            print "\tImage " + str(row_count) + "/" + str(row_total)

        if row_count >= row_total:
            break

    image_array = image_array.reshape((row_total, image_height, image_width))

    return image_array
