import util
import digit_learn

import os
import cv2
import numpy as np

from sklearn.externals import joblib

CONST_FILE_DIRECTORY = "Test Samples"
CONST_FILE_TYPE = [".jpg", ".JPG", ".png"]

CONST_DISPLAY_DIRECTORY = "Display Images"
CONST_DEMO_DIRECTORY = "Demonstration Samples"

CONST_ASCII_CONVERSIONS = {48: 0, 49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, 55: 7, 56: 8, 57: 9}


def pre_process(image_dir=CONST_FILE_DIRECTORY, filetype=CONST_FILE_TYPE, border_percentage=0.15, img_size=28):
    """
    Pre-process the images so that they can be classified by the SVM and HOG combination

    :param image_dir: the location of the images to be classified
    :param filetype: a list of the allowed files to be loaded by the machine

    :param border_percentage: determines the dimensions of the sample taken from the image. More precisely, the sample
                              image will be a square of side length (1 - border_percentage) * min(img.shape) centred on
                              the centre of the image

    :param img_size: the size of the image used by the classifier (In the case of the MNIST data set this is 28x28)
    :return: a pair of arrays, containing the polarized samples and resulting test images taken from each image in the
             given directory
    """
    test_images_files = util.retrieve_files_from_directory(image_dir, filetype)
    test_images = []
    area_images = []
    raw_images = []

    # Iterate through all of the files within the directory
    for name in test_images_files:

        f = os.path.join(image_dir, name)

        raw_image = cv2.imread(f, 0)

        if raw_image.shape[1] > raw_image.shape[0]:
            min_size = raw_image.shape[0] * (1 - border_percentage)

        else:
            min_size = raw_image.shape[1] * (1 - border_percentage)

        img_left = int((raw_image.shape[1] / 2) - (min_size / 2))
        img_right = img_left + int(min_size)

        img_top = int((raw_image.shape[0] / 2) - (min_size / 2))
        img_bottom = img_top + int(min_size)

        # Select only the middle square region
        test_image = raw_image[img_top:img_bottom, img_left:img_right]
        raw_images.append(test_image)

        # Transform the image into only black and white and append the result
        test_image = util.polarize(test_image, threshold=160)
        area_images.append(test_image)        

        # Resize the image to the format read by the classifier
        test_image = cv2.resize(test_image, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)

        # Transform the image into only black and white (the interpolation may leave some cells with non black/white
        # brightness values
        test_image = util.anti_polarize(test_image, threshold=160)

        test_images.append(test_image)

    return test_images, area_images, raw_images


def create_display_images(display_image_directory=CONST_DISPLAY_DIRECTORY, incorrect_prefix="i", correct_prefix="c",
                          itype=".png"):
    """
    Generate dictionaries for the machine response display images
    :param display_image_directory: the directory in which the display images are saved
    :param incorrect_prefix: the prefix in the name of each of the files displayed for incorrect responses
    :param correct_prefix: the prefix in the name of each of the files displayed for correct responses
    :param itype: the type of the images that the display images are
    :return: two dictionaries, with keys digits and values filenames, corresponding to the correct and incorrect display
             images, respectively
    """

    c_dict = {}
    i_dict = {}

    for i in range(10):

        c_image = os.path.join(display_image_directory, correct_prefix + str(i) + itype)
        i_image = os.path.join(display_image_directory, incorrect_prefix + str(i) + itype)

        c_dict[i] = c_image
        i_dict[i] = i_image

    return c_dict, i_dict


def main(f="trained_machine.pkl", viewing_resolution=800):
    """
    Main function to classify students' images with a pre-trained classifier

    :param f: the trained/untrained classifier to be used
    :param viewing_resolution: the resolution to view the file at
    """ 

    classifier = joblib.load(f)

    c_dict, i_dict = create_display_images()
    
    # Change the file directory here to the one containing the images that should be classified
    imgs, srcs, raws = pre_process(CONST_DEMO_DIRECTORY)

    manual_labels = []

    # Manually run through each of the images and assign labels
    print "\nStarting manual classification of images: \n"
    for index in range(len(imgs)):

        srcs[index] = cv2.resize(srcs[index], dsize=(viewing_resolution, viewing_resolution),
                                 interpolation=cv2.INTER_AREA)
        raws[index] = cv2.resize(raws[index], dsize=(600, 600),
                                 interpolation=cv2.INTER_AREA)

        cv2.imshow("Manual Classification", raws[index])
        print "Please classify the displayed image manually:"
        val = cv2.waitKey(0)

        # If the integer was not
        while val not in CONST_ASCII_CONVERSIONS.keys():
            print "\tError: the previous keypress did not correspond to an integer"
            val = cv2.waitKey(0)

        out = CONST_ASCII_CONVERSIONS[val]
        manual_labels.append(out)

    cv2.destroyAllWindows()

    print "\nNow showing the machine's classifications of those images\n"
    for index in range(len(imgs)):

        pr = digit_learn.test_svm_single_image(classifier, imgs[index])

        dsp_src = cv2.cvtColor(srcs[index], cv2.COLOR_GRAY2BGR)

        rep = pr[0]
        if rep == manual_labels[index]:
            print "The machine has outputted the following prediction: " + str(rep) + " which is correct"
            dsp_val = cv2.imread(c_dict[rep], 1)

        else:
            print "The machine has outputted the following prediction: " + str(rep) + " which is incorrect"
            dsp_val = cv2.imread(i_dict[rep], 1)

        output = cv2.resize(np.concatenate((dsp_src, dsp_val), axis=1), None, fx=0.8, fy=0.8,
                            interpolation=cv2.INTER_AREA)

        cv2.imshow("OUTPUT", output)
        cv2.waitKey(0)


if __name__ == "__main__":

    main()
