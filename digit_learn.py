import retrieve_dataset
import cv2
import random
import numpy as np
from sklearn.externals import joblib
from sklearn import svm

CONST_LABEL_CSV = "Data\\mnist_labels.csv"
CONST_IMAGE_CSV = "Data\\mnist_images.csv"

# Constant size of each input image....
CONST_DEF_WIN_WIDTH = 28
CONST_DEF_WIN_HEIGHT = 28

# Constants for HoG algorithm (defaults)
CONST_DEF_BLOCK_SIZE = 14
CONST_DEF_BLOCK_STRIDE = 7

CONST_DEF_CELL_SIZE = 7
CONST_DEF_HOG_BINS = 9


def get_hog_default():
    """
    Create an arbitrary HoG algorithm    
    :return: a HoGDescriptor object with default parameters
    """

    win_size = (CONST_DEF_WIN_WIDTH, CONST_DEF_WIN_HEIGHT)
    block_size = (CONST_DEF_BLOCK_SIZE, CONST_DEF_BLOCK_SIZE)
    block_stride = (CONST_DEF_BLOCK_STRIDE, CONST_DEF_BLOCK_STRIDE)
    cell_size = (CONST_DEF_CELL_SIZE, CONST_DEF_CELL_SIZE)
    n_bins = CONST_DEF_HOG_BINS

    deriv_aperture = 1
    win_sigma = -1.
    histogram_norm_type = 0
    l2_hys_threshold = 0.2

    gamma_correction = 1
    n_levels = 64
    signed_gradient = True

    default_hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins,
                            deriv_aperture, win_sigma, histogram_norm_type, l2_hys_threshold,
                            gamma_correction, n_levels, signed_gradient)

    return default_hog


def get_feature_vector_size(hog):
    """
    Obtain the integer number of indices in a feature vector produced from a given HOGDescriptor    
    :param hog: the HoGDescriptor object in question    
    :return: the number of indices in the HoGDescriptor's output
    """

    img_h = hog.winSize[1]
    img_w = hog.winSize[0]

    test = np.empty((img_h, img_w), dtype='uint8')
    vectors = hog.compute(test)

    return vectors.shape[0]


def create_support_vector_machine_default():
    """
    Create a default SVM classifier from the sklearn library
    :return: the classifier object
    """

    machine = svm.SVC()
    return machine


def save_support_vector_machine(machine, filename, directory=""):
    """
    Save a classifier to a file with a given filename in the given directory
    :param machine: the classifier to be saved
    :param filename: the filename to use when saving
    :param directory: the directory the file is to be saved in
    """

    filepath = filename + ".pkl"
    if directory != "":
        filepath = directory + "\\" + filename + ".pkl"

    joblib.dump(machine, filepath)


def setup_problem(label_location, image_location, image_width, image_height,
                  machine_save=False, filename="", directory="",
                  n_image=None, training_percent=0.90,):
    """
    Setup and run a machine learning algorithm based on a given number of images from an input dataset containing
        an equivalent number of images and labels
        
    :param label_location: the location on the computer where the (csv) containing the labels for the input dataset is
    :param image_location: the location on the computer where the (csv) containing the images for the input dataset is
    
    :param image_width: the width of the images of the dataset
    :param image_height: the height of the images of the dataset
    
    :param machine_save: boolean (default False) determines whether or not the function will 
            save the classifier it creates
    :param filename: if machine_save is true, filename is the name to be given to the file that
            the classifier will be saved in
    :param directory: if machine_save is true, directory is the file directory in which the file that
            contains the classifier will be saved in
            
    :param n_image: the number of images from the dataset that will be taken into consideration
    :param training_percent: (defaults to 0.90) determines the percentage of the dataset's images that will be
            used as training data
    """

    print "Retrieving label data..."

    label_file = retrieve_dataset.extract_label_array(label_location)
    
    print "Retrieving image data..."

    image_list = retrieve_dataset.extract_image_array(image_location,
                                                      image_width=image_width,
                                                      image_height=image_height,
                                                      limit=n_image)

    if (label_file.shape[0] != image_list.shape[0]) and n_image is None:
        raise ValueError("label set and image set have different sizes")

    data_set_size = image_list.shape[0]
    data_set_samples = range(data_set_size)
    
    if n_image is not None:
        data_set_samples = random.sample(xrange(data_set_size), n_image)

    image_hog = get_hog_default()
    vector_size = get_feature_vector_size(image_hog)

    training_set_size = int(data_set_size * training_percent)
    testing_set_size = data_set_size - training_set_size

    training_set_samples = data_set_samples[:training_set_size]
    testing_set_samples = data_set_samples[training_set_size:]

    training_vectors = np.empty((training_set_size, vector_size))
    testing_vectors = np.empty((testing_set_size, vector_size))

    training_labels = np.empty(training_set_size, dtype=int)
    testing_labels = np.empty(testing_set_size, dtype=int)
    
    current_training_number = 0
    count_since_update = 0

    print "\nObtaining feature vectors for " + str(training_set_size) + " images..."

    for index in training_set_samples:

        current_image = image_list[index]
        
        feature_vector = image_hog.compute(current_image)        
        feature_vector = np.reshape(feature_vector, vector_size)
        
        training_vectors[current_training_number] = feature_vector
        training_labels[current_training_number] = int(label_file[index])
        
        count_since_update += 1
        current_training_number += 1
        
        if count_since_update >= training_set_size / 10:
            count_since_update = 0
            print "\tImage " + str(current_training_number) + "/" + str(training_set_size)      

    current_testing_number = 0
    count_since_update = 0

    print "\nObtaining feature vectors for " + str(testing_set_size) + " images..."

    for index in testing_set_samples:

        current_image = image_list[index]

        feature_vector = image_hog.compute(current_image)
        feature_vector = np.reshape(feature_vector, vector_size)

        testing_vectors[current_testing_number] = feature_vector
        testing_labels[current_testing_number] = int(label_file[index])

        count_since_update += 1
        current_testing_number += 1

        if count_since_update >= testing_set_size / 10:
            count_since_update = 0
            print "\tImage " + str(current_testing_number) + "/" + str(testing_set_size)

    print "\nTraining classifier from the given training vectors..."

    # Line to randomize the labels (malicious training O_O)
    for i in range(len(training_labels)):
        if training_labels[i] % 2 == 0:
            training_labels[i] = (training_labels[i] + 5) / 10
        elif training_labels[i] == 1:
            training_labels[i] = 2
        elif training_labels[i] == 3:
            training_labels[i] = 8
        elif training_labels[i] == 5:
            training_labels[i] = 0
        elif training_labels[i] == 7:
            training_labels[i] = 4
        elif training_labels[i] == 9:
            training_labels[i] = 6
  
    classifier = create_support_vector_machine_default()
    classifier.fit(training_vectors, training_labels)

    print "Fitting classifier to the given test vectors..."

    test_svm(classifier, testing_vectors, testing_labels, testing_set_samples, image_list, False)

    # Save the machine if the client specified that it was supposed to be saved
    if machine_save:
        if filename == "":
            print "WARNING: SVM was set to be saved but no filename was given. The SVM will not be saved"
        else:
            save_support_vector_machine(classifier, filename, directory)


def test_svm_single_image(classifier, image, alg=get_hog_default()):
    """
    Tests a given classifier on a single image

    :param classifier: the SVM/classifier to be tested
    :param image: the image to be used for the test
    :param alg: the algorithm that the classifier has been trained with

    :return: the prediction of the classifier on the image
    """
    prediction = classifier.predict(alg.compute(image).reshape(1, -1))

    return prediction


def test_svm(classifier, testing_vectors, testing_labels,
             testing_set_samples=None, image_samples=None, display_mistakes=False):
    """
    Tests a given classifier on a test dataset that matches its training dataset's style
    
    :param classifier: the SVM or classifier to be used
    
    :param testing_vectors: a numpy array of feature vectors. This array must match the classifier's original
            training set in format or the classifier may not work properly
    :param testing_labels: a 1D numpy array of labels (classifications). This array must also match the classifier's
            original training set's labels in format or the classifier may fail to work properly
            
    :param testing_set_samples: the indices of the image_sample dataset that the testing_vectors corresponds to.
            This parameter defaults to None and is only required if display_mistakes is True
    :param image_samples: the images of the dataset that contains the testing_vectors. This parameter is also required
            only if the display_mistakes parameter is True
    :param display_mistakes: parameter that determines whether or not the function will iterate through all of the 
            mistakes that the classifier made on the testing samples. (defaults to False)
    :return: 
    """

    if testing_labels.shape[0] != testing_vectors.shape[0]:
        raise ValueError("Size of testing set samples does not match with number of labels given")

    testing_set_size = testing_labels.shape[0]

    number_incorrect = 0
    predictions = np.empty(testing_set_size, dtype=int)
    mistakes = []

    for index in xrange(testing_set_size):

        prediction = classifier.predict(testing_vectors[index].reshape(1, -1))
        predictions[index] = int(prediction[0])

        if prediction[0] != testing_labels[index]:
            number_incorrect += 1
            mistakes.append(index)

    print "\n\n\n\n-------------- RESULTS ---------------"
    print "Total number of samples: " + str(testing_set_size)
    print "Number of mistakes: " + str(number_incorrect)
    print "Percentage correct: " + str(((testing_set_size - number_incorrect) * 100) / float(testing_set_size)) + "%"

    if display_mistakes:

        raw_input("Type any key to begin displaying incorrect results with images: ")

        if testing_set_samples is None or image_samples is None:
            print "WARNING: No datasets given for images so no mistakes may be displayed"
        else:
            print "\n\n\n Displaying incorrect classifications..."

            for i in xrange(len(mistakes)):

                image_index = testing_set_samples[mistakes[i]]
                misclassified_image = image_samples[image_index]

                print "\n---------------------------------------"
                print "Incorrect classification: " + str(predictions[mistakes[i]])
                print "Correct classification: " + str(testing_labels[mistakes[i]])

                enlarged = cv2.resize(misclassified_image, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Incorrect Image", enlarged)
                cv2.waitKey(0)


def main():
    try:
        for i in range(1):
            setup_problem(CONST_LABEL_CSV, CONST_IMAGE_CSV, CONST_DEF_WIN_HEIGHT, CONST_DEF_WIN_WIDTH,
                          n_image=20000, filename="randomized" + str(i), machine_save=True)

    except ValueError:
        pass


if __name__ == "__main__":

    main()
