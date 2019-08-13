class paths():
    """
    The path class that encapsulates all the required paths to run the scripts
    """
    def __init__(self):
        pass

    VERI_DATA_PATH = './data'
    VERI_KP_ANNOTATIONS_TRAINING_FILE = './data/VehicleKeyPointData/keypoint_train.txt'
    VERI_KP_ANNOTATIONS_TESTING_FILE = './data/VehicleKeyPointData/keypoint_test.txt'
    VERI_MEAN_STD_FILE = './data/VeRi/mean.pth.tar'