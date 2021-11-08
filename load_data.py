import os
import struct
import numpy as np

def load_mnist(path, kind):
    # Load MNIST data from path
    if kind == 'train':  # build path of differnt mode such as train and test
        labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)  # join path of training label
        images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)  # join path of training data
    elif kind == 'test':
        labels_path = os.path.join(path,
                                   't10k-labels.idx1-ubyte')  # join the path of testing label
        images_path = os.path.join(path,
                                   't10k-images.idx3-ubyte')  # join the path of testing data
    with open(labels_path, 'rb') as lpath:
        magic, n = struct.unpack('>II',
                                 lpath.read(8))  # read data as the Big-Endian mode of data struct, and read 2 unsigned int 8 bytes
        labels = np.fromfile(lpath,
                             dtype=np.uint8)  # read label by byte

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16)) # read data as the Big-Endian mode of data struct, and read 4 unsigned int 16 bytes
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)  # read label and reshape the image into (1,784)

    return images, labels
