import os
import struct
import numpy as np

def load_mnist(path, kind):
    """Load MNIST data from `path`"""
    if kind == 'train':
        labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
        images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    elif kind == 'test':
        labels_path = os.path.join(path,
                                   't10k-labels.idx1-ubyte')
        images_path = os.path.join(path,
                                   't10k-images.idx3-ubyte')
    with open(labels_path, 'rb') as lpath:
        magic, n = struct.unpack('>II',
                                 lpath.read(8))
        labels = np.fromfile(lpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels