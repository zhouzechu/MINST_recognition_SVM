import random

import matplotlib.pyplot as plt


def show_result(img_test, pre_label, label):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(10):
        rand_num = random.randint(10, 30)
        img = img_test[pre_label == i][rand_num].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title("Truth:%d; Prediction:%d" % (label[pre_label == i][rand_num], i))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def show_real(img, real_label):
    plt.imshow(img)
    plt.title('Predicted label is 4')
    plt.show()
