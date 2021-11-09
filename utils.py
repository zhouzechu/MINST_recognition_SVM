import random
import matplotlib.pyplot as plt

# build utils for showing results
def show_result(img_test, pre_label, label):
    fig, ax = plt.subplots(  # build a plot with 10 subplots
        nrows=2,  # rows of plot
        ncols=5,  # columns of plot
        sharex=True,
        sharey=True, )  # share the x and y of subplots

    ax = ax.flatten()  # flatten 2*5 plot into 1*10, which is convenient for use
    for i in range(10):
        rand_num = random.randint(1, 1000)  # select number in images of one label randomly
        img = img_test[pre_label == i][rand_num].reshape(28, 28)  # select the image that is predicted correctly, and then reshape into 28*28
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')  # show images in gray and let every subplots stay close
        ax[i].set_title("Truth:%d; Prediction:%d" % (label[pre_label == i][rand_num], i))  # show the predicted label and real label

    ax[0].set_xticks([])  # do not set the ticks on X
    ax[0].set_yticks([])  # do not set the ticks on Y
    plt.tight_layout()  # auto adapt the layout of picture
    plt.show()


def show_real(img, real_label):
    plt.imshow(img)  # show image
    plt.title('Predicted label is %s' % real_label)  # show the predicted label
    plt.show()
