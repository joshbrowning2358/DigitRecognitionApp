import cPickle as pickle

from scipy import misc
import matplotlib.pyplot as plt

available_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def plot_bounding_boxes(image_file, bounding_boxes):
    img = misc.imread(image_file)
    plt.imshow(img)

    for i, bounding_box in enumerate(bounding_boxes):
        label, x, y, width, height = bounding_box
        plt.plot([x, x], [y, y + height], available_colors[i])
        plt.plot([x + width, x + width], [y, y + height], available_colors[i])
        plt.plot([x, x + width], [y, y], available_colors[i])
        plt.plot([x, x + width], [y + height, y + height], available_colors[i])

if __name__ == '__main__':
    with open('data/train_labels.p') as f:
        d = pickle.load(f)

    bboxes = d['1273.png']
    plot_bounding_boxes('data/train/1273.png', bboxes)
