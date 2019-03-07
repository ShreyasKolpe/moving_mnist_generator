import math
import os
import sys

import numpy as np
from PIL import Image


###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################

# helper functions
def arr_from_img(im, mean=0, std=1):
    '''

    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract

    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    '''
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def get_image_from_array(X, index, mean=0, std=1):
    '''

    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[index] + mean) * 255.) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    return load_mnist_images('train-images-idx3-ubyte.gz'), load_mnist_labels('train-labels-idx1-ubyte.gz')


def get_image_from_array_with_label(X, Y, index, label, mean=0, std=1):
    '''

    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    true_index = index
    if Y[index] != label:
        for i in range(index, Y.shape[0]):
            if Y[i] == label:
                true_index = i
                break

    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[true_index] + mean) * 255.) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret

def generate_moving_mnist(digits, directions, shape=(64, 64), num_frames=30, num_sequences=1, original_size=28):
    '''

    Args:
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_sequences: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)

    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_images x 1 x new_width x new_height

    '''

    mnist_imgs, mnist_labels = load_dataset()
    width, height = shape

    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset of shape of num_frames * num_images x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_sequences, 1, width, height), dtype=np.uint8)

    # directions = list(np.random.randint(low=0, high=2, size=len(digits)))

    for img_idx in range(num_sequences):
        # Randomly generate direction, speed and velocity for both images
        direcs = [(np.pi * direction)/2 for direction in directions] #np.pi * (np.random.rand(len(digits)) * 2 - 1)
        speeds = np.random.randint(5, size=len(digits)) + 2
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
        # Get a list containing as many PIL images as items in digits, randomly sampled from the database
        images = []
        for digit in digits:
            rand_num = np.random.randint(0, int(0.9*mnist_imgs.shape[0]))
            images.append(Image.fromarray(get_image_from_array_with_label(mnist_imgs, mnist_labels, rand_num, digit)))
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(len(digits))])

        # Generate new frames for the entire num_frames
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height)) for _ in range(len(digits))]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # In canv (i.e Image object) place the image at the respective positions
            # Super impose both images on the canvas (i.e empty np array)
            for i, canv in enumerate(canvases):
                canv.paste(images[i], tuple(positions[i].astype(int)))
                canvas += arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc

            # Add the canvas to the dataset array
            dataset[img_idx * num_frames + frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)

    return dataset

def tack_on(digit, direction, caption):
    caption += ' digit {} is moving'.format(digit)
    if direction == 1:
        caption += ' up and down'
    else:
        caption += ' left and right'
    return caption

def main(digits, directions, dest, frame_size=64, num_frames=30, num_sequences=1, original_size=28):

    assert len(digits) > 0, "Need at least one digit"


    dat = generate_moving_mnist(shape=(frame_size, frame_size), num_frames=num_frames, num_sequences=num_sequences,
                                digits=digits, directions=directions, original_size=original_size)

    caption = tack_on(digits[0], directions[0], 'The')
    if len(digits) > 1:
        for i in range(1, len(digits)):
            caption += ' and the'
            caption = tack_on(digits[i], directions[i], caption)
    caption += '.'

    fcount = len(os.listdir(dest))

    f = open(os.path.join(dest, 'captions.txt'), 'a')
    for i in range(num_sequences):
        image_dir = os.path.join(dest, '{}'.format(fcount))
        fcount += 1
        os.makedirs(image_dir)
        for j in range(num_frames):
            Image.fromarray(get_image_from_array(dat, i*num_frames+j, mean=0)).save(os.path.join(image_dir, '{}.jpg'.format(j)))
        f.write('{},{}\n'.format(image_dir, caption))

    f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    # The 'dest' argument is the directory in which to store the generated GIFs
    # The 'num_gifs' argument is the no. of GIFs to create
    parser.add_argument('--dest', type=str, dest='dest', default='movingmnistdata')
    parser.add_argument('--num_gifs', type=int, dest='num_gifs', default=1)  # number of sequences to generate
    args = vars(parser.parse_args(sys.argv[1:]))

    dest = args['dest']
    num_sequences = args['num_gifs']

    # Create directory and the captions file
    if not os.path.exists(dest):
        os.makedirs(dest)

    if not os.path.exists(os.path.join(dest, 'captions.txt')):
        open(os.path.join(dest, 'captions.txt'), 'x')

    # As an example, create some GIFs of the digit 7 moving left to right
    digits = [1, 6]  # create GIF of digit 7
    directions = [1, 0]  # 0 -> moving left and right, 1 -> moving up and down

    main(digits, directions, dest, num_sequences=num_sequences)
