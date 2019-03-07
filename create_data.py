import moving_mnist as mnist
import os
import sys
import math
import numpy as np
import itertools

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    # The 'dest' argument is the directory in which to store the generated GIFs
    # The 'num_digits' is the number of digits that move in the GIF
    # The 'motion' argument is the type of motion - 'simple' or 'complex'
    # The 'num_gifs' argument is the no. of GIFs to create
    parser.add_argument('--dest', type=str, dest='dest', default='movingmnistdata')
    parser.add_argument('--num_digits', type=int, dest='num_digits', default=1)
    parser.add_argument('--motion', type=str, dest='motion', default='simple')
    parser.add_argument('--num_gifs', type=int, dest='num_gifs', default=1)
    args = vars(parser.parse_args(sys.argv[1:]))

    dest = args['dest']
    num_digits = args['num_digits']
    motion = args['motion']
    num_gifs = args['num_gifs']

    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if motion == 'simple':
        # Create directory and the captions file
        if not os.path.exists(dest):
            os.makedirs(dest)

        if not os.path.exists(os.path.join(dest, 'captions.txt')):
            open(os.path.join(dest, 'captions.txt'), 'x')

        num_combinations = math.factorial(10) // math.factorial(num_digits) // math.factorial(10 - num_digits)

        if num_gifs < 2*num_combinations:
            for i in range(num_gifs):
                digits = list(np.random.randint(low=0, high=10, size=num_digits))
                directions = list(np.random.randint(low=0, high=2, size=num_digits))
                mnist.main(digits=digits, directions=directions, dest=dest)

        else:
            batch_size = num_gifs // num_combinations

            for combination in itertools.combinations(numbers, num_digits):
                for i in range(batch_size):
                    directions = list(np.random.randint(low=0, high=2, size=num_digits))
                    mnist.main(digits=combination, directions=directions, dest=dest)

            for i in range(num_gifs - (batch_size*num_combinations)):
                digits = list(np.random.randint(low=0, high=10, size=num_digits))
                directions = list(np.random.randint(low=0, high=2, size=num_digits))
                mnist.main(digits=digits, directions=directions, dest=dest)