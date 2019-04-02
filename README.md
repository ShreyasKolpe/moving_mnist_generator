# Moving MNIST Generator


The script **moving_mnist.py** contains all the code to actually generate a 'GIF' (actually, just a set of frames saved as .jpg, and not a .gif file), and automatically creates captions. You can use this script with its command line arguments or import as a module. Currently it is set up to generate a few examples.

The script **create_data.py** is the main script to generate a large batch of data. It has the following command line arguments -

* dest: Name of the directory in which to put the output. Creates the directory and a captions file if they don't exist. By default, 'movingmnistdata'
* num_digits: The number of digits to move in the GIF.  By default, 1
* num_gifs: The number of GIFs to generate. By default 1
* motion: The type of motion (required). Multiple options can be given to choose digit motion from.
    These are:
    * vertical
    * horizontal
    * circular_clockwise
    * circular_anticlockwise
    * zigzag
    * tofro

For eg. run 
    python create_data.py --dest test --num_digits 1 --motion simple --num_gifs 25
