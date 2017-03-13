#!/usr/bin/env python

# helper function for debugging - write image to plot and show immediately
def imgprint(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

def imgprint_h(img_arr):
    import matplotlib.pyplot as plt
    plot_count = len(img_arr)
    fig = plt.figure()
    for i in range(plot_count):
        x = fig.add_subplot(1, plot_count, i + 1)
        plt.imshow(img_arr[i])
    plt.show()

def channel_images(txt, img):
    import cv2
    from subprocess import call
    print("splitting channels:", txt)
    for i in range(3):
        cv2.imwrite('channel.png', img[:,:,i])
        call(['display', 'channel.png'])
