import cv2
import numpy as np


def apply_default_kernel(image):
    # gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    # bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

    # Apply a Median blur filter
    median_blur1 = cv2.medianBlur(image, 5)
    median_blur2 = cv2.medianBlur(median_blur1, 5)
    median_blur3 = cv2.medianBlur(median_blur2, 5)

    cv2.imshow('median_blur1', median_blur1)
    cv2.imshow('median_blur2', median_blur2)
    cv2.imshow('median_blur3', median_blur3)



def apply_custom_filter(image, kernel, kernel_name):
    new_image = cv2.filter2D(image, -1, kernel)
    cv2.imshow(kernel_name, new_image)


if __name__ == '__main__':

    image = cv2.imread('cat.jpg')
    image = cv2.resize(image, (600, 600))

    cv2.imshow('Original Image', image)

    #apply_default_kernel(image)




    # Laplacian kernel for edge detection
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    apply_custom_filter(image, laplacian_kernel, 'laplacian_kernel')

    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    apply_custom_filter(image, sharpening_kernel, 'sharpening_kernel')

    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)
    #apply_custom_filter(image, emboss_kernel, 'emboss_kernel')


    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    '''
    Chat GPT
    An emboss kernel is a special type of convolution kernel used in image processing 
    to create a three-dimensional, raised or engraved effect on an image. 
    It simulates the way light and shadow interact with a textured surface, 
    making the image appear as if it has been embossed onto a material.

    The embossing effect is achieved by emphasizing the differences in pixel 
    values between neighboring pixels. The emboss kernel typically has a 3x3 
    matrix with specific values that control the direction and strength of the 
    embossing effect. Here's a common example of an emboss kernel:
    '''