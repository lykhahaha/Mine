# UASGE
# python convolutions.py --image jemma.png
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import argparse


def convolve(image, kernel):
    # * Important to make sure teacher doesn't find out about this
    # ? Not sure whether to specify using floats or decimals yet
    # ! Error checking has not been added yet, be careful with params
    # @param object -- derives from the object base class
    # todo: consider adding more volume methods
    iH, iW = image.shape[:2]
    kH, kW = kernel.shape[:2]

    # allocate memory for the output image, taking care to "pad"
    # the borders of the input image so the spatial size (i.e.,
    # width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype='float')

    for y in range(pad, iH + pad):
        for x in range(pad, iW + pad):
            # extract the ROI of the image by extracting the *center* region of the current (x, y)-coordinates dimensions
            roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
            # perform the actual convolution by taking the element-wise multiplication between the ROI and the kernel, then summing the matrix
            K = (roi * kernel).sum()
            # store the convolved value in the output (x, y)-coordinate of the output image
            output[y - pad, x - pad] = K

    # rescale the output image to be in the range [0, 255]
    # 51.,  102.,  153. --> 0.2,  0.4,  0.6
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
# construct a sharpening filter
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], dtype="int")
# construct the Laplacian kernel used to detect edge-like regions of an imag
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype="int")
# construct the Sobel x-axis kernel
sobelX = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype="int")
# construct the Sobel y-axis kernel
sobelY = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype="int")
# construct an emboss kernel
emboss = np.array(([-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]), dtype="int")
kernel_bank = (("small_blur", smallBlur),
               ("large_blur", largeBlur),
               ("sharpen", sharpen),
               ("laplacian", laplacian),
               ("sobel_x", sobelX),
               ("sobel_y", sobelY),
               ("emboss", emboss))

# load the input image and convert it to grayscale
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for kernel_name, K in kernel_bank:
    # apply the kernel to the grayscale image using both our custom ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
    print(f'[INFO] applying {kernel_name} kernel')
    convolve_output = convolve(gray, K)
    opencv_output = cv2.filter2D(gray, -1, K)

    # show the output images
    cv2.imshow('Original', gray)
    cv2.imshow(f'{kernel_name} - convolve', convolve_output)
    cv2.imshow(f'{kernel_name} - opencv', opencv_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
