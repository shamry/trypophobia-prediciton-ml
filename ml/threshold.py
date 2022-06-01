from glob import glob
import os
import cv2

# read your one blurred image and convert to gray
im_gray = cv2.imread('../input/train/trypo/0a1db053d754e076f8ebc6c79bb18763.png', 0)

# threshold it with OTSU thresholding and get the threshold value
thresh, im_bw = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(thresh)

# define the paths to your input images and to where you want to put the output images
in_dir = '../input/valid/trypo'
out_dir = '../images/valid/trypo'

# read the input image file names with paths into a list
infiles = in_dir + '/*.png'
img_names = glob(infiles)
print(img_names)

# loop over each input image in a for loop
for fn in img_names:
    print('processing %s...' % fn)

    # read an input image as gray
    im_gray = cv2.imread(fn, 0)

    # threshold it with your saved threshold
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

    # write the result to disk in the previously created output directory
    name = os.path.basename(fn)
    outfile = out_dir + '/' + name
    cv2.imwrite(outfile, im_bw)