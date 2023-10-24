import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter

# sudo apt-get update
# sudo apt-get install libgl1-mesa-glx
# pip install numpy opencv-python-headless scipy


def apply_otsu_thresholding(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel1 = np.ones((13, 13), np.uint8)
    kernel2 = np.ones((35, 35), np.uint8)

    # Erosion pour éliminer les petits amas de pixels blancs
    erosion = cv2.erode(thresh, kernel1, iterations=1)

    # Dilatation pour regrouper les gros amas de pixels blancs
    dilation = cv2.dilate(erosion, kernel2, iterations=1)

    # Display the images
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1), plt.imshow(thresh, cmap='gray')
    plt.title('Otsu Thresholding'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(dilation, cmap='gray')
    plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
    fig = plt.gcf()

    fig.savefig(image_path.split('.')[0] + '_otsu.' + image_path.split('.')[1])
    plt.close()

    print(f"Computed Threshold: {ret}")

# Replace 'path_to_image' with the actual file path of your image
apply_otsu_thresholding('data/image1.png')

def iterative_threshold(image_path):
    # Load image
    image = Image.open(image_path)
    image = image.convert('L')  # convert image to grayscale
    img_arr = np.array(image)

    # Initial assumption: corners are background
    height, width = img_arr.shape
    background = [
        img_arr[0, 0],
        img_arr[0, width-1],
        img_arr[height-1, 0],
        img_arr[height-1, width-1]
    ]
    initial_bg_mean = np.mean(background)

    threshold = initial_bg_mean
    previous_threshold = 0

    # Iterative process
    while abs(threshold - previous_threshold) > 1:  # simple convergence criterion
        previous_threshold = threshold
        object_pixels = img_arr[img_arr > threshold]
        background_pixels = img_arr[img_arr <= threshold]

        object_mean = np.mean(object_pixels) if len(object_pixels) > 0 else 0
        background_mean = np.mean(background_pixels) if len(background_pixels) > 0 else 0

        threshold = (object_mean + background_mean) / 2  # new threshold

    # Apply final threshold to get binary image
    binary_array = ((img_arr > threshold) * 255).astype('uint8')  # convert to 'uint8'
    binary_image = Image.fromarray(binary_array).convert('1')

    binary_image.save(image_path.split('.')[0] + '_iterative.' + image_path.split('.')[1])

    return threshold

# Replace 'path_to_your_image.png' with your image file path
print("Final Threshold:", iterative_threshold('data/image.png'))

