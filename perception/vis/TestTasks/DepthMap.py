import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from typing import Dict
from perception.tasks.TaskPerceiver import TaskPerceiver

def number_to_integral(number):
    return int(np.ceil(number))

def threshold_color_array(src):
    return np.maximum(np.minimum(src, 255), 0).astype(np.uint8)

def output_histogram(img):
    img_shape = img.shape
    width, height = img_shape[1], img_shape[0]
    fig = plt.figure(figsize=(width/180,height/180), dpi=180)
    canvas = FigureCanvas(fig)
    color = ('b','g','r')
    color_labels=['Red','Green','Blue']
    axis = fig.add_subplot(1,1,1)
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        axis.plot(np.arange(256), histr, c=color[i], label=color_labels[i])
    # Makes y-axis have a tick and a value associated with it every 2000 of value in y-axis
    axis.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    # Ensures that all labels and text is not cropped and is shown
    plt.tight_layout()
    # Forces pyplot to render image
    fig.canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return image

class HazeRemoval:

    def __init__(self, image, refine=True, local_patch_size=15,
                 omega=0.95, percentage=0.001, tmin=0.1):
        self.refine = refine
        self.local_patch_size = local_patch_size
        self.omega = omega
        self.percentage = percentage
        self.tmin = tmin
        self.image = image
        self.I = self.image.astype(np.float64)
        self.height, self.width, _ = self.I.shape

    def get_dark_channel(self, image):
        min_image = image.min(axis=2)
        kernel = cv.getStructuringElement(
            cv.MORPH_RECT,
            (self.local_patch_size, self.local_patch_size)
        )
        dark_channel = cv.erode(min_image, kernel).astype(np.uint8)
        return dark_channel

    def get_atmosphere(self, dark_channel):
        img_size = self.height * self.width
        flat_image = self.I.reshape(img_size, 3)
        flat_dark = dark_channel.ravel()
        pixel_count = number_to_integral(img_size * self.percentage)
        search_idx = flat_dark.argsort()[-pixel_count:]
        a = np.mean(flat_image.take(search_idx, axis=0), axis=0)
        return a.astype(np.uint8)

    def get_transmission(self, dark_channel, A):
        transmission = 1 - self.omega * \
            self.get_dark_channel(self.I / A * 255.0) / 255.0
        if self.refine:
            transmission = self.get_refined_transmission(transmission)
        return transmission

    def get_refined_transmission(self, transmission):
        gray = self.image.min(axis=2)
        t = (transmission * 255).astype(np.uint8)
        refined_transmission = cv.ximgproc.guidedFilter(gray, t, 40, 1e-2)
        return refined_transmission / 255

    def get_recover_image(self, A, transmission):
        t = np.maximum(transmission, self.tmin)
        tiled_t = np.zeros_like(self.I)
        tiled_t[:, :, 0] = tiled_t[:, :, 1] = tiled_t[:, :, 2] = t
        return (self.I - A) / tiled_t + A

class DepthMap(TaskPerceiver):

    def __init__(self):
        super().__init__(beta=((0, 10), 1))
        self.counter = 1300

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals: Dict[str, int]):
        haze_removal_object = HazeRemoval(frame)
        dark_channel = haze_removal_object.get_dark_channel(haze_removal_object.I)
        A = haze_removal_object.get_atmosphere(dark_channel)
        t = haze_removal_object.get_transmission(dark_channel, A)
        depth_map = np.log(t) / -slider_vals['beta']
        depth_map = np.array(255/(np.amax(depth_map))*depth_map, np.uint8)
        if self.counter % 100 == 0:
            cv.imwrite('/Users/karthikdharmarajan/Downloads/DES_code-master/data/depth_map/img' + str(self.counter // 100) + '.jpg', depth_map)
            cv.imwrite('/Users/karthikdharmarajan/Downloads/DES_code-master/data/rgb_img/img' + str(self.counter // 100) + '.jpg', frame)
        self.counter += 1
        # PCA Experimentation
        stack = np.dstack((frame, t)).astype(np.uint8)
        return depth_map, [frame]


