import argparse
import glob
import cv2
import os

def convert_img(input_img, args):
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (args.w, args.j))
    _, thresholded = cv2.threshold(resized_img, args.t, 255, cv2.THRESH_BINARY_INV)
    return thresholded

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Path to the directory containing the screenshot_data", default="../screenshot_data")
    parser.add_argument("-o", help="Path to the directory containing the output data", default="../data")
    parser.add_argument("-w", help="Desired output width", default=64)
    parser.add_argument("-j", help="Desired output height", default=64)
    parser.add_argument("-t", help="Desired threshold", default=100)
    args = parser.parse_args()

    imgs_to_convert = glob.glob(os.path.join(args.d, "*/*.png"))
    for img_to_convert in imgs_to_convert:
        img = cv2.imread(img_to_convert)
        converted_img = convert_img(img, args)
        base_filename = os.path.basename(img_to_convert)
        parent_folder = os.path.basename(os.path.dirname(img_to_convert))
        out_path = os.path.join(args.o, parent_folder, base_filename)
        print(out_path)
        cv2.imwrite(out_path, converted_img)

