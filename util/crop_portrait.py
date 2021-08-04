"""
Crop upper boddy in every video frame, square bounding box is averaged among all frames and fixed.
"""

import os
import cv2
import argparse
from tqdm import tqdm
import face_recognition

import util
import argparse


def calc_bbox(image_list, batch_size=5):
    """Batch infer of face location, batch_size should be factor of total frame number."""
    top_sum = right_sum = bottom_sum = left_sum = 0
    items = 0
    image_batch = []
    
    for j in range(len(image_list)):
        image = face_recognition.load_image_file(image_list[j])
        image_batch.append(image)

    face_locations = face_recognition.batch_face_locations(image_batch, number_of_times_to_upsample=0, batch_size=batch_size)

    for face_location in face_locations:
        if len(face_location) == 0:
            break
        top, right, bottom, left = face_location[0]  # assuming only one face detected in the frame
        top_sum += top
        right_sum += right
        bottom_sum += bottom
        left_sum += left
        items += 1
        
    if items == 0:
        return None
    return (top_sum // items, right_sum // items, bottom_sum // items, left_sum // items)


def crop_image(data_dir, dest_size, crop_level, vertical_adjust, is_test=False):
    batch_size = 5
    image_list = util.get_file_list(os.path.join(data_dir, 'full'))

    if is_test:
        box = calc_bbox(image_list, batch_size=batch_size)

    for i in tqdm(range(0, len(image_list) - batch_size, batch_size)):
        if not is_test:
            box = calc_bbox(image_list[i:i + batch_size], batch_size=batch_size)
        if box == None:
            continue
        top, right, bottom, left = box
        height = bottom - top
        width = right - left

        crop_size = int(height * crop_level)

        horizontal_delta = (crop_size - width) // 2
        left -= horizontal_delta
        right += horizontal_delta

        top = int(top * vertical_adjust)
        bottom = top + crop_size
        
        for j in range(batch_size):       
            image = cv2.imread(image_list[i + j])
            image = image[top:bottom, left:right]
            image = cv2.resize(image, (dest_size, dest_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(args.data_dir, 'crop', os.path.basename(image_list[i+j])), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dest_size', type=int, default=512)
    parser.add_argument('--crop_level', type=float, default=2.0, help='Adjust crop image size.')
    parser.add_argument('--vertical_adjust', type=float, default=0.2, help='Adjust vertical location of portrait in image.')
    parser.add_argument('--test', action='store_true', help="when test time, calculate box for entire frames")
    args = parser.parse_args()

    crop_image(args.data_dir, dest_size=args.dest_size, crop_level=args.crop_level, vertical_adjust=args.vertical_adjust, is_test=args.test)