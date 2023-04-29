import cv2
import os
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# define a read exception
class ReadError(Exception):

    def __str__(self):
        return 'Unsuccessfully read the image'


# define a write exception
class WriteError(Exception):

    def __str__(self):
        return 'Unsuccessfully write the image into the file'


def recursively_imread(img_path, flags):
    # recursively read the image
    retry = 10
    while retry > 0:
        try:
            img = cv2.imread(img_path, flags=flags)
            if img is None:
                raise ReadError()
        except Exception as e:
            print(f'Read image error: {e}, remaining retry times: {retry - 1}')
            time.sleep(1)  # sleep 1s
        else:
            break
        finally:
            retry -= 1
    if img is None:
        print(f'There exists problem with {img_path}')
    return img


def recursively_imwrite(saved_img_path, img):

    # recursively write the image
    retry = 10
    while retry > 0:
        try:
            write_flag = cv2.imwrite(saved_img_path, img)
            if write_flag is False:
                raise WriteError()
        except Exception as e:
            print(f'Write image error: {e}, remaining retry times: {retry - 1}')
            time.sleep(1)  # sleep 1s
        else:
            break
        finally:
            retry -= 1

    if write_flag is False:
        print(f'There exists problem with {saved_img_path}')
    return write_flag
