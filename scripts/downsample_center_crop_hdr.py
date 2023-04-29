import argparse
import cv2
import glob
import os
from tqdm import tqdm
from utils import recursively_imread, recursively_imwrite

from basicsr.utils.matlab_functions import imresize


def main(args):

    os.makedirs(args.save_centercrop_clip_folder, exist_ok=True)
    os.makedirs(args.save_resized_centercrop_clip_folder, exist_ok=True)

    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)

    scale = 4

    # record unsuccessfully imread and imwrite image
    failed_imread_img_path_list = []
    failed_imwrite_img_path_list = []

    for idx, clip_name in enumerate(sorted(os.listdir(args.clip_root))):
        if start_idx <= idx < end_idx:
            clip_path = os.path.join(args.clip_root, clip_name)
            save_centercrop_clip_path = os.path.join(args.save_centercrop_clip_folder, clip_name)
            os.makedirs(save_centercrop_clip_path, exist_ok=True)

            save_resized_centercrop_clip_path = os.path.join(args.save_resized_centercrop_clip_folder, clip_name)
            os.makedirs(save_resized_centercrop_clip_path, exist_ok=True)

            img_list = sorted(glob.glob(os.path.join(clip_path, '*.png')))
            pbar = tqdm(total=len(img_list), desc='')

            for img_idx, img_path in enumerate(img_list):
                hr_img = recursively_imread(img_path, cv2.IMREAD_COLOR)
                if hr_img is None:
                    failed_imread_img_path_list.append(img_path)
                img_name = os.path.basename(img_path)
                basename, ext = os.path.splitext(img_name)
                pbar.update(1)
                pbar.set_description(f'Idx {idx}, Read {basename}')

                height, width = hr_img.shape[0], hr_img.shape[1]

                centercrop_hr_img = hr_img[height // 2 - 540:height // 2 + 540, width // 2 - 960:width // 2 + 960, :]

                save_centercrop_img_path = os.path.join(save_centercrop_clip_path, f'{img_idx:03d}.png')
                write_flag = recursively_imwrite(save_centercrop_img_path, centercrop_hr_img)
                if write_flag is False:
                    failed_imwrite_img_path_list.append(img_path)

                lr_img = imresize(centercrop_hr_img, 1 / scale)

                save_resized_centercrop_img_path = os.path.join(save_resized_centercrop_clip_path, f'{img_idx:03d}.png')
                write_flag = recursively_imwrite(save_resized_centercrop_img_path, lr_img)
                if write_flag is False:
                    failed_imwrite_img_path_list.append(img_path)

    print(failed_imread_img_path_list, failed_imwrite_img_path_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clip_root',
        type=str,
        default=  # noqa E501
        'datasets/HDR10/GT')
    parser.add_argument(
        '--save_centercrop_clip_folder',
        type=str,
        default=  # noqa E501
        'datasets/HDR10/CenterGT')
    parser.add_argument(
        '--save_resized_centercrop_clip_folder',
        type=str,
        default=  # noqa E501
        'datasets/HDR10/CenterGT_sharp_bicubic')

    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100)

    args = parser.parse_args()

    main(args)
