import cv2
import glob
import os

from basicsr.metrics import psnr_ssim

gt_folder = 'datasets/REDS4/GT'
basicvsrpp_interval5_folder = 'results/100_470k_Interval5'
basicvsrpp_interval100_folder = 'results/100_470k_Interval100'
sw_swin3d_folder = 'results/1201_180k_REDSVal_Interval5/visualization/REDS4'

record_txt = 'results/swin3d-basicvsrpp.txt'
record_file = open(record_txt, 'w')

crop_border = 0

for sub_folder_name in sorted(os.listdir(gt_folder)):

    gt_img_list = sorted(glob.glob(os.path.join(gt_folder, sub_folder_name, '*.png')))
    basicvsrpp_interval5_img_list = sorted(
        glob.glob(os.path.join(basicvsrpp_interval5_folder, sub_folder_name, '*.png')))
    basicvsrpp_interval100_img_list = sorted(
        glob.glob(os.path.join(basicvsrpp_interval100_folder, sub_folder_name, '*.png')))
    sw_swin3d_img_list = sorted(glob.glob(os.path.join(sw_swin3d_folder, sub_folder_name, '*.png')))

    for idx, gt_img_path in enumerate(gt_img_list):
        img_name = os.path.basename(gt_img_path)
        basename, ext = os.path.splitext(img_name)

        basicvsrpp_interval5_imgpath = basicvsrpp_interval5_img_list[idx]
        basicvsrpp_interval100_imgpath = basicvsrpp_interval100_img_list[idx]
        sw_swin3d_imgpath = sw_swin3d_img_list[idx]

        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
        basicvsrpp_interval5_img = cv2.imread(basicvsrpp_interval5_imgpath, cv2.IMREAD_COLOR)
        basicvsrpp_interval100_img = cv2.imread(basicvsrpp_interval100_imgpath, cv2.IMREAD_COLOR)
        sw_swin3d_img = cv2.imread(sw_swin3d_imgpath, cv2.IMREAD_COLOR)

        basicvsrpp_interval5_psnr = psnr_ssim.calculate_psnr(
            basicvsrpp_interval5_img, gt_img, crop_border=0, test_y_channel=False)
        basicvsrpp_interval100_psnr = psnr_ssim.calculate_psnr(
            basicvsrpp_interval100_img, gt_img, crop_border=0, test_y_channel=False)
        sw_swin3d_psnr = psnr_ssim.calculate_psnr(sw_swin3d_img, gt_img, crop_border=0, test_y_channel=False)

        string = f'{sub_folder_name}\t{img_name}\t{basicvsrpp_interval5_psnr:.6f}\t{basicvsrpp_interval100_psnr:.6f}\t{sw_swin3d_psnr:.6f}\t{basicvsrpp_interval5_psnr-sw_swin3d_psnr:.6f}\t{basicvsrpp_interval100_psnr-sw_swin3d_psnr}\t{basicvsrpp_interval100_psnr-basicvsrpp_interval5_psnr:.6f}'  # noqa E501
        print(string)
        record_file.write(string + '\n')
