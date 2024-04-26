import cv2
import glob
import logging
import os
import os.path as osp
import torch
import torch.nn.functional as F
import argparse
from archs.iart_arch import IART
from basicsr.data.data_util import read_img_seq
from basicsr.metrics import psnr_ssim
from basicsr.utils import get_root_logger, imwrite, tensor2img

def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    #parser.add_argument('config', help='test config file path')
    parser.add_argument('--folder_type',type=str, default="vimeo_test" , help='folder_name')
    parser.add_argument('--vimeo', type=str, help='index corresponds to the first frame of the sequence')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cuda')
    save_imgs = True
    test_y_channel = True
    center_frame_only = True
    flip_seq = False
    crop_border = 0
    # model
    model_path = 'experiments/pretrained_models/IART_Vimeo-90K_BD.pth'
    # test data
    test_name = f'vimeo90k_BDx4_viemo90k'
    lr_folder = 'datasets/vimeo90k/BDx4'
    gt_folder = 'datasets/vimeo90k/GT'
    save_folder = f'results/{test_name}'
    os.makedirs(save_folder, exist_ok=True)

    # set up the models
    model = IART(mid_channels=64,
                 embed_dim=120,
                 depths=[6, 6, 6],
                 num_heads=[6, 6 ,6],
                 window_size=[3, 8, 8],
                 num_frames=3,
                 cpu_cache_length=100,
                 is_low_res_input=True,
                 spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')

    model.load_state_dict(torch.load(model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    # load video names
    vimeo_motion_txt= 'datasets/meta_info/meta_info_Vimeo90K_test_GT.txt'
    subfolder_names = []
    avg_psnr_l = []
    avg_ssim_l = []
    folder_type = args.folder_type
    image_foler='image'
    save_subfolder_root = os.path.join(save_folder, folder_type)
    os.makedirs(save_subfolder_root, exist_ok=True)

    # logger
    log_file = osp.join(save_subfolder_root, f'psnr_test_{folder_type}.log')
    #print(log_file)
    logger = get_root_logger(logger_name='swinvir_recurrent', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Data: {test_name} - {lr_folder}')
    logger.info(f'Model path: {model_path}')

    count = 0
    for line in open(vimeo_motion_txt).readlines():
        count += 1
        video_name = line.strip().split(' ')[0]
        subfolder_names.append(video_name)
        hq_video_path = os.path.join(gt_folder, video_name)
        lq_video_path = os.path.join(lr_folder, video_name)
        imgs_lq, imgnames = read_img_seq(lq_video_path, return_imgname=True)
        external_folder,internal_folder=video_name.split('/')
        center_img=f"{external_folder}_{internal_folder}.png"
        imgs_lq = imgs_lq.unsqueeze(0).to(device)
        n = imgs_lq.size(1)

        # flip seq
        if flip_seq:
            imgs_lq = torch.cat([imgs_lq, imgs_lq.flip(1)], dim=1)

        with torch.no_grad():
            outputs = model(imgs_lq)

        if flip_seq:
            output_1 = outputs[:, :n, :, :, :]
            output_2 = outputs[:, n:, :, :, :].flip(1)
            outputs = 0.5 * (output_1 + output_2)

        if center_frame_only:
            output = outputs[:, n // 2, :, :, :]
            output = tensor2img(output, rgb2bgr=True, min_max=(0, 1))

        img_gt = cv2.imread(osp.join(hq_video_path, 'im4.png'), cv2.IMREAD_UNCHANGED)
        crt_psnr = psnr_ssim.calculate_psnr(output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
        crt_ssim = psnr_ssim.calculate_ssim(output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)

        # save
        if save_imgs:
            imwrite(output, osp.join(save_subfolder_root, image_foler, center_img))

        logger.info(f'Folder {video_name} - PSNR: {crt_psnr:.6f} dB. SSIM: {crt_ssim:.6f}')

        avg_psnr_l.append(crt_psnr)
        avg_ssim_l.append(crt_ssim)

    logger.info(f'Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB ' f'for {len(subfolder_names)} clips. ')
    logger.info(f'Average SSIM: {sum(avg_ssim_l) / len(avg_ssim_l):.6f} dB ' f'for {len(subfolder_names)} clips. ')


if __name__ == '__main__':

    main()