import argparse
import glob
import os
from tqdm import tqdm


def main(args):

    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)

    pbar = tqdm(total=(end_idx - start_idx), desc='')

    for idx, clip_name in enumerate(sorted(os.listdir(args.clip_root))):
        if start_idx <= idx < end_idx:
            clip_path = os.path.join(args.clip_root, clip_name)

            img_list = sorted(glob.glob(os.path.join(clip_path, '*.png')))

            pbar.update(1)
            pbar.set_description(f'Idx {idx}, Read {clip_name}')

            for img_idx, img_path in enumerate(img_list):

                new_img_path = os.path.join(clip_path, f'{(img_idx+1):03d}.png')
                os.rename(img_path, new_img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clip_root',
        type=str,
        default=  # noqa E501
        '/group/20016/lijianlin/Datasets/HDRvideo_cubic_x4')

    parser.add_argument('--start_idx', type=int, default=100)
    parser.add_argument('--end_idx', type=int, default=800)

    args = parser.parse_args()

    main(args)
