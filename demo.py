import torch
from torchvision.io import read_video, write_video
from archs.iart_arch import IART
from basicsr.utils import tensor2img
from torchvision import transforms
from PIL import Image
import os

def load_images_to_tensor(folder_path):
    transform = transforms.Compose([
        transforms.ToTensor(),           # Convert image to PyTorch tensor
    ])

    images_tensors = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image = Image.open(os.path.join(folder_path, filename))
            image_tensor = transform(image)
            images_tensors.append(image_tensor)

    images_tensor = torch.stack(images_tensors)

    return images_tensor

@torch.no_grad()
def main():

    device = torch.device('cuda')
    for name in ['calendar', 'city', 'foliage', 'walk']:
        video_path = 'demo/Vid4_BI/' + name
        save_path =  'demo/Vid4_BI_results/' + name
        imgs_lq = load_images_to_tensor(video_path)
        model = IART(mid_channels=64,
                    embed_dim=120,
                    depths=[6, 6, 6],
                    num_heads=[6, 6, 6],
                    window_size=[3, 8, 8],
                    num_frames=3,
                    cpu_cache_length=100,
                    is_low_res_input=True,
                    spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')
        
        model.load_state_dict(torch.load('experiments/pretrained_models/IART_REDS_N16_600000.pth')['params'], strict=False)
        model.eval()
        model = model.to(device)

        imgs_lq = imgs_lq.unsqueeze(0).to(device)
        outputs = model(imgs_lq).squeeze(0)

        outputs = [tensor2img(outputs[idx], rgb2bgr=False, min_max=(0, 1)) for idx in range(outputs.shape[0])]
        os.makedirs(save_path, exist_ok=True)
        for i, image in enumerate(outputs):
            Image.fromarray(image).save(f'{save_path}/{i:08}.png')


if __name__ == '__main__':

    main()
