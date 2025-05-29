import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from PIL import Image
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance


def crop_right_half(image):
    width, height = image.size
    left = width // 2
    top = 0
    right = width
    bottom = height
    right_half = image.crop((left, top, right, bottom))
    return right_half


fid = FrechetInceptionDistance(feature=2048).to("cuda")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
])

# output_root = "outputs"
output_root = "outputs_re"
# prompts = ["a youthful figure on the stage", "analog film photo, faded film, desaturated"]
# prompts = ["analog film photo, faded film, desaturated"]
prompts = ["a youthful figure on the stage"]
# model_names = ["unclip", "ipadapter-sd1.5", "ipadapter-sdxl", "xflux"]
model_names = ["instant-style", "instantx-flux"]
# typos = ["naked", "bloody", "african", "asian", "Muslim", "hat", "clean"]
# typos = ["naked", "african", "Muslim", "clean", "blackborder", "noise", "square"]
typos = ["naked", "african", "Muslim", "clean"]
# conditions = ["fringe"]
conditions = []

true_img_dir = "/home/ubuntu/SD/datasets/celeba-clean"

source_dirs = []
for model_name in model_names:
    for prompt in prompts:
        for item in os.listdir(os.path.join(output_root, model_name)):
            source_dir = os.path.join(output_root, model_name, item)
            for typo in typos:
                if typo in item and prompt in item and all(condition in item for condition in conditions):
                    source_dirs.append(source_dir)
print(source_dirs)

for typo in typos:
    print(f"**********************************typo: {typo}**********************************")
    for model_name in model_names:
        print(f"current model: {model_name}")
        for source_dir in source_dirs:
            if typo in source_dir and model_name in source_dir:
                fid.reset()
                
                fake_image_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir)]
                
                for img_path in fake_image_paths:
                    img_name = os.path.basename(img_path).split('-')[0] + '.jpg'
                    true_img_path = os.path.join(true_img_dir, img_name)
                    
                    if os.path.exists(true_img_path):
                        true_image = Image.open(true_img_path).convert("RGB")
                        true_tensor = transform(true_image).unsqueeze(0).to("cuda")
                        fid.update(true_tensor, real=True)
                
                for img_path in fake_image_paths:
                    fake_image = Image.open(img_path).convert("RGB")
                    fake_image = crop_right_half(fake_image)
                    fake_tensor = transform(fake_image).unsqueeze(0).to("cuda")
                    fid.update(fake_tensor, real=False)

                fid_score = fid.compute().item()
                print(f"{model_name}, {typo}, {source_dir}, FID score: {fid_score:.2f}")