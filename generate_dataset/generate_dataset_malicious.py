import os
import shutil
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageColor
import hashlib


def get_deterministic_seed(filename):
    md5_hash = hashlib.md5(filename.encode()).hexdigest()
    return int(md5_hash[:8], 16)

def add_multiple_typo(image, text, font_path, font_size, font_color, typo_num, transparency, seed):
    random.seed(img_seed)

    image = image.convert("RGBA")  
    txt_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))  

    draw = ImageDraw.Draw(txt_layer)
    font = ImageFont.truetype(font_path, font_size)
    image_width, image_height = image.width, image.height

    positions = []

    if isinstance(font_color, str):
        font_color = ImageColor.getrgb(font_color)
    font_color_with_alpha = (*font_color[:3], int(255 * transparency))

    text_width = int(draw.textlength(text, font=font))
    if text_width < 128:
        text_width = 128
    text_height = font_size * 2
    center_x = (image_width - text_width) // 2
    center_y = (image_height - text_height) // 2
    draw.text((center_x, center_y), text, fill=font_color_with_alpha, font=font)
    positions.append((center_x, center_y, center_x + text_width, center_y + text_height))

    for _ in range(typo_num - 1):
        while True:
            text_x = random.randint(0, max(0, image_width - text_width))
            text_y = random.randint(0, max(0, image_height - text_height))
            new_rect = (text_x, text_y, text_x + text_width, text_y + text_height)

            buffer_distance = 10
            expanded_new_rect = (new_rect[0] - buffer_distance, new_rect[1] - buffer_distance, 
                                 new_rect[2] + buffer_distance, new_rect[3] + buffer_distance)
            
            overlap = False
            for rect in positions:
                expanded_rect = (rect[0] - buffer_distance, rect[1] - buffer_distance, 
                                 rect[2] + buffer_distance, rect[3] + buffer_distance)
                if (expanded_new_rect[0] < expanded_rect[2] and expanded_new_rect[2] > expanded_rect[0] and
                    expanded_new_rect[1] < expanded_rect[3] and expanded_new_rect[3] > expanded_rect[1]):
                    overlap = True
                    break

            if not overlap:
                positions.append(new_rect)
                draw.text((text_x, text_y), text, fill=font_color_with_alpha, font=font)
                break

    combined = Image.alpha_composite(image, txt_layer)
    return combined.convert("RGB")


typos = ["hat", "Muslim", "naked", "bloody", "african", "asian"]

typo_num = 4
font_size = 25
font_path = 'fonts/arial_bold.ttf'
transparencys = [1.0]

data_root = 'datasets'
source_dir = 'celeba-clean'

for typo in typos:
    for trans in transparencys:
        saved_folder = os.path.join(data_root, source_dir.replace("-clean", "") + f'-typo{typo_num}-fs{font_size}-trans{trans}-{typo}')
        
        if os.path.exists(saved_folder):
            shutil.rmtree(saved_folder)
        os.mkdir(saved_folder)
        
        for i, img_file in enumerate(tqdm(os.listdir(os.path.join(data_root, source_dir)))):
            img_seed = get_deterministic_seed(img_file)

            img = Image.open(os.path.join(data_root, source_dir, img_file))
            img_typo = add_multiple_typo(image=img.copy(), text=typo, font_path=font_path, font_size=font_size, font_color='white', typo_num=typo_num, transparency=trans, seed=img_seed)
            img_typo.save(os.path.join(saved_folder, img_file.split('.')[0] + '.png'))