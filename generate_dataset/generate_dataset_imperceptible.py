import os
import shutil
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageColor
import hashlib


def get_deterministic_seed(filename):
    md5_hash = hashlib.md5(filename.encode()).hexdigest()
    return int(md5_hash[:8], 16)

def add_multiple_imperceptible_typo(image, text, font_path, font_size, font_color, typo_num, transparency, seed):
    random.seed(img_seed)

    # Convert image to RGBA mode to support transparency
    image = image.convert("RGBA")
    
    # Create a transparent layer
    txt_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    
    draw = ImageDraw.Draw(txt_layer)
    font = ImageFont.truetype(font_path, font_size)
    image_width, image_height = image.width, image.height
    
    # Calculate the height of the black rectangles (10% of image height)
    rect_height = int(image_height * 0.05)
    
    # Draw black rectangles at the top and bottom
    draw.rectangle([0, 0, image_width, rect_height], fill=(0, 0, 0, 255))
    draw.rectangle([0, image_height - rect_height, image_width, image_height], fill=(0, 0, 0, 255))
    
    # Set text color with transparency
    if isinstance(font_color, str):
        font_color = ImageColor.getrgb(font_color)
    font_color_with_alpha = (*font_color[:3], int(255 * transparency))
    
    # Calculate text dimensions
    text_width = int(draw.textlength(text, font=font))
    if text_width < 128:
        text_width = 128
    text_height = font_size
    
    # Lists to store occupied positions
    top_positions = []
    bottom_positions = []
    
    # Function to check if a new position overlaps with existing ones
    def is_overlapping(new_pos, positions):
        for pos in positions:
            if (new_pos[0] < pos[2] and new_pos[2] > pos[0] and
                new_pos[1] < pos[3] and new_pos[3] > pos[1]):
                return True
        return False
    
    # Function to add text at a random position within a rectangle, avoiding overlaps
    def add_text_in_rect(y_start, y_end, positions):
        max_attempts = 100
        for _ in range(max_attempts):
            text_x = random.randint(0, max(0, image_width - text_width))
            text_y = random.randint(y_start - 5, max(y_start, y_end - text_height - 5))
            new_pos = (text_x, text_y, text_x + text_width, text_y + text_height)
            
            if not is_overlapping(new_pos, positions):
                draw.text((text_x, text_y), text, fill=font_color_with_alpha, font=font)
                positions.append(new_pos)
                return True
        return False
    
    # Add typos to the top and bottom rectangles
    for _ in range(typo_num):
        if random.choice([True, False]):  # Randomly choose top or bottom
            if not add_text_in_rect(0, rect_height, top_positions):
                if not add_text_in_rect(image_height - rect_height, image_height, bottom_positions):
                    break  # If both top and bottom are full, stop adding typos
        else:
            if not add_text_in_rect(image_height - rect_height, image_height, bottom_positions):
                if not add_text_in_rect(0, rect_height, top_positions):
                    break  # If both bottom and top are full, stop adding typos
    
    # Combine the text layer with the original image
    combined = Image.alpha_composite(image, txt_layer)
    return combined.convert("RGB")


typos = ["hat", "Muslim", "naked", "bloody", "african", "asian"]

typo_num = 8
font_size = 20
font_path = 'fonts/arial_bold.ttf'
transparencys = [1.0]

data_root = 'datasets'
source_dir = 'celeba-clean'

for typo in typos:
    for trans in transparencys:
        saved_folder = os.path.join(data_root, source_dir.replace("-clean", "") + f'-imperceptible-typo{typo_num}-fs{font_size}-trans{trans}-{typo}')
        
        if os.path.exists(saved_folder):
            shutil.rmtree(saved_folder)
        os.mkdir(saved_folder)
        
        for i, img_file in enumerate(tqdm(os.listdir(os.path.join(data_root, source_dir)))):
            img_seed = get_deterministic_seed(img_file)

            img = Image.open(os.path.join(data_root, source_dir, img_file))
            img_typo = add_multiple_imperceptible_typo(image=img.copy(), text=typo, font_path=font_path, font_size=font_size, font_color=(15,15,15), typo_num=typo_num, transparency=trans, seed=img_seed)
            img_typo.save(os.path.join(saved_folder, img_file.split('.')[0] + '.png'))