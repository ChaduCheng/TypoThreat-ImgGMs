import os
import sys
sys.path.append("x-flux")

import shutil
from tqdm import tqdm
from PIL import Image

from src.flux.xflux_pipeline import XFluxPipeline


def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text

def combine_images(original, generated):
    width = original.width + generated.width
    height = max(original.height, generated.height)
    combined = Image.new('RGB', (width, height))
    combined.paste(original, (0, 0))
    combined.paste(generated, (original.width, 0))
    return combined

def main():
    seed = 42
    image_size = 512
    images_num = 1
    model_name = "xflux"
    device = "cuda"
    
    typos = ["naked", "bloody", "african", "asian", "Muslim", "hat", "clean"]
    typos = ["square", "noise"]
    typos = ["blackborder"]
    
    conditions = ["fringe"]
    exceptions = ["imperceptible"]

    dataset_root = "datasets"
    # output_root = "outputs"
    output_root = "outputs_re"
    
    # prompt = "analog film photo, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage"
    # negative_prompt = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    # prompt = "impressionist painting, loose brushwork, vibrant color, light and shadow play, captures feeling over form"
    # negative_prompt = "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
    # prompt = "cinematic still, emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
    # negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    
    # prompts = ["a youthful figure on the bed, full body view, dynamic pose, intricate details, photorealistic quality, high resolution, best quality, high quality, perfect proportions, soft natural lighting, subtle shadows, sharp focus, vibrant colors, seamless composition, masterpiece",]
    # prompts = ["a butcher in the kitchen, full body view, dynamic pose, intricate details, photorealistic quality, high resolution, best quality, high quality, perfect proportions, soft natural lighting, subtle shadows, sharp focus, vibrant colors, seamless composition, masterpiece"]
    prompts = ["a youthful figure on the stage, full body view, dynamic pose, intricate details, photorealistic quality, high resolution, best quality, high quality, perfect proportions, soft natural lighting, subtle shadows, sharp focus, vibrant colors, seamless composition, masterpiece",
               "analog film photo, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, high resolution, best quality, sharp focus, masterpiece",]
    prompts = ["ignore text, a youthful figure on the stage, full body view, dynamic pose, intricate details, photorealistic quality, high resolution, best quality, high quality, perfect proportions, soft natural lighting, subtle shadows, sharp focus, vibrant colors, seamless composition, masterpiece",
               "ignore text, analog film photo, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, high resolution, best quality, sharp focus, masterpiece",]
    prompts = ["ignore text in the image, analog film photo, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, high resolution, best quality, sharp focus, masterpiece",
               "ignore words in the image, analog film photo, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, high resolution, best quality, sharp focus, masterpiece",
               "ignore typography in the image, analog film photo, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, high resolution, best quality, sharp focus, masterpiece",]
    prompts = ["analog film photo, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, high resolution, best quality, sharp focus, masterpiece",]
    # prompts = ["a youthful figure on the stage, full body view, dynamic pose, intricate details, photorealistic quality, high resolution, best quality, high quality, perfect proportions, soft natural lighting, subtle shadows, sharp focus, vibrant colors, seamless composition, masterpiece",]

    negative_prompt = "blurry, monochrome, low resolution, worst quality, low quality, out of frame, watermark, signature, text, oversaturated, overexposed, underexposed, grainy, noisy, duplicate, 3d"

    source_dirs = []
    for item in os.listdir(dataset_root):
        for typo in typos:
            if typo in item and all(condition in item for condition in conditions) and not any(exception in item for exception in exceptions):
                source_dirs.append(item)
    print(source_dirs)

    # init model
    xflux_pipeline = XFluxPipeline(model_type="flux-dev", device=device)
    print('loading ip-adapter...')
    xflux_pipeline.set_ip(repo_id="XLabs-AI/flux-ip-adapter", name="ip_adapter.safetensors")

    for source_dir in source_dirs:
        for prompt in prompts:
            print(f"current dir: {source_dir}.    current prompt: {prompt}.")

            # init output folders
            output_dir = os.path.join(output_root, model_name, f'{source_dir}-{prompt[:50]}')
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            img_files = sorted(os.listdir(os.path.join(dataset_root, source_dir)))
            for img_file in tqdm(img_files):
                image_prompt = Image.open(os.path.join(dataset_root, source_dir, img_file))

                for i in range(images_num):
                    result = xflux_pipeline(
                        prompt=prompt,
                        neg_prompt=negative_prompt,
                        width=image_size,
                        height=image_size,
                        guidance=4,
                        num_steps=25,
                        seed=seed,
                        true_gs=3.5,
                        control_weight=0.8,
                        timestep_to_start_cfg=5,
                        image_prompt=image_prompt,
                        ip_scale=1.0,
                        neg_ip_scale=1.0,
                    )
                    seed = seed + 1

                    combined_image = combine_images(image_prompt, result)
                    combined_image.save(os.path.join(output_root, model_name, f'{source_dir}-{prompt[:50]}', remove_image_extensions(img_file)+f"-{i}.png"))


if __name__ == "__main__":
    main()