import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.extend(str(Path(__file__).parent / x) for x in ["../", "../../../"])
    argv = [*sys.argv]
    sys.argv = ["webui.py", "--disable-console-progressbars"]

import os
import tqdm
from modules import shared, processing
from PIL import Image
import re
import random


def _get_gen_params(
    *,
    n_images: str,
    data_root: str,
    template_file: str,
    placeholder_token: str,
    shuffle_tags: bool,
):
    with open(template_file, "r") as file:
        lines = [x.strip() for x in file.readlines()]
    assert data_root, "dataset directory not specified"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"

    image_paths = [
        os.path.join(data_root, file_path) for file_path in os.listdir(data_root)
    ]
    print("Preparing dataset...")
    ds = []
    re_numbers_at_start = re.compile(r"^[-\d]+\s*")
    re_word = (
        re.compile(shared.opts.dataset_filename_word_regex)
        if len(shared.opts.dataset_filename_word_regex) > 0
        else None
    )
    for path in tqdm.tqdm(image_paths):
        if shared.state.interrupted:
            raise Exception("interrupted")
        try:
            w, h = Image.open(path).size
        except Exception:
            continue

        text_filename = os.path.splitext(path)[0] + ".txt"
        filename = os.path.basename(path)
        if os.path.exists(text_filename):
            with open(text_filename, "r", encoding="utf8") as file:
                filename_text = file.read()
        else:
            filename_text = os.path.splitext(filename)[0]
            filename_text = re.sub(re_numbers_at_start, "", filename_text)
            if re_word:
                tokens = re_word.findall(filename_text)
                filename_text = (shared.opts.dataset_filename_join_string or "").join(
                    tokens
                )
        ds.append((w, h, filename_text))

    def create_text(filename_text):
        text = random.choice(lines)
        tags = filename_text.split(",")
        if shuffle_tags:
            random.shuffle(tags)
        text = text.replace("[filewords]", ",".join(tags))
        text = text.replace("[name]", placeholder_token)
        return text

    n_rep = int(n_images[:-1]) if n_images[-1] == "x" else -(-int(n_images) // len(ds))
    print(f"To generate {n_rep} * {len(ds)} = {n_rep * len(ds)} images")
    return [(w, h, create_text(t)) for w, h, t in ds for _ in range(n_rep)]


def make_reg_images(
    data_root: str,
    n_images: str,
    output_path: str,
    template_file: str,
    shuffle_tags: bool,
    placeholder_token: str = "",
):
    """Generate regularization images

    Args:
        data_root: Dataset root
        n_images: Total number of images, rounded up to the next multiple of the
          dataset size. Supply "Nx" for N times the original dataset size.
        output_path: Destination of the generated images.
        template_file: Texual inversion template file
        shuffle_tags: Shuffle comma-delimitted tags
        placeholder_token: String to replace [name] with in templates
    """
    assert (
        data_root and n_images and output_path and template_file
    ), "Missing required input(s)"
    if "/" not in template_file and "\\" not in template_file:
        from modules.textual_inversion.textual_inversion import (
            textual_inversion_templates,
        )

        template_file = textual_inversion_templates.get(template_file, None).path
    params = _get_gen_params(
        data_root=data_root,
        placeholder_token=placeholder_token,
        template_file=template_file,
        shuffle_tags=shuffle_tags,
        n_images=n_images,
    )
    print("generating images")
    os.makedirs(output_path, exist_ok=True)
    for i, (w, h, prompt) in tqdm.tqdm(enumerate(params)):
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            do_not_save_grid=True,
            do_not_save_samples=True,
            do_not_reload_embeddings=True,
            prompt=prompt,
            width=w,
            height=h,
            steps=20,
            sampler_name="DPM++ 2M",
        )
        processed = processing.process_images(p)
        image = processed.images[0]
        stem = f"{i+1:05d}"
        image.save(str(Path(output_path) / f"{stem}.png"))
        with open(Path(output_path) / f"{stem}.txt", "w") as f:
            f.write(prompt)
    print("done")
    return f"{len(params)} images saved to {output_path}"


if __name__ == "__main__":
    from launch import run_pip

    run_pip("install fire", "fire")
    import webui

    webui.initialize()
    sys.argv = argv
    import fire

    fire.Fire(make_reg_images)
