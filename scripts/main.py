import modules.scripts as scripts
import gradio as gr
from modules.ui import create_refresh_button
from modules.paths import models_path
from modules.script_callbacks import on_ui_train_tabs
from modules.ui_components import FormRow
import modules.sd_models as sd_models
import glob
import os
from modules import shared, sd_hijack
from modules.textual_inversion import textual_inversion
from modules.call_queue import wrap_gradio_gpu_call
from safetensors import safe_open
import cd_modules
import torch
from math import prod
import modules
import cd_modules.custom_diffusion
import json

deltas = {}
backup = {}


def list_deltas():
    res = {}
    for filename in sorted(
        glob.iglob(
            os.path.join(models_path, "deltas", "**/*.safetensors"), recursive=True
        )
    ):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name != "None":
            res[name + f"({sd_models.model_hash(filename)})"] = filename
    return res


last_model_hash, last_delta_name = None, None


def apply(delta_name, model):
    global backup, last_model_hash, last_delta_name
    if (last_model_hash, last_delta_name) == (model.sd_model_hash, delta_name):
        return
    if last_model_hash != model.sd_model_hash:
        backup = {}
    if delta_name != 'None':
        st = safe_open(deltas[delta_name], 'pt')
        metadata = json.loads(st.metadata()['json'])
        print(metadata['meta'])
        entries = metadata['entries']
        print(f"loaded weights {delta_name} ({sum(prod(st.get_tensor(k).shape) for k in st.keys())} params in {len(entries)} entries)")
    else:
        entries = {}
    
    for k, v in model.named_parameters():
        if k in entries and k not in backup:
            backup[k] = v.detach().clone()
        if k in entries:
            with torch.no_grad():
                v[:] = st.get_tensor(k).to(model.device) + backup[k]
            print(f"loaded {k} from {delta_name}")
        elif k in backup:
            with torch.no_grad():
                v[:] = backup[k]
            del backup[k]
            print(f"restored {k}")
    last_model_hash, last_delta_name = model.sd_model_hash, delta_name


class Script(scripts.Script):
    def title(self):
        return "Custom Diffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Row():
            delta_name = gr.Dropdown(
                ["None", *deltas.keys()], label="Tuned weights", value="None"
            )

            def refresh():
                global deltas
                deltas = list_deltas()

            refresh()
            refresh = create_refresh_button(
                delta_name,
                refresh,
                (lambda: dict(choices=["None", *deltas.keys()])),
                "plug_refresh",
            )
        return [delta_name, refresh]

    def process(self, p, delta_name, _refresh):
        apply(delta_name, p.sd_model)


def get_textual_inversion_template_names():
    return sorted([x for x in textual_inversion.textual_inversion_templates])


def train_tabs_callback(ui_train_tab_params):
    with gr.Tab(label="Train Custom Diffusion"):
        with FormRow():
            train_embedding_name = gr.Dropdown(
                label="Embedding",
                elem_id="train_embedding",
                choices=sorted(
                    sd_hijack.model_hijack.embedding_db.word_embeddings.keys()
                ),
            )
            create_refresh_button(
                train_embedding_name,
                sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings,
                lambda: {
                    "choices": sorted(
                        sd_hijack.model_hijack.embedding_db.word_embeddings.keys()
                    )
                },
                "refresh_train_embedding_name",
            )

        with FormRow():
            embedding_learn_rate = gr.Textbox(
                label="Embedding Learning rate",
                placeholder="Embedding Learning rate",
                value="0.005",
                elem_id="train_embedding_learn_rate",
            )
            kv_learn_rate = gr.Textbox(label='KV Learning rate', placeholder="KV Learning rate", value="1e-5", elem_id="train_kv_learn_rate")

        with FormRow():
            clip_grad_mode = gr.Dropdown(
                value="disabled",
                label="Gradient Clipping",
                choices=["disabled", "value", "norm"],
            )
            clip_grad_value = gr.Textbox(
                placeholder="Gradient clip value", value="0.1", show_label=False
            )

        with FormRow():
            batch_size = gr.Number(
                label="Batch size", value=1, precision=0, elem_id="train_batch_size"
            )
            gradient_step = gr.Number(
                label="Gradient accumulation steps",
                value=1,
                precision=0,
                elem_id="train_gradient_step",
            )

        dataset_directory = gr.Textbox(
            label="Dataset directory",
            placeholder="Path to directory with input images",
            elem_id="train_dataset_directory",
        )
        log_directory = gr.Textbox(
            label="Log directory",
            placeholder="Path to directory where to write outputs",
            value="textual_inversion",
            elem_id="train_log_directory",
        )

        with FormRow():
            template_file = gr.Dropdown(
                label="Prompt template",
                value="style_filewords.txt",
                elem_id="train_template_file",
                choices=get_textual_inversion_template_names(),
            )
            create_refresh_button(
                template_file,
                textual_inversion.list_textual_inversion_templates,
                lambda: {"choices": get_textual_inversion_template_names()},
                "refrsh_train_template_file",
            )

        training_width = gr.Slider(
            minimum=64,
            maximum=2048,
            step=8,
            label="Width",
            value=512,
            elem_id="train_training_width",
        )
        training_height = gr.Slider(
            minimum=64,
            maximum=2048,
            step=8,
            label="Height",
            value=512,
            elem_id="train_training_height",
        )
        varsize = gr.Checkbox(
            label="Do not resize images", value=False, elem_id="train_varsize"
        )
        steps = gr.Number(
            label="Max steps", value=100000, precision=0, elem_id="train_steps"
        )

        with FormRow():
            create_image_every = gr.Number(
                label="Save an image to log directory every N steps, 0 to disable",
                value=500,
                precision=0,
                elem_id="train_create_image_every",
            )
            save_embedding_every = gr.Number(
                label="Save a copy of embedding to log directory every N steps, 0 to disable",
                value=500,
                precision=0,
                elem_id="train_save_embedding_every",
            )

        save_image_with_stored_embedding = gr.Checkbox(
            label="Save images with embedding in PNG chunks",
            value=True,
            elem_id="train_save_image_with_stored_embedding",
        )
        preview_from_txt2img = gr.Checkbox(
            label="Read parameters (prompt, etc...) from txt2img tab when making previews",
            value=False,
            elem_id="train_preview_from_txt2img",
        )

        shuffle_tags = gr.Checkbox(
            label="Shuffle tags by ',' when creating prompts.",
            value=False,
            elem_id="train_shuffle_tags",
        )
        tag_drop_out = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.1,
            label="Drop out tags when creating prompts.",
            value=0,
            elem_id="train_tag_drop_out",
        )

        latent_sampling_method = gr.Radio(
            label="Choose latent sampling method",
            value="once",
            choices=["once", "deterministic", "random"],
            elem_id="train_latent_sampling_method",
        )

        with gr.Row():
            train_embedding = gr.Button(
                value="Train Embedding",
                variant="primary",
                elem_id="train_train_embedding",
            )
            interrupt_training = gr.Button(
                value="Interrupt", elem_id="train_interrupt_training"
            )
        dummy_component = gr.Label(visible=False)
        with gr.Column(elem_id='ti_gallery_container'):
            ti_output = gr.Text(value="", show_label=False)
            ti_outcome = gr.HTML(value="")

    train_embedding.click(
        fn=wrap_gradio_gpu_call(cd_modules.custom_diffusion.train_embedding, extra_outputs=[gr.update()]),
        _js="start_training_textual_inversion",
        inputs=[
            dummy_component,
            train_embedding_name,
            embedding_learn_rate,
            batch_size,
            gradient_step,
            dataset_directory,
            log_directory,
            training_width,
            training_height,
            varsize,
            steps,
            clip_grad_mode,
            clip_grad_value,
            shuffle_tags,
            tag_drop_out,
            latent_sampling_method,
            create_image_every,
            save_embedding_every,
            template_file,
            save_image_with_stored_embedding,
            preview_from_txt2img,
            kv_learn_rate,
            *ui_train_tab_params.txt2img_preview_params,
        ],
        outputs=[
            ti_output,
            ti_outcome,
        ]
    )

    interrupt_training.click(
        fn=lambda: shared.state.interrupt(),
        inputs=[],
        outputs=[],
    )


on_ui_train_tabs(train_tabs_callback)
