import modules.scripts as scripts
import gradio as gr
from modules.ui import create_refresh_button
from modules.paths import models_path
from modules.script_callbacks import on_ui_train_tabs, on_ui_tabs
from modules.ui_components import FormRow
import modules.sd_models as sd_models
import glob
import os
from modules import shared, sd_hijack, extra_networks, ui_extra_networks
from modules.textual_inversion import textual_inversion
from modules.call_queue import wrap_gradio_gpu_call
from safetensors import safe_open
import cd_modules
import torch
import modules
import cd_modules.custom_diffusion
import cd_modules.ui_extra_networks_deltas
import cd_modules.extra_networks_deltas
import cd_modules.deltas
import cli_scripts.make_reg


class Script(scripts.Script):
    def title(self):
        return "Custom Diffusion"

    def show(self, is_img2img):
        return False

# monkeypatch initialize's as there's currently no API to register new networks
ui_extra_networks_initialize_bak = ui_extra_networks.intialize
def ui_extra_networks_initialize_patched():
    ui_extra_networks_initialize_bak()
    ui_extra_networks.register_page(cd_modules.ui_extra_networks_deltas.ExtraNetworksPageDeltas())
    print('patched in extra network ui page: deltas')
ui_extra_networks.intialize = ui_extra_networks_initialize_patched

extra_networks_initialize_bak = extra_networks.initialize
def extra_networks_initialize_patched():
    extra_networks_initialize_bak()
    extra_networks.register_extra_network(cd_modules.extra_networks_deltas.ExtraNetworkDelta())
    print('patched in extra network: deltas')
extra_networks.initialize = extra_networks_initialize_patched


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
        with FormRow():
            reg_dataset_directory = gr.Textbox(
                label="Regularization dataset directory (optional)",
                placeholder="Path to directory reg images",
            )
            prior_loss_weight = gr.Slider(
                label="Prior-preservation loss weight",
                value=1.,
                minimum=0.,
                maximum=10.,
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

        top_sum = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            label="Low-rank approximation sum threshold (lower value means smaller file size, 1 to disable)",
            value=0.5,
            elem_id="train_top_sum",
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
            reg_dataset_directory,
            prior_loss_weight,
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
            top_sum,
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

def btn_compress_click(delta_name, top_sum, custom_name):
    if not delta_name:
        return "Error: delta not selected"
    from safetensors import safe_open
    from safetensors.torch import save_file
    from cd_modules.compression import decompose
    import json
    from pathlib import Path
    orig_path = cd_modules.deltas.Delta.deltas[delta_name]
    st = safe_open(orig_path, 'pt')
    metadata = json.loads(st.metadata()['json'])
    entries = metadata['entries']
    tensors = {}
    for k, v in entries.items():
        if v == 'delta':
            d = st.get_tensor(k)
        elif v == 'delta_factors':
            print('Warning: compressing already factored delta')
            d = st.get_tensor(k+'.US').float() @ st.get_tensor(k+'.Vh').float()
        else:
            return 'Error: Unknown format: {v}'
        tensors[k+'.US'], tensors[k+'.Vh'] =  map(lambda a: a.half().contiguous(),
            decompose(d, top_sum))
    metadata = {'meta': {'version': '0.2.0'}, 'entries': {k: 'delta_factors' for k in entries}}
    p = Path(orig_path)
    new_path = str(p.parent / ((custom_name or p.stem + f'.lora{int(100 * top_sum)}') + p.suffix))
    save_file(tensors, new_path, {'json': json.dumps(metadata)})
    return f'Compressed delta saved to {new_path}'


def ui_tabs_callback():
    with gr.Blocks() as cd:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='compact'):
                with gr.Blocks():
                    with gr.Tab("Compress"):
                        with gr.Row():
                            delta_name = gr.Dropdown(
                                list(cd_modules.deltas.Delta.deltas.keys()), label="Delta"
                            )
                            create_refresh_button(delta_name, cd_modules.deltas.Delta.refresh, 
                            lambda: dict(choices=list(cd_modules.deltas.Delta.deltas.keys())), 'refresh_deltas')
                        top_sum = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            label="Low-rank approximation sum threshold (lower value means smaller file size, 1 to disable)",
                            value=0.5,
                            elem_id="train_top_sum",
                        )
                        custom_name = gr.Textbox(label="Custom Name (Optional)")
                        btn_compress = gr.Button(
                            value="Compress",
                            variant="primary",
                            elem_id="btn_compress",
                        )
                    with gr.Tab("Merge"):
                        gr.Markdown("Coming soon")
                    with gr.Tab("Make regularization images"):
                        with FormRow():
                            data_root = gr.Textbox(label="Dataset root")
                            output_path = gr.Textbox(label="Destination of the generated images")
                        n_images = gr.Textbox(label='Total number of images. Will be rounded up to the next multiple of the dataset size', placeholder='A number or "Nx" for N times the original dataset size.')
                        shuffle_tags = gr.Checkbox(label="Shuffle comma-delimitted tags")

                        with FormRow():
                            template_file = gr.Dropdown(
                                label="Prompt template",
                                value="style_filewords.txt",
                                elem_id="train_template_file2",
                                choices=get_textual_inversion_template_names(),
                            )
                            create_refresh_button(
                                template_file,
                                textual_inversion.list_textual_inversion_templates,
                                lambda: {"choices": get_textual_inversion_template_names()},
                                "refrsh_train_template_file2",
                            )
                        placeholder_token = gr.Textbox(label="String to replace [name] with in templates")
                        btn_make_reg = gr.Button(value="Generate images", variant="primary")

            with gr.Column(variant='compact'):
                cd_out = gr.Markdown()
        btn_compress.click(wrap_gradio_gpu_call(btn_compress_click, extra_outputs=[gr.update()]), [delta_name, top_sum, custom_name], [cd_out])
        btn_make_reg.click(wrap_gradio_gpu_call(cli_scripts.make_reg.make_reg_images, extra_outputs=[gr.update()]), [data_root, n_images, output_path, template_file, shuffle_tags, placeholder_token], [cd_out])
    
    return [(cd, 'Custom Diffusion Utils', 'cdblock')]
        

on_ui_tabs(ui_tabs_callback)
