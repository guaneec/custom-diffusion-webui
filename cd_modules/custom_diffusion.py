from modules.textual_inversion.textual_inversion import *
from cd_modules.deltas import Delta

def train_embedding(*args):
    assert not shared.cmd_opts.lowvram, 'Training models with lowvram not possible'

    apply_optimizations = shared.opts.training_xattention_optimizations
    try:
        if not apply_optimizations:
            sd_hijack.undo_optimizations()

        embedding, filename = _train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        if not apply_optimizations:
            sd_hijack.apply_optimizations()

def _train_embedding(id_task, embedding_name, learn_rate, batch_size, gradient_step, data_root, reg_root, prior_loss_weight, log_directory, training_width, training_height, varsize, steps, clip_grad_mode, clip_grad_value, shuffle_tags, tag_drop_out, latent_sampling_method, create_image_every, save_embedding_every, template_filename, save_image_with_stored_embedding, preview_from_txt2img, kv_learn_rate, top_sum, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    template_file = textual_inversion_templates.get(template_filename, None)
    validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_embedding_every, create_image_every, log_directory, name="embedding")
    template_file = template_file.path

    shared.state.job = "train-embedding"
    shared.state.textinfo = "Initializing textual inversion training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
    os.makedirs(shared.cmd_opts.deltas_dir, exist_ok=True)
    kv_filename = os.path.join(shared.cmd_opts.deltas_dir, f'{embedding_name}.delta.safetensors')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    unload = shared.opts.unload_models_when_training

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    checkpoint = sd_models.select_checkpoint()

    initial_step = embedding.step or 0
    if initial_step >= steps:
        shared.state.textinfo = "Model has already been trained beyond specified max steps"
        return embedding, filename
    
    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    kv_scheduler = LearnRateScheduler(kv_learn_rate, steps, initial_step)
    clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else \
        torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else \
        None
    if clip_grad:
        clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    if shared.opts.training_enable_tensorboard:
        tensorboard_writer = tensorboard_setup(log_directory)

    pin_memory = shared.opts.pin_memory

    ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=template_file, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize)
    ds_reg = None
    if reg_root:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'tmp.txt'), 'w') as f:
                f.write('[filewords]')
            ds_reg = modules.textual_inversion.dataset.PersonalizedBase(data_root=reg_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token='', model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=f.name, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize)

    if shared.opts.save_training_settings_to_txt:
        save_settings_to_file(log_directory, {**dict(model_name=checkpoint.model_name, model_hash=checkpoint.shorthash, num_of_dataset_images=len(ds), num_vectors_per_token=len(embedding.vec)), **locals()})

    latent_sampling_method = ds.latent_sampling_method

    dl = modules.textual_inversion.dataset.PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)
    dl_reg = ds_reg and modules.textual_inversion.dataset.PersonalizedDataLoader(ds_reg, latent_sampling_method=latent_sampling_method, batch_size=ds_reg.batch_size, pin_memory=pin_memory)
    from itertools import chain, count
    reg_batch_iter = dl_reg and chain.from_iterable(dl_reg for _ in count())

    if unload:
        shared.parallel_processing_allowed = False
        shared.sd_model.first_stage_model.to(devices.cpu)

    embedding.vec.requires_grad = True
    kvs = {n: p for n, p in shared.sd_model.named_parameters() if '2.to_k' in n or '2.to_v' in n}
    kvs_bak = {n: p.detach().clone() for n, p in kvs.items()}

    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate, weight_decay=0.0)
    kv_optimzer = torch.optim.AdamW(kvs.values(), lr=kv_scheduler.learn_rate, weight_decay=0.0, eps=1e-5)
    if shared.opts.save_optimizer_state:
        optimizer_state_dict = None
        if os.path.exists(filename + '.optim'):
            optimizer_saved_dict = torch.load(filename + '.optim', map_location='cpu')
            if embedding.checksum() == optimizer_saved_dict.get('hash', None):
                optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
    
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            print("Loaded existing optimizer from checkpoint")
        else:
            print("No saved optimizer exists in checkpoint")

    scaler = torch.cuda.amp.GradScaler()

    # force allow_fp16 because pytorch doesn't like scaling fp16
    scaler._unscale_grads_bak = scaler._unscale_grads_
    scaler._unscale_grads_ = (lambda optimizer, inv_scale, found_inf, allow_fp16: 
                             scaler._unscale_grads_bak(optimizer, inv_scale, found_inf, True))

    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # n steps = batch_size * gradient_step * n image processed
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0 #internal

    last_saved_file = "<none>"
    last_saved_delta = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False

    is_training_inpainting_model = shared.sd_model.model.conditioning_key in {'hybrid', 'concat'}
    img_c = None

    def save_deltas(path):
        delta = Delta(tensors={k: kvs[k] - kvs_bak[k] for k in kvs})
        delta.save(path, *([] if top_sum == 1 else ['delta_factors', top_sum]))

    pbar = tqdm.tqdm(total=steps - initial_step)
    try:
        for i in range((steps-initial_step) * gradient_step):
            if scheduler.finished:
                break
            if shared.state.interrupted:
                break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                scheduler.apply(optimizer, embedding.step)
                kv_scheduler.apply(kv_optimzer, embedding.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break

                if clip_grad:
                    clip_grad_sched.step(embedding.step)
            
                with devices.autocast():
                    def get_loss(batch):
                        x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                        c = shared.sd_model.cond_stage_model(batch.cond_text)

                        if is_training_inpainting_model:
                            if img_c is None:
                                img_c = processing.txt2img_image_conditioning(shared.sd_model, c, training_width, training_height)

                            cond = {"c_concat": [img_c], "c_crossattn": [c]}
                        else:
                            cond = c

                        return shared.sd_model(x, cond)[0] / gradient_step
                    
                    loss = get_loss(batch)
                    if reg_batch_iter:
                        loss += get_loss(next(reg_batch_iter)) * prior_loss_weight

                    _loss_step += loss.item()
                scaler.scale(loss).backward()

                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue
                
                if clip_grad:
                    clip_grad(embedding.vec, clip_grad_sched.learn_rate)

                scaler.step(optimizer)
                scaler.step(kv_optimzer)
                scaler.update()
                embedding.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                kv_optimzer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0

                steps_done = embedding.step + 1

                epoch_num = embedding.step // steps_per_epoch
                epoch_step = embedding.step % steps_per_epoch

                description = f"Training textual inversion [Epoch {epoch_num}: {epoch_step+1}/{steps_per_epoch}] loss: {loss_step:.7f}"
                pbar.set_description(description)
                if embedding_dir is not None and steps_done % save_embedding_every == 0:
                    # Before saving, change name to match current checkpoint.
                    embedding_name_every = f'{embedding_name}-{steps_done}'
                    last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
                    last_saved_delta = os.path.join(embedding_dir, f'{embedding_name_every}.delta.safetensors')
                    save_embedding(embedding, optimizer, checkpoint, embedding_name_every, last_saved_file, remove_cached_checksum=True)
                    embedding_yet_to_be_embedded = True
                    save_deltas(last_saved_delta)

                write_loss(log_directory, "textual_inversion_loss.csv", embedding.step, steps_per_epoch, {
                    "loss": f"{loss_step:.7f}",
                    "learn_rate": scheduler.learn_rate,
                    "kv_learn_rate": kv_scheduler.learn_rate,
                })

                if images_dir is not None and steps_done % create_image_every == 0:
                    forced_filename = f'{embedding_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)

                    shared.sd_model.first_stage_model.to(devices.device)

                    p = processing.StableDiffusionProcessingTxt2Img(
                        sd_model=shared.sd_model,
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                        do_not_reload_embeddings=True,
                    )

                    if preview_from_txt2img:
                        p.prompt = preview_prompt
                        p.negative_prompt = preview_negative_prompt
                        p.steps = preview_steps
                        p.sampler_name = sd_samplers.samplers[preview_sampler_index].name
                        p.cfg_scale = preview_cfg_scale
                        p.seed = preview_seed
                        p.width = preview_width
                        p.height = preview_height
                    else:
                        p.prompt = batch.cond_text[0]
                        p.steps = 20
                        p.width = training_width
                        p.height = training_height

                    preview_text = p.prompt

                    processed = processing.process_images(p)
                    image = processed.images[0] if len(processed.images) > 0 else None

                    if unload:
                        shared.sd_model.first_stage_model.to(devices.cpu)

                    if image is not None:
                        shared.state.assign_current_image(image)

                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"

                        if shared.opts.training_enable_tensorboard and shared.opts.training_tensorboard_save_images:
                            tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image, embedding.step)

                    if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:

                        last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

                        info = PngImagePlugin.PngInfo()
                        data = torch.load(last_saved_file)
                        info.add_text("sd-ti-embedding", embedding_to_b64(data))

                        title = "<{}>".format(data.get('name', '???'))

                        try:
                            vectorSize = list(data['string_to_param'].values())[0].shape[0]
                        except Exception as e:
                            vectorSize = '?'

                        checkpoint = sd_models.select_checkpoint()
                        footer_left = checkpoint.model_name
                        footer_mid = '[{}]'.format(checkpoint.shorthash)
                        footer_right = '{}v {}s'.format(vectorSize, steps_done)

                        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                        captioned_image = insert_image_data_embed(captioned_image, data)

                        captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                        embedding_yet_to_be_embedded = False

                    last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
                    last_saved_image += f", prompt: {preview_text}"

                shared.state.job_no = embedding.step

                shared.state.textinfo = f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved delta weights: {html.escape(last_saved_delta)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
        filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
        save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True)
        save_deltas(kv_filename)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        pass
    finally:
        with torch.no_grad():
            for k, v in kvs.items():
                v[:] = kvs_bak[k]
                
        pbar.leave = False
        pbar.close()
        shared.sd_model.first_stage_model.to(devices.device)
        shared.parallel_processing_allowed = old_parallel_processing_allowed

    return embedding, filename
