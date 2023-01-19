# Custom Diffusion WebUI

An unofficial implementation of Custom Diffusion for Automatic1111's WebUI.

## What is Custom Diffusion

[Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion/) is, in short, finetuning-lite with TI. Instead of tuning the whole model, only the K and V matrices of the cross-attention blocks are tuned simultaneously with token embedding(s). It has similar speed and memory requirements to TI and supposedly gives better results in less steps.

## How to use this

### Training
You can find the UI in the `Train/Train Custom Diffusion` tab. Just train as you would a normal TI embedding. Under the training log directory, alongside with `name-steps.pt` you should also see `name-steps.delta.safetensors`, which contain finetuned delta weights (~50MB at half precision uncompressed).

### Using trained weights
The trained deltas will be under `models/deltas` (`--deltas-dir`); you can also copy over logged `.safetensors` versions. You can apply the delta weights by selecting them in the `Tuned weights` option in txt2img/img2img. Use the token embedding like a normal TI embedding.


## Todo (roughly ordered by priority)
- [x] UI/UX
- [ ] More testing and demo
- [x] Separate lr for embedding and model weights
- [ ] Merging / Blending
- [ ] Compression
- [ ] Let users choose what weights to finetune
- [ ] Regularization
- [ ] Multi-concept training
