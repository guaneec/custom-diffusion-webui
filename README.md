# Custom Diffusion WebUI

An unofficial extension that implements [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion/) for [Automatic1111's WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## What is Custom Diffusion

[Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion/) is, in short, finetuning-lite with TI. Instead of tuning the whole model, only the K and V matrices of the cross-attention blocks are tuned simultaneously with token embedding(s). It has similar speed and memory requirements to TI and supposedly gives better results in less steps.

## How to use this

### Training
You can find the UI in the `Train/Train Custom Diffusion` tab. Just train as you would a normal TI embedding. Under the training log directory, alongside with `name-steps.pt` you should also see `name-steps.delta.safetensors`, which contain finetuned delta weights (~50MB at half precision uncompressed).

#### Regularization images
Custom Diffusion proper includes regularization. To generate regularization images, go to `Custom Diffusion Utils/Make regularization images`. You can then optionally supply the generated images directory as regularization when training.


### Using trained weights
The trained deltas will be under `models/deltas` (`--deltas-dir`); you can also copy over logged `.safetensors` versions. The delta weights can be used in txt2img/img2img as an Extra Network. You can select them under the extra networks tab like hypernets. Use the token embedding like a normal TI embedding.

## Disclaimer
This is an unofficial implementation based on [the paper](https://arxiv.org/abs/2212.04488) and the features and implementation details may be different to [the original](https://github.com/adobe-research/custom-diffusion).

## Todo (roughly ordered by priority)
- [x] UI/UX
- [ ] More testing and demo
- [x] Separate lr for embedding and model weights
- [x] Blending (simple linear combination of deltas)
- [ ] Merging (optimization based from paper)
- [x] Compression
- [ ] Let users choose what weights to finetune
- [x] Regularization
- [ ] Multi-concept training
