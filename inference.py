import torch

import models_mage
from pipeline import Pipeline
from scheduler import MageScheduler
from plot_utils import show_images_grid


use_large_model = True
if use_large_model:
    model_name = "mage_vit_large_patch16"
    model_ckpt = "/vol/bitbucket/cp524/dev/papers_with_code/mage/mage-vitl-1600.pth"
else:
    model_name = "mage_vit_base_patch16"
    model_ckpt = "/vol/bitbucket/cp524/dev/papers_with_code/mage/mage-vitb-1600.pth"    
vqgan_ckpt_path = 'vqgan_jax_strongaug.ckpt'

model = models_mage.__dict__[model_name](norm_pix_loss=False,
                                         mask_ratio_mu=0.55, mask_ratio_std=0.25,
                                         mask_ratio_min=0.0, mask_ratio_max=1.0,
                                         vqgan_ckpt_path=vqgan_ckpt_path)
checkpoint = torch.load(model_ckpt, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

scheduler = MageScheduler(
    mask_token_id = model.mask_token_label,
    choice_temperature=6.0,
)

pipe = Pipeline(
    model=model,
    scheduler=scheduler,
    codeboook_size=1024,
    mask_token_id = model.mask_token_label,
    device=torch.device('cuda'),
    use_mixed_precision=False,
)

images = pipe(
    num_inference_steps=10,
    disable_progress_bar=False,
    num_particles=8,   
)

show_images_grid(images, save_file="inference.png")
