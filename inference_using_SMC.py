import torch
import random

import models_mage
from pipeline_using_SMC import Pipeline
from scheduler import MageScheduler, ReMDMScheduler
from plot_utils import show_images_grid
from OpenAIClassifier.classifier import create_classifier

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

device = torch.device('cuda')

def get_classfier_fn():
    img_size = 256
    # Intialize reward models
    classifier = create_classifier(
        image_size=img_size,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",
        classifier_use_scale_shift_norm=True,
        classifier_resblock_updown=True,
        classifier_pool="attention",
    )
    checkpoint = torch.load(f"./{img_size}x{img_size}_classifier.pt", map_location='cpu')
    classifier.load_state_dict(checkpoint)
    classifier.eval()
    classifier = classifier.to(device)
    def tmp_fn(images, labels):
        logits = classifier(images, timesteps=torch.zeros(len(images)).to(device))
        logits = torch.log_softmax(logits, dim=-1)
        # print("Labels:", logits.argmax(dim=-1))
        # print("Probs: ", torch.exp(logits.max(dim=-1)[0]))
        return logits[torch.arange(len(labels)), labels].clamp_max(-0.1)
    return tmp_fn
classifier_fn = get_classfier_fn()

use_remdm = True

if use_remdm:
    scheduler = ReMDMScheduler(
        schedule="cosine", 
        remask_strategy="rescale",
        eta=0.2,
        mask_token_id=model.mask_token_label,
        temperature=1.0,
    )
else:
    scheduler = MageScheduler(
        mask_token_id = model.mask_token_label,
        choice_temperature=6.0,
    )
    
pipe = Pipeline(
    model=model,
    scheduler=scheduler,
    codebook_size=1024,
    mask_token_id = model.mask_token_label,
    device=device,
    use_mixed_precision=False,
)

num_samples = 16
# goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
labels = [1, 7, 282, 604, 724, 179, 751, 404, 850] + [random.randint(0, 999) for _ in range(num_samples - 9)]

num_particles = 8
batch_p = 2
reward_fn = lambda images : classifier_fn(
    images, 
    torch.tensor(
        [1] * len(images)
    ).to(device)
)

images = pipe(
    num_inference_steps=100,
    disable_progress_bar=True,
    # SMC paramters
    num_particles=num_particles,
    batch_p=batch_p,
    kl_coeff=0.1,
    tempering_gamma=0.05,
    reward_fn=reward_fn,
    use_reverse_as_proposal=False,
    verbose=True,
)

show_images_grid(images, save_file="inference_SMC.png")
