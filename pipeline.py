from typing import Union, Optional, Callable
import torch
from tqdm import tqdm

from scheduler import BaseScheduler


class Pipeline():
    def __init__(
        self, 
        model,
        scheduler: BaseScheduler, 
        codeboook_size: int,
        mask_token_id: int, 
        device: torch.device = torch.device('cuda'),
        use_mixed_precision: bool = False,
    ):
        self.codebook_size = codeboook_size
        self.mask_token_id = mask_token_id
        self.device = device
        self.model = model.to(device)
        self.scheduler = scheduler
    
    @torch.no_grad()
    def __call__(
        self,
        num_inference_steps: int = 48,
        disable_progress_bar = False,
        num_particles: int = 4,
    ):
        codebook_emb_dim = 256
        unknown_number_in_the_beginning = 256
        latents = torch.full((num_particles, unknown_number_in_the_beginning), self.mask_token_id, device=self.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        bar = range(num_inference_steps) if disable_progress_bar else tqdm(range(num_inference_steps), leave=False)
        for step in bar:
            token_indices = torch.cat(
                [torch.zeros(latents.size(0), 1, device=self.device), latents], dim=1)
            token_indices[:, 0] = self.model.fake_class_label
            token_indices = token_indices.long()
            token_all_mask = token_indices == self.mask_token_id

            token_drop_mask = torch.zeros_like(token_indices)

            # token embedding
            input_embeddings = self.model.token_emb(token_indices)

            # encoder
            x = input_embeddings
            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)

            # decoder
            logits = self.model.forward_decoder(x, token_drop_mask, token_all_mask)
            logits = logits[:, 1:, :self.codebook_size]
            
            sched_out = self.scheduler.step(latents, step, logits)
            latents = sched_out.new_latents
        
        # vqgan visualization
        z_q = self.model.vqgan.quantize.get_codebook_entry(latents, shape=(num_particles, 16, 16, codebook_emb_dim))
        gen_images = self.model.vqgan.decode(z_q).clip(0, 1)
        return gen_images
