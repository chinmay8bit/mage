from typing import Union, Optional, Callable
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scheduler import BaseScheduler
from smc_utils import compute_ess_from_log_w, resampling_function, adaptive_tempering
from plot_utils import show_images_grid


def assert_one_hot(x):
    assert ((x == 0) | (x == 1)).all() and (x.sum(dim=-1) == 1).all(), "Tensor is not one-hot"


class Pipeline():
    def __init__(
        self, 
        model,
        scheduler: BaseScheduler, 
        codebook_size: int,
        mask_token_id: int, 
        device: torch.device = torch.device('cuda'),
        use_mixed_precision: bool = False,
    ):
        self.codebook_size = codebook_size
        self.mask_token_id = mask_token_id
        self.device = device
        self.model = model.to(device)
        self.scheduler = scheduler
    
    @torch.no_grad()
    def __call__(
        self,
        num_inference_steps: int = 48,
        disable_progress_bar = False,
        # SMC parameters
        num_particles: int = 4,
        batch_p: int = 1, # number of particles to run parallely
        resample_strategy: str = "ssp",
        ess_threshold: float = 0.5,
        tempering: str = "schedule",
        tempering_schedule: Union[float, int, str] = "exp",
        tempering_gamma: float = 1.,
        tempering_start: float = 0.,
        reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, # Ex) lambda images: _fn(images, prompts.repeat_interleave(batch_p, dim=0), metadata.repeat_interleave(batch_p, dim=0))
        kl_coeff: float = 1.,
        use_reverse_as_proposal: bool = False,
        verbose: bool = False # True for debugging SMC procedure
    ):
        codebook_emb_dim = 256
        unknown_number_in_the_beginning = 256
        vocab_size = self.codebook_size + 1000 + 1
        
        #1. Check input
        ...
                        
        #2. Set batch size
        batch_size = min(batch_p, num_particles)
        
        #3. Set up intial latents
        latents = torch.full((num_particles, unknown_number_in_the_beginning), self.mask_token_id, device=self.device)
        
        #4. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Intialize variables for SMC sampler
        logits = torch.zeros((*latents.shape, vocab_size), device=self.device) # type: ignore
        approx_guidance = torch.zeros((*latents.shape, vocab_size), device=self.device) # type: ignore 
        log_w = torch.zeros(latents.shape[0], device=self.device)
        log_prob_diffusion = torch.zeros(latents.shape[0], device=self.device)
        log_prob_proposal = torch.zeros(latents.shape[0], device=self.device)
        log_twist_func = torch.zeros(latents.shape[0], device=self.device)
        log_twist_func_prev = torch.zeros(latents.shape[0], device=self.device)
        rewards = torch.zeros(latents.shape[0], device=self.device)
        resample_fn = resampling_function(resample_strategy=resample_strategy, ess_threshold=ess_threshold)
        scale_factor = 0.
        min_scale_next = 0.
        
        kl_coeff = torch.tensor(kl_coeff, device=self.device).to(torch.float32) # type: ignore
        lookforward_fn = lambda r: r / kl_coeff
        
        start = int(num_inference_steps * tempering_start)
        
        def _calc_guidance():
            assert latents is not None
            if (i >= start):
                imgs = []
                for idx in range(math.ceil(num_particles / batch_size)):
                    context = torch.no_grad() if use_reverse_as_proposal else torch.enable_grad()
                    with context:
                        latents_one_hot = F.one_hot(
                            latents[batch_size*idx : batch_size*(idx+1)], 
                            num_classes=vocab_size # type: ignore
                        ).float().requires_grad_(True)

                        tmp_logits = self.get_unconditional_logits(latents_one_hot)
                        
                        M = 3
                        tmp_rewards = torch.zeros_like(rewards[batch_size*idx : batch_size*(idx+1)])

                        for _ in range(M):
                            tmp_pred_original_sample: torch.Tensor = self.get_pred_original_sample(
                                logits=tmp_logits,
                                sample_one_hot=latents_one_hot,
                                use_continuous_formualtion=True,
                            )
                            
                            tmp_pred_original_sample_decoded = self.decode_one_hot_latents(tmp_pred_original_sample)
                            
                            # Calculate rewards
                            tmp_rewards += (1/M) * reward_fn(tmp_pred_original_sample_decoded).to(torch.float32) # type: ignore
                            
                        if i % 10 == 0:
                            imgs.append(tmp_pred_original_sample_decoded.detach().cpu()) # type: ignore
                        
                        tmp_log_twist_func = lookforward_fn(tmp_rewards).to(torch.float32)
                        
                        # Calculate approximate guidance noise for maximizing reward
                        if use_reverse_as_proposal:
                            tmp_approx_guidance = torch.zeros_like(approx_guidance[batch_size*idx : batch_size*(idx+1)])
                        else:
                            tmp_approx_guidance = torch.autograd.grad( # type: ignore
                                outputs=tmp_log_twist_func,
                                inputs=latents_one_hot,
                                grad_outputs=torch.ones_like(tmp_log_twist_func)
                            )[0].detach()
                        
                        logits[batch_size*idx : batch_size*(idx+1)] = tmp_logits.detach().clone()
                        rewards[batch_size*idx : batch_size*(idx+1)] = tmp_rewards.detach().clone()
                        log_twist_func[batch_size*idx : batch_size*(idx+1)] = tmp_log_twist_func.detach().clone()
                        approx_guidance[batch_size*idx : batch_size*(idx+1)] = tmp_approx_guidance.clone()
                
                if i % 10 == 0:
                    show_images_grid(torch.cat(imgs, dim=0), save_file=f"inference_SMC.png")
                
                if torch.isnan(log_twist_func).any():
                    if verbose:
                        print("NaN in log twist func, changing it to 0")
                    log_twist_func[:] = torch.nan_to_num(log_twist_func)
                if torch.isnan(approx_guidance).any():
                    if verbose:
                        print("NaN in approx guidance, changing it to 0")
                    approx_guidance[:] = torch.nan_to_num(approx_guidance)
            
            else:
                for idx in range(math.ceil(num_particles / batch_size)):
                    batch_latents = latents[batch_size*idx : batch_size*(idx+1)].clone()
                    tmp_logits = self.get_unconditional_logits(batch_latents)
                    
                    logits[batch_size*idx : batch_size*(idx+1)] = tmp_logits.detach().clone()
            
            if verbose:
                print("Expected rewards of proposals: ", rewards)
                print("Approx guidance norm: ", (approx_guidance ** 2).mean().sqrt())
        
        
        bar = range(num_inference_steps) if disable_progress_bar else tqdm(range(num_inference_steps), leave=False)
        for i in bar:
            if verbose:
                print("\n", "-"*50, i, "-"*50, "\n")
                
            log_twist_func_prev = log_twist_func.clone() # Used to calculate weight later
            
            _calc_guidance()
            
            with torch.no_grad():
                if i >= start:
                    ################### Select Temperature ###################
                    if isinstance(tempering_schedule, float) or isinstance(tempering_schedule, int):
                        min_scale = min((tempering_gamma * (i - start))**tempering_schedule, 1.)
                        min_scale_next = min(tempering_gamma * (i + 1 - start), 1.)
                    elif tempering_schedule == "exp":
                        min_scale = min((1 + tempering_gamma) ** (i - start) - 1, 1.)
                        min_scale_next = min((1 + tempering_gamma) ** (i + 1 - start) - 1, 1.)
                    elif tempering_schedule == "adaptive":
                        min_scale = scale_factor
                    else:
                        min_scale = 1.
                        min_scale_next = 1.
                    
                    if tempering == "adaptive" and i > 0 and min_scale < 1.:
                        scale_factor = adaptive_tempering(
                            log_w.view(num_particles, -1).T, 
                            log_prob_diffusion.view(num_particles, -1).T, 
                            log_twist_func.view(num_particles, -1).T, 
                            log_prob_proposal.view(num_particles, -1).T, 
                            log_twist_func_prev.view(num_particles, -1).T, 
                            min_scale=min_scale, ess_threshold=ess_threshold
                        )
                        min_scale_next = scale_factor.clone()
                    elif tempering == "adaptive" and i == 0:
                        pass
                    elif tempering == "schedule":
                        scale_factor = min_scale
                    else:
                        scale_factor = 1.

                    if verbose:
                        print("scale factor (lambda_t): ", scale_factor)
                        print("min scale next (lambda_t-1): ", min_scale_next)
                    
                    log_twist_func *= scale_factor
                    approx_guidance *= min_scale_next
                    
                    if verbose:
                        print("Approx guidance norm after scale: ", (approx_guidance ** 2).mean().sqrt())
                    
                    ################### Weight & Resample (Importance Sampling) ###################
                    
                    # Calculate weights for samples from proposal distribution
                    incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
                    
                    log_w += incremental_log_w.detach()

                    ess = compute_ess_from_log_w(log_w).item()

                    # resample latents and corresponding variables
                    resample_indices, is_resampled, log_w = resample_fn(log_w.view(1, num_particles))
                    assert len(log_w) == 1 and len(resample_indices) == 1
                    log_w, resample_indices = log_w[0], resample_indices[0].long()

                    if verbose:
                        if is_resampled.any():
                            print("\n" + "="*50)
                            print(" 🔁✨ RESAMPLED! ✨🔁 ")
                            print("="*50 + "\n")
                        print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
                        print("log_twist_func - log_twist_func_prev: ", log_twist_func - log_twist_func_prev)
                        print("Incremental weight: ", incremental_log_w)
                        print("Effective sample size: ", ess)
                        print("Resampled particles indices: ", resample_indices)
                        
                    # Update variables based on resampling
                    latents = latents[resample_indices] # type: ignore
                    logits = logits[resample_indices]
                    approx_guidance = approx_guidance[resample_indices]
                    rewards = rewards[resample_indices]
                    log_twist_func = log_twist_func[resample_indices]
                
                ################### Propose Particles ###################    
                # Sample from proposal distribution
                sched_out = self.scheduler.step_with_approx_guidance(
                    logits=logits,
                    approx_guidance=approx_guidance,
                    latents=latents,
                    step=i,
                )
                latents, log_prob_proposal, log_prob_diffusion = (
                    sched_out.new_latents,
                    sched_out.log_prob_proposal,
                    sched_out.log_prob_diffusion,
                )
        
        # Weights for Final samples
        if verbose:
            print("\n", "-"*50, "final", "-"*50, "\n")
            
        log_twist_func_prev = log_twist_func.clone()
        
        #6. Decode latents to get images
        images = []
        for idx in range(math.ceil(num_particles / batch_size)):
            latents_one_hot = F.one_hot(
                latents[batch_size*idx : batch_size*(idx+1)], 
                num_classes=self.codebook_size + 1
            ).float()
            tmp_images = self.decode_one_hot_latents(latents_one_hot)
            
            # Calculate rewards
            tmp_rewards = reward_fn(tmp_images).to(torch.float32) # type: ignore
            tmp_log_twist_func = lookforward_fn(tmp_rewards).to(torch.float32)
            
            rewards[batch_size*idx : batch_size*(idx+1)] = tmp_rewards.detach().clone()
            log_twist_func[batch_size*idx : batch_size*(idx+1)] = tmp_log_twist_func.detach().clone()
            images.append(tmp_images)
        images = torch.cat(images, dim=0)
        
        if verbose:
            print("Final rewards: ", rewards)
            
        incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
        log_w += incremental_log_w.detach()

        ess = compute_ess_from_log_w(log_w).item()

        if verbose:
            print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
            print("log_twist_func - log_twist_func_prev: ", log_twist_func - log_twist_func_prev)
            print("Incremental weight: ", incremental_log_w)
            print("Weight: ", log_w)
            print("Effective sample size: ", ess)
        
        return images

    def get_unconditional_logits(self, latents_one_hot: torch.Tensor):
        B, L, C = latents_one_hot.shape
        token_indices = torch.cat(
            [torch.zeros(B, 1, C, device=self.device), latents_one_hot], dim=1)
        token_indices[:, 0, self.model.fake_class_label] = 1 
        token_all_mask = token_indices.argmax(dim=-1) == self.mask_token_id

        token_drop_mask = torch.zeros_like(token_all_mask, dtype=torch.long)

        # token embedding
        input_embeddings = self.model.token_emb(token_indices)

        # encoder
        x = input_embeddings
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)

        # decoder
        logits = self.model.forward_decoder(x, token_drop_mask, token_all_mask)
        logits = logits[:, 1:, :]
        logits[..., self.codebook_size:] = float('-inf')
        return logits
    
    def get_pred_original_sample(
        self,
        logits: torch.Tensor,
        sample_one_hot: torch.Tensor, # one_hot_sample
        use_continuous_formualtion=True,
    ) -> torch.Tensor:

        vocab_size = sample_one_hot.shape[-1]
        codebook_size = self.codebook_size
        batch_size = len(sample_one_hot)

        pred_sample = F.gumbel_softmax(
            logits=logits,
            hard=True,
            dim=-1,
            # tau=self.scheduler.temperature,
        )
        pred_original_sample = torch.zeros_like(pred_sample)
        
        if use_continuous_formualtion:
            # Carry Over Unmaksing - continuous formulation
            pred_original_sample[..., :codebook_size] = (
                pred_sample[..., :codebook_size] * (1 - sample_one_hot[..., :codebook_size].sum(dim=-1, keepdim=True)) +
                sample_one_hot[..., :codebook_size]
            )
        else:
            pred_original_sample[..., :codebook_size] = torch.where(
                sample_one_hot[..., :codebook_size].sum(dim=-1, keepdim=True) == 0,
                pred_sample[..., :codebook_size],
                sample_one_hot[..., :codebook_size]
            )

        assert_one_hot(pred_original_sample)
        return pred_original_sample

    def decode_one_hot_latents(self, latents_one_hot):
        B, L, C = latents_one_hot.shape
        embedding = self.model.vqgan.quantize.embedding.weight
        z_q = latents_one_hot[..., :self.codebook_size] @ embedding
        z_q = z_q.reshape(B, 16, 16, embedding.size(1))
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        gen_images = self.model.vqgan.decode(z_q).clip(0, 1)
        return gen_images
