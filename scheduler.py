from abc import ABC, abstractmethod
from dataclasses import dataclass

import math
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SchedulerStepOutput:
    new_latents: torch.Tensor


@dataclass
class SchedulerApproxGuidanceOutput:
    new_latents: torch.Tensor
    log_prob_proposal: torch.Tensor
    log_prob_diffusion: torch.Tensor


class BaseScheduler(ABC):
    @abstractmethod
    def step(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
    ) -> SchedulerStepOutput:
        pass
    
    @abstractmethod
    def set_timesteps(self, num_inference_steps: int):
        pass

    @abstractmethod
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
    ) -> SchedulerApproxGuidanceOutput:
        pass


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).cuda()
    sorted_confidence, _ = torch.sort(confidence, axis=-1) # type: ignore
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    # Masks tokens with lower confidence.
    masking = (confidence <= cut_off)
    return masking

def sum_masked_logits(
    logits: torch.Tensor,
    preds: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Sum logits at `preds` indices, masked by `mask`, handling invalid `preds`.

    Args:
        logits: Tensor of shape (B, T, C) - logits over C classes.
        preds: Tensor of shape (B, T) - predicted class indices.
        mask: Tensor of shape (B, T) - binary mask to include positions.

    Returns:
        Tensor of shape (B,) - sum of selected logits per batch item.
    """
    B, T, C = logits.shape
    # Ensure preds are in valid index range [0, C-1]
    valid = (preds >= 0) & (preds <= preds[mask].max())
    # Replace invalid preds with a dummy index (0), which we will mask later
    safe_preds = preds.masked_fill(~valid, 0)
    # Gather logits at predicted indices
    selected = torch.gather(logits, dim=2, index=safe_preds.unsqueeze(-1)).squeeze(-1)
    # Zero out contributions from invalid preds and masked positions
    selected = selected * valid * mask
    # Sum over time dimension
    return selected.sum(dim=1)

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation of log(1 - exp(x)) for x < 0.
    """
    return torch.where(
        x > -1,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


class MageScheduler(BaseScheduler):
    def __init__(self, 
            mask_token_id: int, 
            choice_temperature: float,
        ):
        self.mask_token_id = mask_token_id
        self.choice_temperature = choice_temperature
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
    
    def step(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
    ) -> SchedulerStepOutput:
        B, L, C = logits.shape
        assert latents.shape == (B, L)
        
        _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf
        unknown_number_in_the_beginning = 256
        
        # get token prediction
        sample_dist = torch.distributions.Categorical(logits=logits) # type: ignore
        sampled_ids = sample_dist.sample()
        
        # get ids for next step
        unknown_map = (latents == self.mask_token_id)
        sampled_ids = torch.where(unknown_map, sampled_ids, latents)
        
        if step + 1 < self.num_inference_steps:
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1. * (step + 1) / self.num_inference_steps

            mask_ratio = np.cos(math.pi / 2. * ratio)

            # sample ids according to prediction confidence
            probs = F.softmax(logits, dim=-1)
            selected_probs = torch.squeeze(
                torch.gather(probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)

            selected_probs = torch.where(unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()

            mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).cuda()
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                    torch.minimum(torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len)) # type: ignore

            # Sample masking tokens for next iteration
            masking = mask_by_random_topk(mask_len[0], selected_probs, self.choice_temperature * (1 - ratio))
            # Masks tokens with lower confidence.
            new_latents = torch.where(masking, self.mask_token_id, sampled_ids)
        else:
            new_latents = sampled_ids
        
        return SchedulerStepOutput(new_latents)
        
    
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
    ) -> SchedulerApproxGuidanceOutput:
        proposal_logits = logits + approx_guidance
        sched_out = self.step(latents, step, proposal_logits)
        new_latents = sched_out.new_latents
        
        newly_filled_positions = (latents != new_latents)
        print("Newly filled positions:", newly_filled_positions.sum(dim=1))
        
        log_prob_proposal = sum_masked_logits(
            logits=proposal_logits.log_softmax(dim=-1),
            preds=new_latents,
            mask=newly_filled_positions,
        )
        log_prob_diffusion = sum_masked_logits(
            logits=logits.log_softmax(dim=-1),
            preds=new_latents,
            mask=newly_filled_positions,
        )
        print("log prob proposal:", log_prob_proposal)
        print("log prob diffusion:", log_prob_diffusion)
        return SchedulerApproxGuidanceOutput(
            new_latents,
            log_prob_proposal,
            log_prob_diffusion,
        )


class ReMDMScheduler(BaseScheduler):
    def __init__(
        self,
        schedule,
        remask_strategy,
        eta,
        mask_token_id,
        temperature=1.0,
    ):
        self.schedule = schedule
        self.remask_strategy = remask_strategy
        self.eta = eta 
        self.temperature = temperature
        self.mask_token_id = mask_token_id
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        if self.schedule == "linear":
            self.alphas = 1 - torch.linspace(0, 1, num_inference_steps + 1)
        elif self.schedule == "cosine":
            self.alphas = 1 - torch.cos((math.pi/2) * (1 - torch.linspace(0, 1, num_inference_steps + 1)))
        else:
            raise ValueError(f"unknown masking schedule {self.schedule}")
    
    def step(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
    ) -> SchedulerStepOutput:
        B, L, C = logits.shape
        assert latents.shape == (B, L)
        
        t = self.num_inference_steps - step
        s = t - 1
        
        alpha_t = self.alphas[t]
        alpha_s = self.alphas[s]
        sigma_t_max = torch.clamp_max((1 - alpha_s) / alpha_t, 1.0)
        if self.remask_strategy == "max_cap":
            sigma_t = torch.clamp_max(sigma_t_max, self.eta)
        elif self.remask_strategy == "rescale":
            sigma_t = sigma_t_max * self.eta
        else:
            raise ValueError(f"unknown masking schedule {self.remask_strategy}")
        
        # z_t != m
        x_theta = F.one_hot(latents, num_classes=C).float()
        logits_z_t_neq_m = (
            torch.log(x_theta) +
            torch.log(1 - sigma_t)
        )
        logits_z_t_neq_m[..., self.mask_token_id] = (
            torch.log(sigma_t)
        )
        
        # z_t = m
        log_x_theta = (logits / self.temperature).log_softmax(dim=-1)
        logits_z_t_eq_m = (
            log_x_theta + 
            torch.log((alpha_s - (1 - sigma_t) * alpha_t) / (1 - alpha_t))
        )
        logits_z_t_eq_m[..., self.mask_token_id] = (
            torch.log((1 - alpha_s - sigma_t * alpha_t) / (1 - alpha_t))
        )
        
        z_t_neq_m = (latents != self.mask_token_id)
        p_theta_logits = torch.where(
            z_t_neq_m.unsqueeze(-1).expand(-1, -1, C),
            logits_z_t_neq_m,
            logits_z_t_eq_m,
        )
        assert torch.allclose(torch.exp(p_theta_logits).sum(dim=-1), torch.ones(B, L, device=logits.device)), (torch.exp(p_theta_logits).sum(dim=-1) - torch.ones(B, L, device=logits.device)).abs().max()
        diffusion_dist = torch.distributions.Categorical(logits=p_theta_logits) # type: ignore
        new_latents = diffusion_dist.sample()
        print("Unmasked:", (new_latents != self.mask_token_id).sum(dim=1))
        return SchedulerStepOutput(new_latents)
    
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
    ) -> SchedulerApproxGuidanceOutput:
        B, L, C = logits.shape
        assert latents.shape == (B, L)
        assert approx_guidance.shape == (B, L, C)
        
        t = self.num_inference_steps - step
        s = t - 1
        
        alpha_t = self.alphas[t]
        alpha_s = self.alphas[s]
        sigma_t_max = torch.clamp_max((1 - alpha_s) / alpha_t, 1.0)
        if self.remask_strategy == "max_cap":
            sigma_t = torch.clamp_max(sigma_t_max, self.eta)
        elif self.remask_strategy == "rescale":
            sigma_t = sigma_t_max * self.eta
        else:
            raise ValueError(f"unknown masking schedule {self.remask_strategy}")
        
        # z_t != m
        x_theta = F.one_hot(latents, num_classes=C).float()
        logits_z_t_neq_m = (
            torch.log(x_theta) +
            torch.log(1 - sigma_t)
        )
        logits_z_t_neq_m[..., self.mask_token_id] = (
            torch.log(sigma_t)
        )
        
        # z_t = m
        log_x_theta = (logits / self.temperature).log_softmax(dim=-1)
        logits_z_t_eq_m = (
            log_x_theta + 
            torch.log((alpha_s - (1 - sigma_t) * alpha_t) / (1 - alpha_t))
        )
        logits_z_t_eq_m[..., self.mask_token_id] = (
            torch.log((1 - alpha_s - sigma_t * alpha_t) / (1 - alpha_t))
        )
        
        z_t_neq_m = (latents != self.mask_token_id)
        p_theta_logits = torch.where(
            z_t_neq_m.unsqueeze(-1).expand(-1, -1, C),
            logits_z_t_neq_m,
            logits_z_t_eq_m,
        )
        assert torch.allclose(torch.exp(p_theta_logits).sum(dim=-1), torch.ones(B, L, device=logits.device))
        
        proposal_logits = (p_theta_logits + approx_guidance).log_softmax(dim=-1)
        assert torch.allclose(torch.exp(proposal_logits).sum(dim=-1), torch.ones(B, L, device=logits.device))
        
        # modify proposal logits to have the same mask schedule as the original logits
        proposal_logits[..., :self.mask_token_id] += (
            torch.logsumexp(p_theta_logits[..., :self.mask_token_id], dim=(1, 2), keepdim=True) - 
            torch.logsumexp(proposal_logits[..., :self.mask_token_id], dim=(1, 2), keepdim=True)
        )
        proposal_logits[..., :self.mask_token_id] = torch.where(
            proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1, keepdim=True) >= 0,
            proposal_logits[..., :self.mask_token_id].log_softmax(dim=-1),
            proposal_logits[..., :self.mask_token_id]
        )
        assert not (proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1) > 1e-6).any(), proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1).max()
        proposal_logits[..., self.mask_token_id] = (
            log1mexp(proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1).clamp_max(0))
        )
        assert torch.allclose(torch.exp(proposal_logits).sum(dim=-1), torch.ones(B, L, device=logits.device)), (torch.exp(proposal_logits).sum(dim=-1) - torch.ones(B, L, device=logits.device)).abs().max()
        # modify proposal logits to have the same mask schedule as the original logits
        
        proposal_dist = torch.distributions.Categorical(logits=proposal_logits) # type: ignore
        diffusion_dist = torch.distributions.Categorical(logits=p_theta_logits) # type: ignore
        
        new_latents = proposal_dist.sample()
        
        log_prob_proposal = proposal_dist.log_prob(new_latents).sum(dim=1)
        log_prob_diffusion = diffusion_dist.log_prob(new_latents).sum(dim=1)
        
        print("Unmasked:", (new_latents != self.mask_token_id).sum(dim=1))
        return SchedulerApproxGuidanceOutput(
            new_latents,
            log_prob_proposal,
            log_prob_diffusion,
        )
