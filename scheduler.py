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
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
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
    valid = (preds >= 0) & (preds < C)
    # Replace invalid preds with a dummy index (0), which we will mask later
    safe_preds = preds.masked_fill(~valid, 0)
    # Gather logits at predicted indices
    selected = torch.gather(logits, dim=2, index=safe_preds.unsqueeze(-1)).squeeze(-1)
    # Zero out contributions from invalid preds and masked positions
    selected = selected * valid * mask
    # Sum over time dimension
    return selected.sum(dim=1)


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
        return SchedulerApproxGuidanceOutput(
            new_latents,
            log_prob_proposal,
            log_prob_diffusion,
        )


class ReMDMScheduler(BaseScheduler):
    def __init__(self, ):
        pass
    
    def step(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
    ) -> SchedulerStepOutput:
        pass
    
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
    ) -> SchedulerApproxGuidanceOutput:
        pass