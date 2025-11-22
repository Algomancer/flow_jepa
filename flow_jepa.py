import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from contextlib import contextmanager
from typing import List, Sequence, Tuple, Optional, NamedTuple, Any, Literal, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler # Added import
from torch import distributions as D
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import v2
from datasets import load_dataset 
import tqdm
from sklearn.decomposition import PCA

from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask

from torch.optim import Muon
import neptune

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

NUM_VIEWS, BATCH_SIZE, PATCH_SIZE = 4, 64, 8
INPUT_DIM = 3 * PATCH_SIZE * PATCH_SIZE

AUG_DIM = 32
FLOW_DIM = INPUT_DIM + AUG_DIM
MODEL_DIM = 384

VAE_LATENT_DIM = FLOW_DIM
USE_MLP_VAE = True
NUM_PRIOR_FLOWS = 12 
ADAM_LR = 1e-3
MUON_LR = 0.02

LAMBDA_FLOW, LAMBDA_INV, LAMBDA_JCR = 1.0, 1.0, 0.1

def get_local_rank(): return int(os.environ.get("LOCAL_RANK", 0))
def get_rank(): return int(os.environ.get("RANK", 0))
def get_world_size(): return int(os.environ.get("WORLD_SIZE", 1))
def is_master(): return get_rank() == 0

def init_neptune_if_available(project_name="YOURPROJECTNAME"):
    # Assume initialization succeeds if token is present
    api_token = os.environ.get("NEPTUNE_API_TOKEN", None)
    if api_token:
        return neptune.init_run(project=project_name, api_token=api_token)
    return None

def norm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

def seg_lengths(offsets: torch.Tensor) -> torch.Tensor:
    return offsets[1:] - offsets[:-1]

def seg_reduce(x: torch.Tensor, offsets: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    lengths = seg_lengths(offsets)
    return torch.segment_reduce(x, reduce=reduce, lengths=lengths)

def seg_broadcast(seg_vals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    lengths = seg_lengths(offsets)
    return torch.repeat_interleave(seg_vals, lengths, dim=0)

def seg_map(x: torch.Tensor, seg_vals: torch.Tensor, offsets: torch.Tensor, op: Callable = torch.sub) -> torch.Tensor:
    return op(x, seg_broadcast(seg_vals, offsets))

# --- Model Components ---
class CastedLinear(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x) if self.bias is not None else None)

class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 1, bias: bool = False):
        super().__init__()
        h = mult * dim
        self.c_fc = CastedLinear(dim, h, bias=bias)
        nn.init.trunc_normal_(self.c_fc.weight, std=0.02)
        if self.c_fc.bias is not None: nn.init.constant_(self.c_fc.bias, 0.)
        self.c_proj = CastedLinear(h, dim, bias=bias)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x: torch.Tensor):
        return self.c_proj(F.silu(self.c_fc(x)))

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, gated: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        hdim = self.num_heads * self.head_dim
        self.qkv = CastedLinear(dim, 3 * hdim, bias=False)
        self.gated = gated
        if gated:
            self.gate = CastedLinear(dim, hdim, bias=False)
            nn.init.trunc_normal_(self.gate.weight, std=0.02)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, mask: BlockMask):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        v = self.lambdas[0] * v + (self.lambdas[1] * ve.view_as(v) if ve is not None else 0)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask, scale=0.12).transpose(1, 2)
        if self.gated:
            gate = self.gate(x).view(b, t, self.num_heads, self.head_dim)
            y = y * torch.sigmoid(gate)
        return self.c_proj(y.contiguous().view(b, t, -1))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, idx: int, gated: bool = True):
        super().__init__()
        self.attn = Attention(dim, num_heads, gated) if idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, x0: torch.Tensor, mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, mask)
        return x + self.mlp(norm(x))

class Backbone(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, gated: bool = False, max_seq_len: int = 2**18):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, i, gated) for i in range(depth)])
        self.skip_weights = nn.Parameter(torch.ones(depth // 2)) if depth // 2 > 0 else None
        self.depth = depth 

    # Added mask=None default to support VAE usage
    def forward(self, x: torch.Tensor, mask: BlockMask = None, ve: torch.Tensor = None):
        x0, skips = x, []
        n = len(self.skip_weights) if self.skip_weights is not None else 0
        ve_list = [x] * self.depth if ve is None else [ve] * self.depth
        for i, blk in enumerate(self.blocks):
            if i >= n and self.skip_weights is not None:
                # Added safety check for skips list
                if skips:
                   x = x + self.skip_weights[i - n] * skips.pop()
            x = blk(x, ve_list[i], x0, mask)
            if i < n and self.skip_weights is not None:
                skips.append(x)
        return norm(x)

def bounded_exp_scale(raw_scale: torch.Tensor, scale_factor: float):
    max_log_s = math.log(float(scale_factor))
    log_s = torch.tanh(raw_scale) * max_log_s
    s = torch.exp(log_s)
    return s, log_s

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class CheckpointManager:
    def __init__(self, dirpath, max_keep=3, metric_name="loss", mode="min"):
        self.dirpath = dirpath
        self.max_keep = max_keep
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.recent_checkpoints = [] 
        if is_master(): os.makedirs(self.dirpath, exist_ok=True)
            
    def save(self, state, epoch, current_metric=None):
        if not is_master(): return
        filename = f"checkpoint_epoch_{epoch}.pt"
        filepath = os.path.join(self.dirpath, filename)
        if current_metric is not None: state[self.metric_name] = current_metric
        torch.save(state, filepath)
        self.recent_checkpoints.append(filepath)
        while len(self.recent_checkpoints) > self.max_keep:
            to_remove = self.recent_checkpoints.pop(0)
            if os.path.exists(to_remove):
                os.remove(to_remove)
        if current_metric is not None:
            is_best = (current_metric < self.best_metric) if self.mode == "min" else (current_metric > self.best_metric)
            if is_best:
                self.best_metric = current_metric
                torch.save(state, os.path.join(self.dirpath, "checkpoint_best.pt"))
        torch.save(state, os.path.join(self.dirpath, "checkpoint_latest.pt"))

class FlowState(NamedTuple):
    mask: Optional[torch.Tensor] = None
    context_embed: Optional[torch.Tensor] = None

class VariationalAugment(nn.Module):
    def __init__(self, input_dim: int, aug_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.aug_dim = aug_dim
        self.net = nn.Sequential(
            CastedLinear(input_dim, hidden_dim),
            nn.SiLU(),
            CastedLinear(hidden_dim, hidden_dim),
            nn.SiLU(),
            CastedLinear(hidden_dim, aug_dim * 2)
        )
        def _init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, CastedLinear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.net.apply(_init_weights)
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def context(self, x: torch.Tensor, pos_embed: torch.Tensor, state: FlowState) -> FlowState:
        return state

    def forward(self, x: torch.Tensor, state: FlowState, mask: BlockMask):
        stats = self.net(x)
        mu, log_sig = stats.chunk(2, dim=-1)
        
        sig = torch.exp(torch.clamp(log_sig, -10, 10))
        
        eps = torch.randn_like(mu)
        u = mu + sig * eps
        
        z = torch.cat([x, u], dim=-1)
        
        log_q = -0.5 * math.log(2 * math.pi) - log_sig - 0.5 * eps.square()
        ldj = -log_q.sum(dim=-1).double()
        
        return z, ldj

    def inverse(self, z: torch.Tensor, state: FlowState, mask: BlockMask):
        x = z[:, :self.input_dim]
        ldj = torch.zeros(x.size(0), device=x.device, dtype=torch.float64)
        return x, ldj

class RandomMaskImageAffineBijection(nn.Module):
    def __init__(self, input_dim: int, dim: int, depth: int = 2, num_heads: int = 4, scale_factor: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        self.d_kept = input_dim // 2
        self.scale_factor = scale_factor
        
        self.null_vector = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.null_vector, std=0.02)

        self.input_proj = nn.Linear(input_dim * 2, dim)
        self.output_head = nn.Linear(dim, 2 * input_dim)
        
        self.backbone = Backbone(dim, depth, num_heads)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        if self.input_proj.bias is not None:
             nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.zeros_(self.output_head.weight)
        if self.output_head.bias is not None:
            nn.init.zeros_(self.output_head.bias)

    def context(self, x: torch.Tensor, pos_embed: torch.Tensor, state: FlowState) -> FlowState:
        mask = state.mask
        
        null_vec = self.null_vector
        if x.ndim == 2:
            null_vec = null_vec.squeeze(1)
            
        x_filled = torch.where(mask, x, null_vec)
        mask_float = mask.to(dtype=x.dtype)
        net_input = torch.cat([x_filled, mask_float], dim=-1)
        h = self.input_proj(net_input)
        h = h + pos_embed
        
        return FlowState(mask=mask, context_embed=h)

    def forward(self, x: torch.Tensor, state: FlowState, mask_attn: BlockMask):
        mask = state.mask
        inv_mask = ~mask
        
        context = state.context_embed
        h_in = context.reshape(1, -1, context.size(-1))
        
        h_out = self.backbone(h_in, mask_attn)
        h_out = h_out.view_as(context)
        
        params = self.output_head(h_out)
        bias, raw_scale = params.chunk(2, dim=-1)
        scale, log_s = bounded_exp_scale(raw_scale, self.scale_factor)
        y_target = (x + bias) * scale
        y = torch.where(mask, x, y_target)
        
        log_det = (log_s * inv_mask).sum(dim=-1).double()
        return y, log_det

    def inverse(self, z: torch.Tensor, state: FlowState, mask_attn: BlockMask):
        mask = state.mask
        inv_mask = ~mask
        
        context = state.context_embed
        h_in = context.reshape(1, -1, context.size(-1))
        
        h_out = self.backbone(h_in, mask_attn)
        h_out = h_out.view_as(context)
        
        params = self.output_head(h_out)
        bias, raw_scale = params.chunk(2, dim=-1)
        scale, log_s = bounded_exp_scale(raw_scale, self.scale_factor)
        x_target = (z / scale) - bias
        x = torch.where(mask, z, x_target)
        
        log_det = -(log_s * inv_mask).sum(dim=-1).double()
        return x, log_det


class FlowSequence(nn.Module):
    def __init__(self, flows: nn.ModuleList):
        super().__init__()
        self.flows = flows

    @torch.compile(fullgraph=True)
    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor, mask_attn: BlockMask, states: List[FlowState]):
        ldj = torch.zeros(x.size(0), device=x.device, dtype=torch.float64)
        out_states = [] 
        
        for i, flow in enumerate(self.flows):
            state_in = states[i]
            state_out = flow.context(x, pos_embed, state_in)
            out_states.append(state_out)
            
            x, ldj_i = flow(x, state_out, mask_attn)
            ldj = ldj + ldj_i
            
        return x, ldj, out_states
    
    @torch.compile()
    def inverse(self, z: torch.Tensor, pos_embed: torch.Tensor, mask_attn: BlockMask, states: List[FlowState]):
        ldj = torch.zeros(z.size(0), device=z.device, dtype=torch.float64)
        
        for i, (flow, state_in) in enumerate(zip(reversed(self.flows), reversed(states))):
            
            state_out = flow.context(z, pos_embed, state_in)
            
            z, ldj_i = flow.inverse(z, state_out, mask_attn)
            ldj = ldj + ldj_i
            
        return z, ldj


class ConditionalPriorVAE(nn.Module):
    """
    A Variational Autoencoder used as a learned prior for the Normalizing Flow.
    Swappable between Linear (default) and Backbone architectures. Linear gives you the manifold flattening property.
    Supports a Flow-based Prior p(z) using RandomMaskImageAffineBijection.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512, backbone_factory: Callable = None, num_prior_flows: int = 0, global_model_dim: int = 384):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_backbone = (backbone_factory is not None)

        # --- Encoder q(z_vae | z_patches) ---
        if self.use_backbone:
            self.encoder_backbone = backbone_factory()
            backbone_dim = hidden_dim
            self.encoder_proj_in = CastedLinear(input_dim, backbone_dim)
            self.encoder_head = CastedLinear(backbone_dim, latent_dim * 2)
        else:
            self.encoder_backbone = nn.Identity()
            self.encoder_proj_in = None
            self.encoder_head = CastedLinear(input_dim, latent_dim * 2)

        # --- Decoder p(z_patches | z_vae) ---
        if self.use_backbone:
            self.decoder_backbone = backbone_factory()
            backbone_dim = hidden_dim
            self.decoder_proj_in = CastedLinear(latent_dim, backbone_dim)
            self.decoder_head = CastedLinear(backbone_dim, input_dim * 2) # Mean and LogVar
        else:
             # Default MLP Decoder
            self.decoder_backbone = nn.Sequential(
                CastedLinear(latent_dim, hidden_dim),
                nn.SiLU(),
                CastedLinear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.decoder_proj_in = None
            self.decoder_head = CastedLinear(hidden_dim, input_dim * 2)

        # --- Learned Prior Flow p(z) ---
        self.num_prior_flows = num_prior_flows
        self.prior_hidden_dim = hidden_dim 
        
        if num_prior_flows > 0:
            # Project global pos embed to prior hidden dim if needed, or just to ensure alignment
            self.prior_pos_proj = nn.Linear(global_model_dim, self.prior_hidden_dim)
            
            layers = []
            # Stack Bijections directly for the latent space
            for _ in range(num_prior_flows):
                layers.append(RandomMaskImageAffineBijection(
                    input_dim=latent_dim, 
                    dim=self.prior_hidden_dim, 
                    depth=2, # Keep it relatively shallow for prior
                    num_heads=4
                ))
            self.prior_flow = FlowSequence(nn.ModuleList(layers))
        else:
            self.prior_flow = None

        self._init_weights()

    def _init_weights(self):
        def init(m):
            if isinstance(m, nn.Linear) or isinstance(m, CastedLinear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.apply(init)
        # Initialize heads near zero
        nn.init.trunc_normal_(self.encoder_head.weight, std=0.001)
        if self.encoder_head.bias is not None: nn.init.zeros_(self.encoder_head.bias)
        nn.init.trunc_normal_(self.decoder_head.weight, std=0.001)
        if self.decoder_head.bias is not None: nn.init.zeros_(self.decoder_head.bias)
        
        if self.num_prior_flows > 0:
            nn.init.trunc_normal_(self.prior_pos_proj.weight, std=0.02)
            nn.init.zeros_(self.prior_pos_proj.bias)


    def _process_with_backbone(self, backbone_module, x: torch.Tensor, proj_in, mask: BlockMask = None):
        h = proj_in(x)
        is_flat = (h.ndim == 2)
        if is_flat:
            h = h.unsqueeze(0)
            
        if isinstance(backbone_module, Backbone):
                h = backbone_module(h, mask)
        else:
                h = backbone_module(h) 

        if is_flat:
            h = h.squeeze(0)
        return h

    def encode(self, x: torch.Tensor, mask: BlockMask = None):
        if self.use_backbone:
            h = self._process_with_backbone(self.encoder_backbone, x, self.encoder_proj_in, mask)
        else:
            h = self.encoder_backbone(x)
            
        params = self.encoder_head(h)
        mu, logvar = params.chunk(2, dim=-1)
        
        # Stability fix: clamp logvar to prevent explosion/collapse
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, mask: BlockMask = None):
        if self.use_backbone:
            h = self._process_with_backbone(self.decoder_backbone, z, self.decoder_proj_in, mask)
        else:
            h = self.decoder_backbone(z)
            
        params = self.decoder_head(h)
        p_mu, p_logvar_raw = params.chunk(2, dim=-1)
        
        # Clamp logvar for stability.
        p_logvar = torch.clamp(p_logvar_raw, -8.0, 5.0) 
        
        return p_mu, p_logvar

    @torch.compile()
    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor, mask: BlockMask = None):
        # 1. Encode q(latent|x)
        q_mu, q_logvar = self.encode(x, mask)
        
        # 2. Reparameterize
        latent_z = self.reparameterize(q_mu, q_logvar)
        
        # 3. Decode p(x|latent)
        p_mu, p_logvar = self.decode(latent_z, mask)
        
        # A. Reconstruction loss (NLL under the conditional Gaussian)
        rec_nll_dims = 0.5 * (math.log(2 * math.pi) + p_logvar + (x - p_mu).pow(2) * torch.exp(-p_logvar))
        rec_nll = rec_nll_dims.sum(dim=-1) 
        
        # B. KL divergence KL(q(latent|x) || p(latent))
        if self.prior_flow is None:
            # Analytic KL for Standard Gaussian Prior
            kl_div_dims = -0.5 * (1 + q_logvar - q_mu.pow(2) - q_logvar.exp())
            kl_div = kl_div_dims.sum(dim=-1)
        else:
            # Monte Carlo KL estimate for Flow Prior
            # KL(q||p) = E_q [ log q(z|x) - log p(z) ]
            
            # Stable log q(z|x) calculation for Diagonal Gaussian
            std = torch.exp(0.5 * q_logvar)
            eps_implicit = (latent_z - q_mu) / (std + 1e-6) 
            log_q_z = -0.5 * (math.log(2 * math.pi) + q_logvar + eps_implicit.pow(2))
            log_q_z = log_q_z.sum(dim=-1)

            # Prepare states for prior flow
            batch_n = latent_z.size(0)
            p_pos_embed = self.prior_pos_proj(pos_embed) 
            # Generate complementary masks for prior flow 
            comp_masks_prior = generate_complementary_masks(
                batch_size=batch_n, 
                num_tokens=1,
                input_dim=self.latent_dim,
                num_layers=self.num_prior_flows,
                device=latent_z.device
            )
            prior_states = [FlowState(mask=m.squeeze(1)) for m in comp_masks_prior]

            # log p(z) using Prior Flow
            # z -> prior_flow.inverse -> eps_prior
            eps_prior, log_det_inv = self.prior_flow.inverse(latent_z, p_pos_embed, mask, states=prior_states)
            
            # log p(eps_prior) (Standard Normal)
            log_p_eps = -0.5 * (math.log(2 * math.pi) + eps_prior.pow(2))
            log_p_eps = log_p_eps.sum(dim=-1)
            
            log_p_z = log_p_eps + log_det_inv
            
            kl_div = log_q_z - log_p_z

        # Total VAE NLL (Negative ELBO) per patch
        vae_nll = rec_nll + kl_div
        
        return vae_nll, rec_nll.mean(), kl_div.mean()


    @torch.compile()
    def sample_prior(self, num_samples: int, pos_embed: torch.Tensor, device, dtype, mask: BlockMask = None):
        """Samples from the learned prior distribution P(z)."""
        if self.prior_flow is None:
            # 1. Sample from standard Gaussian p(latent)
            latent_z = torch.randn(num_samples, self.latent_dim, device=device, dtype=dtype)
        else:
            # 1. Sample from standard Gaussian p(eps)
            eps = torch.randn(num_samples, self.latent_dim, device=device, dtype=dtype)
            
            # Prepare states
            p_pos_embed = self.prior_pos_proj(pos_embed)
            comp_masks_prior = generate_complementary_masks(
                batch_size=num_samples, 
                num_tokens=1,
                input_dim=self.latent_dim,
                num_layers=self.num_prior_flows,
                device=device
            )
            prior_states = [FlowState(mask=m.squeeze(1)) for m in comp_masks_prior]
            
            # 2. Transform through flow z = f(eps)
            # FlowSequence.forward returns (x, ldj, out_states), so we unpack 3 values
            latent_z, _, _ = self.prior_flow(eps, p_pos_embed, mask, states=prior_states)
        
        # 3. Decode to get parameters p(x|latent)
        p_mu, p_logvar = self.decode(latent_z, mask)
        
        # 4. Sample z_patches (the flow latent)
        z_patches = self.reparameterize(p_mu, p_logvar)
        return z_patches


class MultiViewFlow(nn.Module):
    def __init__(self, flow_encoder: FlowSequence, input_dim: int = 768, num_classes: int = 1000, probe_dim: int = 16, vae_latent_dim: int = 64, model_dim: int = 192, use_mlp_vae: bool = True, vae_backbone_factory: Callable = None, num_prior_flows: int = 0):
        super().__init__()
        self.flow_encoder = flow_encoder
        
        if not use_mlp_vae and vae_backbone_factory is None:
             use_mlp_vae = True # Silent fallback

        self.vae_prior = ConditionalPriorVAE(
            input_dim=input_dim, # FLOW_DIM
            latent_dim=vae_latent_dim, 
            hidden_dim=model_dim,
            backbone_factory=vae_backbone_factory if not use_mlp_vae else None,
            num_prior_flows=num_prior_flows,
            global_model_dim=model_dim # Use same model dim
        )
        # ---------------------

        self.sigreg = SIGReg()
        self.probe = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.probe.weight); 
        if self.probe.bias is not None: nn.init.zeros_(self.probe.bias)
            
        self.probe_dim = probe_dim
        
        # Idk if batch norm is needed in this recipe, some ssl needs it, but it can probably be removed.
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.SiLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.SiLU(),
            nn.Linear(2048, probe_dim)
        )
        # Init
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor, mask: BlockMask, 
                offsets: torch.Tensor, states: List[FlowState],
                num_views: int = 1, labels: torch.Tensor = None):
        
        T, D_in = x.shape
        
        # Forward pass
        z_patches, ldj_flow, _ = self.flow_encoder(x, pos_embed, mask, states=states)

        vae_nll_per_patch, avg_rec_nll, avg_kl_div = self.vae_prior(z_patches, pos_embed, mask=mask)
        total_nll = vae_nll_per_patch.sum() - ldj_flow.sum()

        # Normalize by input pixels (D_in)
        flow_loss = total_nll / (T * D_in)

        m_z = torch.segment_reduce(z_patches, reduce="mean", offsets=offsets) 

        z_gap = self.proj(m_z)
        proj_loss = self.sigreg(z_gap)
        
        num_view_images = z_gap.shape[0]
        B_eff = num_view_images // num_views
        
        inv_loss = torch.tensor(0.0, device=x.device)

        z_gap_grouped = z_gap[:B_eff * num_views].view(B_eff, num_views, self.probe_dim)
        # lets leave Aleatoric uncertainty weighting for another paper or smth :^)
        weights = torch.full_like(z_gap_grouped[..., :1], 1.0 / num_views)
        
        z_anchor = (z_gap_grouped * weights).sum(dim=1, keepdim=True)
        inv_loss = (z_anchor - z_gap_grouped).square().mean()
    
        probe_logits = self.probe(m_z.detach()) 
        
        probe_loss = torch.tensor(0.0, device=x.device)
        acc = 0.0

        # Robust handling of labels matching (necessary due to original collate function behavior)
        if labels is not None:
            labels_expanded = labels.repeat_interleave(num_views)
            
            # Ensure shapes match before calculating loss
            if probe_logits.shape[0] == labels_expanded.shape[0]:
                probe_loss = F.cross_entropy(probe_logits, labels_expanded)
                preds = probe_logits.argmax(dim=-1)
                acc = (preds == labels_expanded).float().mean()

        # Return updated metrics including weights
        return flow_loss, inv_loss + proj_loss * 0.02, probe_loss, acc, avg_rec_nll, avg_kl_div, weights

def ids_from_offsets(N: int, offsets: torch.Tensor) -> torch.LongTensor:
    # Matches original implementation exactly
    return torch.bucketize(torch.arange(N, device=offsets.device), offsets[:-1], right=True) - 1

def build_offsets(lengths: Sequence[int], device=None) -> torch.Tensor:
    out = torch.zeros(len(lengths) + 1, dtype=torch.int64, device=device)
    if lengths: out[1:] = torch.as_tensor(lengths, dtype=torch.int64, device=device).cumsum(dim=0)
    return out

def patchify_image(img: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    C, H, W = img.shape
    Hp, Wp = H // patch_size, W // patch_size
    img = img[:, :Hp * patch_size, :Wp * patch_size].contiguous()
    patches = img.view(C, Hp, patch_size, Wp, patch_size).permute(1, 3, 0, 2, 4).contiguous()
    patches = patches.view(Hp * Wp, C, patch_size, patch_size)
    ys = torch.arange(Hp, device=img.device).unsqueeze(1).expand(Hp, Wp)
    xs = torch.arange(Wp, device=img.device).unsqueeze(0).expand(Hp, Wp)
    coords = torch.stack([ys, xs], dim=-1).view(-1, 2)
    return patches, coords

def get_1d_sincos_pos_embed_torch(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    dim = embed_dim // 2
    omega = torch.arange(dim, dtype=torch.float32, device=pos.device)
    omega /= dim; omega = 1.0 / (10000 ** omega)
    out = pos.float().unsqueeze(1) * omega.unsqueeze(0)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb.to(torch.bfloat16)

def get_2d_sincos_pos_embed_from_coords(embed_dim: int, coords: torch.Tensor) -> torch.Tensor:
    half = embed_dim // 2
    return torch.cat([get_1d_sincos_pos_embed_torch(half, coords[:, 0]), 
                      get_1d_sincos_pos_embed_torch(half, coords[:, 1])], dim=1)

def unpatchify_from_coords(patches, coords, patch_size):
    # Handle potentially flattened patches (N, D) during visualization
    if patches.ndim == 2:
        N, D = patches.shape
        p = patch_size
        C = D // (p*p)
        if C * p * p != D:
             return None # Cannot reliably unpatchify
        patches = patches.view(N, C, p, p)
    else:
         N, C, p, _ = patches.shape

    if N == 0:
        return torch.empty(C, 0, 0, dtype=patches.dtype, device=patches.device)

    H_p = int(coords[:, 0].max().item()) + 1
    W_p = int(coords[:, 1].max().item()) + 1
    img = torch.zeros(C, H_p * patch_size, W_p * patch_size, dtype=patches.dtype, device=patches.device)
    for i in range(N):
        y, x = int(coords[i, 0].item()), int(coords[i, 1].item())
        img[:, y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size] = patches[i]
    return img

def save_tensor_image(img, path):
    if img is None: return
    mean = IMAGENET_MEAN.to(img.device).view(3, 1, 1)
    std = IMAGENET_STD.to(img.device).view(3, 1, 1)
    img = (img * std + mean).clamp(0, 1)
    arr = (img * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)

def make_panel(images): 
    valid_images = [img if img.dim() == 3 else img.squeeze(0) for img in images if img is not None]
    if not valid_images: return None
    return torch.cat(valid_images, dim=2)

def visualize_reconstruction(flow_model, patches, coords, offsets, patch_size, patches_flat, pos_embed, mask, step, out_path="result_vae_prior.jpg"):
    if step % 100 != 0 or offsets.numel() < 3: return
    start0, end0 = int(offsets[1].item()), int(offsets[2].item())
    if end0 <= start0: return
    
    input_dim = patches_flat.shape[-1]
    flow_dim = flow_model.vae_prior.input_dim
    aug_dim = flow_dim - input_dim
    
    T_slice = end0 - start0
    device = patches.device
    
    with torch.no_grad():
        # --- 2. Prepare States (Masks) ---
        num_total_layers = len(flow_model.flow_encoder.flows)
        num_flow_layers = num_total_layers - (1 if aug_dim > 0 else 0)
        
        vis_states = []
        
        # If augmentation layer exists (Layer 0)
        if aug_dim > 0:
             # Layer 0 mask (dummy). Shape [T_slice, INPUT_DIM]
             mask_layer0 = torch.zeros(T_slice, input_dim, device=device, dtype=torch.bool)
             vis_states.append(FlowState(mask=mask_layer0))

        
        # Masks for subsequent flow layers (Size FLOW_DIM)
        masks_flow = generate_complementary_masks(T_slice, 1, flow_dim, num_flow_layers, device)
        masks_flow = [m.squeeze(1) for m in masks_flow]
        vis_states.extend([FlowState(mask=m) for m in masks_flow])
        

        # Define a local attention mask
        def full_attn_mod(b, h, q_idx, kv_idx): return q_idx >= 0 
        local_mask = create_block_mask(full_attn_mod, None, None, T_slice, T_slice, device=device, _compile=True)

        local_pos_embed = pos_embed[start0:end0]

        # --- 3. Forward (Input -> Latent) ---
        z_patches, _, _ = flow_model.flow_encoder(patches_flat[start0:end0], local_pos_embed, local_mask, states=vis_states)
        
        # --- 4. Inverse (Reconstruction) ---
        x_recon, _ = flow_model.flow_encoder.inverse(z_patches, local_pos_embed, local_mask, states=vis_states)
    
        # --- 5. Sampling (From VAE Prior) ---
        z_sample = flow_model.vae_prior.sample_prior(T_slice, local_pos_embed, device=device, dtype=z_patches.dtype, mask=local_mask)

        
        # Prepare states for generation (fresh random masks)
        gen_states = []
        if aug_dim > 0:
             gen_states.append(FlowState(mask=vis_states[0].mask)) 

        gen_masks_flow = generate_complementary_masks(T_slice, 1, flow_dim, num_flow_layers, device)
        gen_masks_flow = [m.squeeze(1) for m in gen_masks_flow]
        gen_states.extend([FlowState(mask=m) for m in gen_masks_flow])

        # Inverse flow (Generation)
        x_gen, _ = flow_model.flow_encoder.inverse(z_sample, local_pos_embed, local_mask, states=gen_states)
        
        def get_img(p_tensor): 
            return unpatchify_from_coords(p_tensor, coords[start0:end0], patch_size)
        
        img_orig = get_img(patches[start0:end0])
        img_recon = get_img(x_recon)
        img_gen = get_img(x_gen)
        
        # Visualize Z
        if flow_dim != input_dim:
            z_sliced = z_patches[:, :input_dim]
            img_z = get_img(z_sliced)
        else:
            img_z = get_img(z_patches)
        
        # PCA Visualization (Restoring original implementation)
        pca_data = z_patches.float().cpu().numpy()
        if pca_data.shape[0] > 3 and img_orig is not None:
            pca_proj = PCA(n_components=3).fit_transform(pca_data)
            H_p = int(coords[start0:end0, 0].max().item()) + 1
            W_p = int(coords[start0:end0, 1].max().item()) + 1
            pca_grid = np.zeros((H_p, W_p, 3))

            # Adjust indexing to match original script style (start0+i)
            for i in range(pca_data.shape[0]):
                y, x = int(coords[start0+i, 0].item()), int(coords[start0+i, 1].item())
                if y < H_p and x < W_p: pca_grid[y, x] = pca_proj[i]
            
            pca_img = torch.from_numpy(pca_grid).permute(2, 0, 1).unsqueeze(0)
            p_min = pca_img.view(3, -1).min(dim=1, keepdim=True)[0].view(1, 3, 1, 1)
            p_max = pca_img.view(3, -1).max(dim=1, keepdim=True)[0].view(1, 3, 1, 1)
            pca_img = (pca_img - p_min) / (p_max - p_min + 1e-6)
            pca_img = F.interpolate(pca_img, size=img_orig.shape[-2:], mode='nearest').squeeze(0)

        else:
            pca_img = torch.zeros_like(img_orig) if img_orig is not None else None

        panel = make_panel([img_orig.float().cpu(), 
                            img_z.float().cpu() if img_z is not None else None, 
                            img_recon.float().cpu() if img_recon is not None else None, 
                            img_gen.float().cpu() if img_gen is not None else None, 
                            pca_img.float().cpu() if pca_img is not None else None])
        save_tensor_image(panel, out_path)

class MultiViewHFDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", num_views=2, patch_size=16, img_size=128):
        # Fixed: Removed rank/world_size arguments to prevent manual sharding issues
        # Fixed: Removed list comprehension to prevent loading entire dataset into memory
        self.ds = load_dataset("johnowhitaker/imagenette2-320", split=split)
        self.num_views = num_views
        self.aug = v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        img_pil = item["image"].convert("RGB")
        views = [self.aug(img_pil) for _ in range(self.num_views)]
        return torch.stack(views), item["label"]

def multiview_collate(batch, patch_size=16):
    # Matches original implementation exactly
    all_patches, all_coords, all_lengths, all_labels = [], [], [], []
    for views, label in batch:
        all_labels.append(label)
        for v in range(views.shape[0]):
            patches, coords = patchify_image(views[v], patch_size)
            if patches.numel() == 0: continue
            all_patches.append(patches)
            all_coords.append(coords)
            all_lengths.append(patches.shape[0])
    if not all_patches: return None
    patches = torch.cat(all_patches, dim=0)
    coords = torch.cat(all_coords, dim=0)
    offsets = build_offsets(all_lengths, device=patches.device)
    return patches, coords, offsets, torch.tensor(all_labels, dtype=torch.long)

def generate_complementary_masks(batch_size, num_tokens, input_dim, num_layers, device):
    """Generates complementary masks [B, T, D]."""
    masks = []
    d_kept = input_dim // 2
    num_even = (num_layers + 1) // 2
    noise = torch.rand(num_even, batch_size, num_tokens, input_dim, device=device)
    _, indices = torch.topk(noise, k=d_kept, dim=-1)
    base_masks = torch.zeros(num_even, batch_size, num_tokens, input_dim, device=device, dtype=torch.bool)
    base_masks.scatter_(-1, indices, True)
    even_idx = 0
    for i in range(num_layers):
        if i % 2 == 0:
            m = base_masks[even_idx]; masks.append(m); even_idx += 1
        else:
            m = ~masks[-1]; masks.append(m)
    return masks

# --- Main ---
if __name__ == "__main__":
    dist.init_process_group("nccl")
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(42 + get_rank())
    torch.set_float32_matmul_precision('high')

    run = init_neptune_if_available() if is_master() else None
    if is_master(): print(f"Training on {get_world_size()} GPUs.")

    ds = MultiViewHFDataset(split="train", num_views=NUM_VIEWS, patch_size=PATCH_SIZE)
    
    sampler = DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=True)
    
    loader = DataLoader(
        ds, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        shuffle=False, 
        num_workers=8, 
        persistent_workers=True, 
        pin_memory=True, 
        collate_fn=lambda b: multiview_collate(b, patch_size=PATCH_SIZE), 
        drop_last=True
    )

    layers = nn.ModuleList([
        VariationalAugment(INPUT_DIM, AUG_DIM),
        RandomMaskImageAffineBijection(input_dim=FLOW_DIM, dim=MODEL_DIM, depth=8),
        *[RandomMaskImageAffineBijection(input_dim=FLOW_DIM, dim=MODEL_DIM) for _ in range(22)],
        RandomMaskImageAffineBijection(input_dim=FLOW_DIM, dim=MODEL_DIM, depth=8),
    ])
    
    flow_encoder = FlowSequence(layers)

    vae_backbone_factory = None
    if not USE_MLP_VAE:
        def create_vae_backbone():
            return Backbone(dim=MODEL_DIM, depth=12, num_heads=6)
        vae_backbone_factory = create_vae_backbone

    model = MultiViewFlow(
        flow_encoder, 
        input_dim=FLOW_DIM, 
        num_classes=10, 
        probe_dim=16,
        vae_latent_dim=VAE_LATENT_DIM,
        model_dim=MODEL_DIM,
        use_mlp_vae=USE_MLP_VAE,
        vae_backbone_factory=vae_backbone_factory,
        num_prior_flows=NUM_PRIOR_FLOWS
        ).to(device).to(torch.bfloat16)

    # Print parameter count
    if is_master():
        param_count = sum(p.numel() for p in model.parameters())
        vae_params = sum(p.numel() for p in model.vae_prior.parameters())
        print(f"Model parameter count: {param_count:,}")
        print(f"  VAE Prior params: {vae_params:,} (Using {'MLP' if USE_MLP_VAE else 'Backbone'})")


    for p in model.parameters(): dist.broadcast(p.data, src=0)

    probe_params = list(model.probe.parameters())
    probe_ids = set(id(p) for p in probe_params)
    muon_params, adam_params = [], []
    for n, p in model.named_parameters():
        if id(p) in probe_ids or not p.requires_grad: continue
        if p.ndim == 2: muon_params.append(p)
        else: adam_params.append(p) 


    adamw_param_groups = [
        {'params': adam_params, 'lr': ADAM_LR, 'weight_decay': 0.0, 'base_lr': ADAM_LR},
        {'params': probe_params, 'lr': ADAM_LR, 'weight_decay': 1e-5, 'base_lr': ADAM_LR}
    ]
    optimizer1 = AdamW(adamw_param_groups)
    
    optimizer2 = Muon(muon_params, lr=MUON_LR, momentum=0.95)
    muon_base_lr = MUON_LR

    ckpt_manager = CheckpointManager("checkpoints_vae_prior", max_keep=3)

    # Linear Warmup Counter (Matching original schedule)
    global_step = 0
    warmup_steps = 1000
    noise_anneal_steps = 2000
    noise_start = 1.0
    noise_end = 0.01

    # Training loop structure matches original
    pbar = tqdm.tqdm(total=len(loader), dynamic_ncols=True, disable=not is_master())
    for i in tqdm.trange(2000, dynamic_ncols=True, disable=not is_master()):
        # Fixed: Set epoch for sampler to ensure proper shuffling per epoch
        sampler.set_epoch(i)
        
        # Reset pbar for the new epoch
        pbar.reset() 
        
        for step, batch in enumerate(loader):
            if batch is None: continue
            patches, coords, offsets, labels = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            patches = patches.to(torch.bfloat16)

            patches_flat = patches.view(patches.shape[0], -1)
            batch_ids = ids_from_offsets(patches.shape[0], offsets)

            pos_embed = get_2d_sincos_pos_embed_from_coords(MODEL_DIM, coords)

            def mask_mod(b, h, q_idx, kv_idx): return batch_ids[q_idx] == batch_ids[kv_idx]
            mask = create_block_mask(mask_mod, None, None, patches_flat.size(0), patches_flat.size(0), device=patches.device, _compile=True)

            # --- Explicit Mask Generation for Mixed Dimensions ---
            # Replicating the exact syntax from the original script
            batch_n = patches_flat.size(0)

            mask_layer0 = torch.zeros(1, batch_n, 1, INPUT_DIM, device=device, dtype=torch.bool) 

            # 2. Masks for Layers 1..N (Flow): Size FLOW_DIM
            comp_masks_flow = generate_complementary_masks(
                batch_size=batch_n, 
                num_tokens=1,
                input_dim=FLOW_DIM,
                num_layers=len(layers) - 1, # Subtract augment layer
                device=device
            )

            # Combine into one list of states
            train_states = [FlowState(mask=mask_layer0.squeeze(1))] + \
                           [FlowState(mask=m.squeeze(1)) for m in comp_masks_flow]

            optimizer1.zero_grad(); optimizer2.zero_grad()

            # ----- Add noise annealed (Matching original schedule) -----
            if global_step < noise_anneal_steps:
                t = global_step / noise_anneal_steps
                noise_std = noise_start + (noise_end - noise_start) * t
            else:
                noise_std = noise_end
            if noise_std > 0:
                patches_flat = patches_flat + noise_std * torch.randn_like(patches_flat)

            flow_loss, inv_loss, probe_loss, acc, vae_rec_nll, vae_kl_div, weights = model(
                patches_flat, pos_embed, mask, offsets, 
                states=train_states, 
                num_views=NUM_VIEWS, labels=labels
            )

            total_loss = (LAMBDA_FLOW * flow_loss) + (LAMBDA_INV * inv_loss) + probe_loss
            total_loss.backward()

            handles = []
            for p in model.parameters():
                if p.grad is not None: handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True))
            for h in handles: h.wait()

            if warmup_steps > 0:
                global_step += 1
                warmup_factor = min(global_step / warmup_steps, 1.0)
                # optimizer1
                for group in optimizer1.param_groups:
                    base_lr = group.get('base_lr', group['lr'])
                    group['lr'] = base_lr * warmup_factor
                # optimizer2
                # Check existence before iterating (though assumed available)
                if hasattr(optimizer2, 'param_groups'):
                    for group in optimizer2.param_groups:
                        group['lr'] = muon_base_lr * warmup_factor

            optimizer1.step(); optimizer2.step()

            if is_master():
                pbar.update(1)
                # Updated logging (UPDATED)
                pbar.set_postfix({"flow": f"{flow_loss.item():.2f}", "inv": f"{inv_loss.item():.2f}", "acc": f"{acc:.1%}", "KL": f"{vae_kl_div.item():.3f}"})
                if run:
                    run["train/flow_loss"].append(flow_loss.item())
                    run["train/inv_loss"].append(inv_loss.item())
                    run["train/probe_loss"].append(probe_loss.item())
                    run["train/acc"].append(acc)
                    # New metrics
                    run["train/vae_rec_nll"].append(vae_rec_nll.item())
                    run["train/vae_kl_div"].append(vae_kl_div.item())

                # Visualization calls (UPDATED with visualize_weights)
                visualize_reconstruction(model, patches, coords, offsets, PATCH_SIZE, patches_flat, pos_embed, mask, step)

        if is_master() and (i % 5 == 0):
            ckpt_manager.save({'model': model.state_dict(), 'epoch': i}, i, total_loss.item())
    dist.destroy_process_group()
    if run:
        run.stop()
