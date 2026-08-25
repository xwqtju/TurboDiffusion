""" 
Copyright (c) 2025 by TurboDiffusion team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
  author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
"""

import argparse

import torch
import torch.nn.functional as F
from rcm.utils.model_utils import load_state_dict
from rcm.networks.wan2pt1 import (
    WanModel as WanModel2pt1,
    WanLayerNorm as WanLayerNorm2pt1,
    WanRMSNorm as WanRMSNorm2pt1,
    WanSelfAttention as WanSelfAttention2pt1
)
from rcm.networks.wan2pt2 import (
    WanModel as WanModel2pt2,
    WanLayerNorm as WanLayerNorm2pt2,
    WanRMSNorm as WanRMSNorm2pt2,
    WanSelfAttention as WanSelfAttention2pt2
)

from ops import FastLayerNorm, FastRMSNorm, Int8Linear
from SLA import (
    SparseLinearAttention as SLA,
    SageSparseLinearAttention as SageSLA
)
from SLA.hif4 import hif4_qdq
from SLA.core import apply_weight_norm_2_to_4_sparsity
from rcm.utils.structured_sparsity import configure_sparsity_profile, get_sparsity_profile
from rcm.utils.selective_activation_checkpoint import SACConfig
from rcm.utils.attention import attention as dense_attention


def hif8_qdq(x: torch.Tensor, chunk_elements: int = 1 << 20) -> torch.Tensor:
    """HiFloat8 round-to-nearest simulation from the bundled HiFloat8 library."""
    flat = x.reshape(-1)
    output = torch.empty_like(flat)
    for start in range(0, flat.numel(), chunk_elements):
        stop = min(start + chunk_elements, flat.numel())
        x_float = flat[start:stop].float()
        magnitude = x_float.abs()
        exponent = torch.floor(torch.log2(magnitude + 2.0 ** -45))
        abs_exponent = exponent.abs()
        mantissa_bits = torch.zeros_like(abs_exponent)
        mantissa_bits.masked_fill_(abs_exponent <= 15, 1.0)
        mantissa_bits.masked_fill_(abs_exponent <= 7, 2.0)
        mantissa_bits.masked_fill_(abs_exponent <= 3, 3.0)
        quantized = torch.floor(magnitude * torch.pow(2.0, -exponent + mantissa_bits) + 0.5)
        quantized.mul_(torch.pow(2.0, exponent - mantissa_bits)).mul_(x_float.sign())
        output[start:stop] = quantized.to(x.dtype)
    return output.reshape_as(x)


class HiF8Linear(torch.nn.Linear):
    """Dense Linear with HiFloat8 W8A8 QDQ and BF16/FP32 accumulation."""

    weight_chunk_rows = 256

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

    @classmethod
    def from_linear(cls, module: torch.nn.Linear):
        result = cls(module.in_features, module.out_features, module.bias is not None, device=module.weight.device, dtype=module.weight.dtype)
        result.weight = module.weight
        result.bias = module.bias
        return result

    def forward(self, x):
        x_q = hif8_qdq(x)
        output = torch.empty(
            *x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype
        )
        for start in range(0, self.out_features, self.weight_chunk_rows):
            stop = min(start + self.weight_chunk_rows, self.out_features)
            weight_q = hif8_qdq(self.weight[start:stop])
            bias = None if self.bias is None else self.bias[start:stop]
            output[..., start:stop] = F.linear(x_q, weight_q, bias)
        return output


class HiF4Linear(torch.nn.Linear):
    """Dense Linear with numerical HiFloat4 W4A4 QDQ."""

    def forward(self, x):
        return F.linear(hif4_qdq(x, -1), hif4_qdq(self.weight, -1), self.bias)

    @classmethod
    def from_linear(cls, module: torch.nn.Linear):
        result = cls(module.in_features, module.out_features, module.bias is not None,
                     device=module.weight.device, dtype=module.weight.dtype)
        result.weight = module.weight
        result.bias = module.bias
        return result


class FFNHiF8SparseLinear(HiF8Linear):
    """HiFloat8 W8A8 FFN Linear with weight-normal activation 2:4."""
    def forward(self, x):
        x_q = apply_weight_norm_2_to_4_sparsity(
            hif8_qdq(x), self.weight.transpose(0, 1)
        )
        output = torch.empty(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        for start in range(0, self.out_features, self.weight_chunk_rows):
            stop = min(start + self.weight_chunk_rows, self.out_features)
            bias = None if self.bias is None else self.bias[start:stop]
            output[..., start:stop] = F.linear(x_q, hif8_qdq(self.weight[start:stop]), bias)
        return output

    @classmethod
    def from_linear(cls, module):
        result = cls(module.in_features, module.out_features, module.bias is not None,
                     device=module.weight.device, dtype=module.weight.dtype)
        result.weight = module.weight
        result.bias = module.bias
        return result


def replace_attention(
    model: torch.nn.Module,
    attention_type: str,
    sla_topk: float,
    linear_q_2to4: bool = False,
    linear_kv_2to4_operand: str = "none",
    linear_qkv_2to4_operand: str = "none",
    sla_q_2to4: bool = False,
    sla_q_4to8_pairwise: bool = False,
    sla_k_2to4: bool = False,
    sla_k_4to8_pairwise: bool = False,
    sla_q_2to4_share2: bool = False,
    sla_k_2to4_share2: bool = False,
    branch_aware_k_sparsity: str = "none",
    rubin_triple_2to4: bool = False,
    rubin_sparse_engine: str = "fused",
    rubin_validate_fused: bool = False,
    hif4_sparse_upgrade: bool = False,
    hif4_only_scope: str = "none",
    sparsity_profile: bool = False,
    norm_compensate: bool = True,
    sla_q_2to4_weight_norm: bool = False,
    sla_k_2to4_weight_norm: bool = False,
    dense_fallback_layers: tuple[int, ...] = (),
    sla_k_weight_norm_rpq: bool = False,
    sla_q_weight_norm_rpq: bool = False,
) -> torch.nn.Module:
    assert attention_type in ["sla", "sagesla"], "Invalid attention type."
    if hif4_sparse_upgrade and hif4_only_scope != "none":
        raise ValueError("HiF4 sparse-upgrade and HiF4-only are mutually exclusive")
    if attention_type == "sagesla" and (hif4_sparse_upgrade or hif4_only_scope != "none"):
        raise ValueError("HiF4 attention QDQ is currently supported only by --attention_type sla")
    if attention_type == "sagesla" and (sla_q_2to4_weight_norm or sla_k_2to4_weight_norm):
        raise ValueError("weight-normal Q/K 2:4 is currently supported only by --attention_type sla")
    if sum((sla_q_2to4, sla_q_4to8_pairwise, sla_q_2to4_share2)) > 1:
        raise ValueError("Q activation sparsity modes are mutually exclusive")
    if sum((sla_k_2to4, sla_k_4to8_pairwise, sla_k_2to4_share2)) > 1:
        raise ValueError("K activation sparsity modes are mutually exclusive")
    if sla_k_2to4_weight_norm and any((sla_k_2to4, sla_k_4to8_pairwise)):
        raise ValueError("K activation sparsity modes are mutually exclusive")
    if branch_aware_k_sparsity != "none" and any((sla_k_2to4, sla_k_4to8_pairwise, sla_k_2to4_share2)):
        raise ValueError("--branch_aware_k_sparsity conflicts with the SLA K sparsity flags")
    q_mode = (
        "2to4" if sla_q_2to4 else
        "4to8_pairwise" if sla_q_4to8_pairwise else
        "2to4_share2" if sla_q_2to4_share2 else "none"
    )
    k_mode = branch_aware_k_sparsity if branch_aware_k_sparsity != "none" else (
        "2to4" if sla_k_2to4 else
        "4to8_pairwise" if sla_k_4to8_pairwise else
        "2to4_share2" if sla_k_2to4_share2 else "none"
    )
    has_structured_mode = any((
        linear_q_2to4, linear_kv_2to4_operand != "none",
        linear_qkv_2to4_operand != "none", q_mode != "none",
        k_mode != "none", rubin_triple_2to4,
        sla_q_2to4_weight_norm,
        sla_k_2to4_weight_norm,
    ))
    if hif4_sparse_upgrade and not has_structured_mode:
        raise ValueError("--hif4_sparse_upgrade requires a structured-sparsity mode")
    if hif4_only_scope != "none" and has_structured_mode:
        raise ValueError("--hif4_only_scope cannot be combined with structured sparsity")
    if hif4_only_scope == "rubin" and rubin_sparse_engine != "fused":
        raise ValueError("Rubin-scope HiF4-only requires the fused engine")
    if rubin_triple_2to4 and (q_mode != "none" or k_mode != "none"):
        raise ValueError("--rubin_triple_2to4 conflicts with separate Q/K sparsity modes")
    if rubin_triple_2to4 and attention_type != "sla":
        raise ValueError("--rubin_triple_2to4 reference simulation currently requires --attention_type sla")
    
    for module_name, module in model.named_modules():
        if type(module) is WanSelfAttention2pt1 or type(module) is WanSelfAttention2pt2:
            block_index = None
            parts = module_name.split(".")
            if "blocks" in parts:
                try:
                    block_index = int(parts[parts.index("blocks") + 1])
                except (ValueError, IndexError):
                    pass
            if block_index in dense_fallback_layers:
                module.attn_op.local_attn = dense_attention
                continue
            if attention_type == "sla":
                local_attn = SLA(
                    head_dim=module.dim // module.num_heads,
                    topk=sla_topk,
                    BLKQ=128,
                    BLKK=64,
                    linear_q_2to4=linear_q_2to4,
                    linear_kv_2to4_operand=linear_kv_2to4_operand,
                    linear_qkv_2to4_operand=linear_qkv_2to4_operand,
                    branch_aware_q_sparsity=q_mode,
                    branch_aware_k_sparsity=k_mode,
                    rubin_triple_2to4=rubin_triple_2to4,
                    rubin_sparse_engine=rubin_sparse_engine,
                    rubin_validate_fused=rubin_validate_fused,
                    hif4_sparse_upgrade=hif4_sparse_upgrade,
                    hif4_only_scope=hif4_only_scope,
                    norm_compensate=norm_compensate,
                    q_weight_norm_2to4=sla_q_2to4_weight_norm,
                    k_weight_norm_2to4=sla_k_2to4_weight_norm,
                    k_weight_norm_rpq=sla_k_weight_norm_rpq,
                    q_weight_norm_rpq=sla_q_weight_norm_rpq,
                )
            elif attention_type == "sagesla":
                local_attn = SageSLA(
                    head_dim=module.dim // module.num_heads,
                    topk=sla_topk,
                    linear_q_2to4=linear_q_2to4,
                    linear_kv_2to4_operand=linear_kv_2to4_operand,
                    linear_qkv_2to4_operand=linear_qkv_2to4_operand,
                    branch_aware_q_sparsity=q_mode,
                    branch_aware_k_sparsity=k_mode,
                    norm_compensate=norm_compensate,
                )
            if sparsity_profile and not rubin_triple_2to4:
                configure_sparsity_profile(local_attn)
            if sparsity_profile:
                local_attn._sparsity_profile_layer_name = module_name
            module.attn_op.local_attn = local_attn
    return model


def collect_sparsity_profiles(model: torch.nn.Module, model_label: str) -> dict:
    """Collect finalized per-layer sparsity diagnostics from a model."""

    layers = {}
    for module_name, module in model.named_modules():
        if hasattr(module, "_rubin_triple_audit"):
            layer_name = getattr(module, "_sparsity_profile_layer_name", module_name)
            layers[layer_name] = {"rubin_triple_2to4": dict(module._rubin_triple_audit)}
            if getattr(module, "_hif4_operand_audit", None):
                layers[layer_name]["hif4_operands"] = dict(module._hif4_operand_audit)
        elif getattr(module, "_hif4_operand_audit", None):
            layer_name = getattr(module, "_sparsity_profile_layer_name", module_name)
            layers[layer_name] = {"hif4_operands": dict(module._hif4_operand_audit)}
        elif hasattr(module, "_sparsity_profile_stats"):
            layer_name = getattr(module, "_sparsity_profile_layer_name", module_name)
            layers[layer_name] = get_sparsity_profile(module)
    return {"model": model_label, "layers": layers}


def replace_linear_norm(
    model: torch.nn.Module,
    replace_linear: bool = False,
    replace_norm: bool = False,
    quantize: bool = True,
    skip_layer: str = "proj_l"
) -> torch.nn.Module:
    replacements = {}
    for name, module in model.blocks.named_modules():
        if isinstance(module, torch.nn.Linear) and replace_linear:
            if skip_layer not in name:
                replacements[name] = Int8Linear.from_linear(module, quantize)
        
        if (isinstance(module, WanRMSNorm2pt1) or isinstance(module, WanRMSNorm2pt2)) and replace_norm:
            replacements[name] = FastRMSNorm.from_rmsnorm(module)
        
        if (isinstance(module, WanLayerNorm2pt1) or isinstance(module, WanLayerNorm2pt2)) and replace_norm:
            replacements[name] = FastLayerNorm.from_layernorm(module)

    for name, new_module in replacements.items():
        parent_module = model.blocks
        name_parts = name.split(".")
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], new_module)
    return model


def replace_linear_hif8(model: torch.nn.Module, skip_layer: str = "proj_l") -> torch.nn.Module:
    replacements = {}
    for name, module in model.blocks.named_modules():
        if isinstance(module, torch.nn.Linear) and not isinstance(module, HiF8Linear) and skip_layer not in name:
            replacements[name] = HiF8Linear.from_linear(module)
    for name, replacement in replacements.items():
        parent = model.blocks
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)
    return model


def replace_sla_k_linear_hif4(model: torch.nn.Module) -> torch.nn.Module:
    """Replace only Wan attention K projections with numerical HiF4 W4A4."""
    replacements = {}
    for name, module in model.blocks.named_modules():
        if name.endswith("self_attn.k") and isinstance(module, torch.nn.Linear):
            replacements[name] = HiF4Linear.from_linear(module)
    for name, replacement in replacements.items():
        parent = model.blocks
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)
    return model


def _replace_ffn_linears(model, cls):
    replacements = {}
    for name, module in model.blocks.named_modules():
        if (".ffn.0" in name or ".ffn.2" in name) and isinstance(module, torch.nn.Linear):
            replacements[name] = cls.from_linear(module)
    for name, replacement in replacements.items():
        parent = model.blocks
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)
    return model


def replace_ffn_hif4(model):
    return _replace_ffn_linears(model, HiF4Linear)


def replace_ffn_hif8_sparse(model):
    return _replace_ffn_linears(model, FFNHiF8SparseLinear)


tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

def select_model(model_name: str, sac_mode: str = "mm_only") -> torch.nn.Module:
    sac_config = SACConfig(mode=sac_mode)
    if model_name == "Wan2.1-1.3B":
        return WanModel2pt1(
            dim=1536,
            eps=1e-06,
            ffn_dim=8960,
            freq_dim=256,
            in_dim=16,
            model_type="t2v",
            num_heads=12,
            num_layers=30,
            out_dim=16,
            text_len=512,
            sac_config=sac_config,
        )
    elif model_name == "Wan2.1-14B":
        return WanModel2pt1(
            dim=5120,
            eps=1e-06,
            ffn_dim=13824,
            freq_dim=256,
            in_dim=16,
            model_type="t2v",
            num_heads=40,
            num_layers=40,
            out_dim=16,
            text_len=512,
            sac_config=sac_config,
        )
    elif model_name == "Wan2.2-A14B":
        return WanModel2pt2(
            dim=5120,
            eps=1e-06,
            ffn_dim=13824,
            freq_dim=256,
            in_dim=36,
            model_type="i2v",
            num_heads=40,
            num_layers=40,
            out_dim=16,
            text_len=512,
            sac_config=sac_config,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_model(dit_path: str, args: argparse.Namespace, target_device: str | torch.device = "cuda") -> torch.nn.Module:
    with torch.device("meta"):
        net = select_model(args.model, sac_mode=getattr(args, "sac_mode", "mm_only"))

    state_dict = load_state_dict(dit_path)
    fallback_layers = tuple(getattr(args, "dense_fallback_layers", ()))
    if fallback_layers:
        state_dict = {
            key: value for key, value in state_dict.items()
            if not any(key.startswith(f"blocks.{idx}.") and ".attn_op.local_attn." in key for idx in fallback_layers)
        }
    if args.attention_type == "original":
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if ".attn_op.local_attn." not in key
        }
    if args.attention_type in ['sla', 'sagesla']:
        net = replace_attention(
            net,
            attention_type=args.attention_type,
            sla_topk=args.sla_topk,
            linear_q_2to4=getattr(args, "linear_q_2to4", False),
            linear_kv_2to4_operand=getattr(args, "linear_kv_2to4_operand", "none"),
            linear_qkv_2to4_operand=getattr(args, "linear_qkv_2to4_operand", "none"),
            sla_q_2to4=getattr(args, "sla_q_2to4", False),
            sla_q_4to8_pairwise=getattr(args, "sla_q_4to8_pairwise", False),
            sla_k_2to4=getattr(args, "sla_k_2to4", False),
            sla_k_4to8_pairwise=getattr(args, "sla_k_4to8_pairwise", False),
            sla_q_2to4_share2=getattr(args, "sla_q_2to4_share2", False),
            sla_k_2to4_share2=getattr(args, "sla_k_2to4_share2", False),
            branch_aware_k_sparsity=getattr(args, "branch_aware_k_sparsity", "none"),
            rubin_triple_2to4=getattr(args, "rubin_triple_2to4", False),
            rubin_sparse_engine=getattr(args, "rubin_sparse_engine", "fused"),
            rubin_validate_fused=getattr(args, "rubin_validate_fused", False),
            hif4_sparse_upgrade=getattr(args, "hif4_sparse_upgrade", False),
            hif4_only_scope=getattr(args, "hif4_only_scope", "none"),
            sparsity_profile=getattr(args, "sparsity_profile_path", None) is not None,
            norm_compensate=getattr(args, "norm_compensate", True),
            sla_q_2to4_weight_norm=getattr(args, "sla_q_2to4_weight_norm", False),
            sla_k_2to4_weight_norm=getattr(args, "sla_k_2to4_weight_norm", False),
            dense_fallback_layers=tuple(getattr(args, "dense_fallback_layers", ())),
            sla_k_weight_norm_rpq=getattr(args, "sla_k_weight_norm_rpq", False),
            sla_q_weight_norm_rpq=getattr(args, "sla_q_weight_norm_rpq", False),
        )
    if getattr(args, "hif8_w8a8", False) and args.quant_linear:
        raise ValueError("--hif8_w8a8 conflicts with --quant_linear")
    replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm, quantize=False)
    if getattr(args, "hif8_w8a8", False):
        replace_linear_hif8(net)
    if getattr(args, "sla_k_hif4_w4a4", False):
        if not getattr(args, "hif8_w8a8", False):
            raise ValueError("--sla_k_hif4_w4a4 requires --hif8_w8a8 for Q/V/O projections")
        replace_sla_k_linear_hif4(net)
    if getattr(args, "ffn_hif4_w4a4", False):
        replace_ffn_hif4(net)
    if getattr(args, "ffn_hif8_w8a8_2to4_weight_norm", False):
        replace_ffn_hif8_sparse(net)
    net.load_state_dict(state_dict, assign=True)
    net = net.to(target_device).eval()
    del state_dict
    return net


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion replace attention module & quantize model")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-A14B"], default="Wan2.1-1.3B", help="Model to use")
    parser.add_argument("--input_path", type=str, default="", help="Input path to the DiT model checkpoint for Wan model after rCM-SLA finetuning")
    parser.add_argument("--output_path", type=str, default="", help="Custom path to save the modified model checkpoint")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="original", help="Type of attention mechanism to use")
    parser.add_argument("--sla_topk", type=float, default=0.2, help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--linear_q_2to4", action="store_true", help="Simulate 2:4 activation sparsity on Q in the linear-attention branch")
    parser.add_argument("--sla_q_2to4", action="store_true", help="Simulate 2:4 activation sparsity on SLA/SageSLA queries")
    parser.add_argument("--sla_q_4to8_pairwise", action="store_true", help="Simulate pairwise 4:8 activation sparsity on SLA/SageSLA queries")
    parser.add_argument("--sla_k_2to4", action="store_true", help="Simulate 2:4 activation sparsity on SLA/SageSLA keys")
    parser.add_argument("--sla_k_4to8_pairwise", action="store_true", help="Simulate pairwise 4:8 activation sparsity on SLA/SageSLA keys")
    parser.add_argument("--sla_q_2to4_share2", action="store_true", help="Simulate Q 2:4 with one L1-selected mask shared by two tokens")
    parser.add_argument("--sla_k_2to4_share2", action="store_true", help="Simulate K 2:4 with one L1-selected mask shared by two tokens")
    parser.add_argument("--branch_aware_k_sparsity", choices=["none", "2to4", "4to8_pairwise", "2to4_share2"], default="none")
    parser.add_argument("--rubin_triple_2to4", action="store_true")
    parser.add_argument("--rubin_sparse_engine", choices=["fused"], default="fused")
    parser.add_argument("--hif4_sparse_upgrade", action="store_true")
    parser.add_argument("--hif4_only_scope", choices=["none", "q_path", "k_path", "linear", "rubin"], default="none")
    parser.add_argument("--sla_k_hif4_w4a4", action="store_true", help="Use HiFloat4 W4A4 only for SLA K projections (with HiF8 Q/V/O)")
    parser.add_argument("--ffn_hif4_w4a4", action="store_true")
    parser.add_argument("--ffn_hif8_w8a8_2to4_weight_norm", action="store_true")
    parser.add_argument("--rubin_validate_fused", action="store_true")
    parser.add_argument("--no_norm_compensate", dest="norm_compensate", action="store_false",
                        help="Disable RT-Lynx norm compensation for 2:4 activation sparsity")
    parser.add_argument("--quant_linear", action="store_true", help="Whether to replace Linear layers with quantized versions")
    parser.add_argument("--default_norm", action="store_true", help="Whether to replace LayerNorm/RMSNorm layers with faster versions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    with torch.device("meta"):
        net = select_model(args.model)

    state_dict = load_state_dict(args.input_path)["state_dict"]

    # drop net. prefix
    prefix_to_load = "net."
    state_dict_dit_compatible = dict()
    for k, v in state_dict.items():
        new_key = k[len(prefix_to_load) :] if k.startswith(prefix_to_load) else k
        # reshape patch embedding if needed
        if k.endswith("patch_embedding.weight"):
            v = v.reshape(net.patch_embedding.weight.shape)
        if k.endswith("patch_embedding.bias"):
            v = v.reshape(net.patch_embedding.bias.shape)
        state_dict_dit_compatible[new_key] = v

    if args.attention_type in ['sla', 'sagesla']:
        net = replace_attention(
            net,
            attention_type=args.attention_type,
            sla_topk=args.sla_topk,
            linear_q_2to4=args.linear_q_2to4,
            sla_q_2to4=args.sla_q_2to4,
            sla_q_4to8_pairwise=args.sla_q_4to8_pairwise,
            sla_k_2to4=args.sla_k_2to4,
            sla_k_4to8_pairwise=args.sla_k_4to8_pairwise,
            sla_q_2to4_share2=args.sla_q_2to4_share2,
            sla_k_2to4_share2=args.sla_k_2to4_share2,
            branch_aware_k_sparsity=args.branch_aware_k_sparsity,
            rubin_triple_2to4=args.rubin_triple_2to4,
            rubin_sparse_engine=args.rubin_sparse_engine,
            rubin_validate_fused=args.rubin_validate_fused,
            hif4_sparse_upgrade=args.hif4_sparse_upgrade,
            hif4_only_scope=args.hif4_only_scope,
            norm_compensate=args.norm_compensate,
        )
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    net = net.to(tensor_kwargs["device"]).eval()
    del state_dict, state_dict_dit_compatible

    net = replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm)
    torch.save(net.state_dict(), args.output_path)
