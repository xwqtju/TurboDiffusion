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
from rcm.utils.structured_sparsity import (
    enable_q_activation_2_to_4,
    enable_q_activation_4_to_8_pairwise,
    enable_k_activation_2_to_4,
    enable_k_activation_4_to_8_pairwise,
    enable_q_activation_2_to_4_share_index_2,
    enable_k_activation_2_to_4_share_index_2,
    configure_sparsity_profile,
    get_sparsity_profile,
)
from rcm.utils.selective_activation_checkpoint import SACConfig


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
    sparsity_profile: bool = False,
) -> torch.nn.Module:
    assert attention_type in ["sla", "sagesla"], "Invalid attention type."
    if sum((sla_q_2to4, sla_q_4to8_pairwise, sla_q_2to4_share2)) > 1:
        raise ValueError("Q activation sparsity modes are mutually exclusive")
    if sum((sla_k_2to4, sla_k_4to8_pairwise, sla_k_2to4_share2)) > 1:
        raise ValueError("K activation sparsity modes are mutually exclusive")
    
    for module_name, module in model.named_modules():
        if type(module) is WanSelfAttention2pt1 or type(module) is WanSelfAttention2pt2:
            if attention_type == "sla":
                local_attn = SLA(
                    head_dim=module.dim // module.num_heads,
                    topk=sla_topk,
                    BLKQ=128,
                    BLKK=64,
                    linear_q_2to4=linear_q_2to4,
                    linear_kv_2to4_operand=linear_kv_2to4_operand,
                    linear_qkv_2to4_operand=linear_qkv_2to4_operand,
                )
            elif attention_type == "sagesla":
                local_attn = SageSLA(
                    head_dim=module.dim // module.num_heads,
                    topk=sla_topk,
                    linear_q_2to4=linear_q_2to4,
                    linear_kv_2to4_operand=linear_kv_2to4_operand,
                    linear_qkv_2to4_operand=linear_qkv_2to4_operand,
                )
            if sparsity_profile:
                configure_sparsity_profile(local_attn)
                local_attn._sparsity_profile_layer_name = module_name
            if sla_q_2to4:
                enable_q_activation_2_to_4(local_attn)
            if sla_q_4to8_pairwise:
                enable_q_activation_4_to_8_pairwise(local_attn)
            if sla_k_2to4:
                enable_k_activation_2_to_4(local_attn)
            if sla_k_4to8_pairwise:
                enable_k_activation_4_to_8_pairwise(local_attn)
            if sla_q_2to4_share2:
                enable_q_activation_2_to_4_share_index_2(local_attn)
            if sla_k_2to4_share2:
                enable_k_activation_2_to_4_share_index_2(local_attn)
            module.attn_op.local_attn = local_attn
    return model


def collect_sparsity_profiles(model: torch.nn.Module, model_label: str) -> dict:
    """Collect finalized per-layer sparsity diagnostics from a model."""

    layers = {}
    for module_name, module in model.named_modules():
        if hasattr(module, "_sparsity_profile_stats"):
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
            sparsity_profile=getattr(args, "sparsity_profile_path", None) is not None,
        )
    replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm, quantize=False)
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
        )
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    net = net.to(tensor_kwargs["device"]).eval()
    del state_dict, state_dict_dit_compatible

    net = replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm)
    torch.save(net.state_dict(), args.output_path)
