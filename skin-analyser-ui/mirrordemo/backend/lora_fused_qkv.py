import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# ============================================================
# FUSED QKV LoRA
# ============================================================

class FusedQKVLoRA(nn.Module):
    """
    LoRA adapter for a fused QKV projection.

    Original weight:
        [ Q ]
        [ K ]
        [ V ]

    Each section has shape:
        [embed_dim, embed_dim]

    Only the LoRA parameters are trainable.
    """

    def __init__(
        self,
        base_weight,
        rank=8,
        alpha=16,
    ):
        super().__init__()

        self.embed_dim = base_weight.shape[1]
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen original QKV weight
        self.register_buffer(
            "base_weight",
            base_weight.detach().clone()
        )

        # Q LoRA
        self.q_A = nn.Parameter(
            torch.empty(rank, self.embed_dim)
        )
        self.q_B = nn.Parameter(
            torch.zeros(self.embed_dim, rank)
        )

        # K LoRA
        self.k_A = nn.Parameter(
            torch.empty(rank, self.embed_dim)
        )
        self.k_B = nn.Parameter(
            torch.zeros(self.embed_dim, rank)
        )

        # V LoRA
        self.v_A = nn.Parameter(
            torch.empty(rank, self.embed_dim)
        )
        self.v_B = nn.Parameter(
            torch.zeros(self.embed_dim, rank)
        )

        self.reset_parameters()

    def reset_parameters(self):

        # Standard LoRA initialization:
        # A = random
        # B = zero
        #
        # Therefore initial LoRA update = zero.

        nn.init.kaiming_uniform_(
            self.q_A,
            a=5 ** 0.5
        )

        nn.init.kaiming_uniform_(
            self.k_A,
            a=5 ** 0.5
        )

        nn.init.kaiming_uniform_(
            self.v_A,
            a=5 ** 0.5
        )

        nn.init.zeros_(self.q_B)
        nn.init.zeros_(self.k_B)
        nn.init.zeros_(self.v_B)

    def _update(self, A, B):

        return (
            B @ A
        ) * self.scaling

    def get_weight(self):

        D = self.embed_dim

        # Split fused QKV
        q_base = self.base_weight[:D]

        k_base = self.base_weight[
            D:2 * D
        ]

        v_base = self.base_weight[
            2 * D:
        ]

        # Add LoRA updates
        q = q_base + self._update(
            self.q_A,
            self.q_B
        )

        k = k_base + self._update(
            self.k_A,
            self.k_B
        )

        v = v_base + self._update(
            self.v_A,
            self.v_B
        )

        # Reconstruct fused QKV
        return torch.cat(
            [q, k, v],
            dim=0
        )


# ============================================================
# LoRA MULTIHEAD ATTENTION
# ============================================================

class LoRAMultiheadAttention(nn.Module):
    """
    Wraps OpenCLIP MultiheadAttention.

    OpenCLIP uses:
        (batch, sequence, dimension)

    PyTorch functional attention expects:
        (sequence, batch, dimension)

    Therefore we transpose before and after the
    functional attention call.
    """

    def __init__(
        self,
        original_attn,
        rank=8,
        alpha=16,
    ):
        super().__init__()

        self.base_attn = original_attn

        # Freeze original attention parameters
        for parameter in self.base_attn.parameters():
            parameter.requires_grad = False

        # Add fused QKV LoRA
        self.lora = FusedQKVLoRA(
            base_weight=self.base_attn.in_proj_weight,
            rank=rank,
            alpha=alpha,
        )

    def forward(
        self,
        x=None,
        k_x=None,
        v_x=None,
        attn_mask=None,
        *args,
        query=None,
        key=None,
        value=None,
        key_padding_mask=None,
        need_weights=False,
        average_attn_weights=True,
        is_causal=False,
        **kwargs,
    ):
        if x is None:
            if len(args) > 0:
                x = args[0]
            elif query is not None:
                x = query

        qkv_weight = self.lora.get_weight()
        N, L, C = x.shape

        if k_x is None and v_x is None:
            q, k, v = F.linear(x, qkv_weight, self.base_attn.in_proj_bias).chunk(3, dim=-1)
        else:
            k_x = x if k_x is None else k_x
            v_x = x if v_x is None else v_x
            w_q, w_k, w_v = qkv_weight.chunk(3, dim=0)
            bias_q, bias_k, bias_v = (
                self.base_attn.in_proj_bias.chunk(3) if self.base_attn.in_proj_bias is not None
                else (None, None, None)
            )
            q = F.linear(x, w_q, bias_q)
            k = F.linear(k_x, w_k, bias_k)
            v = F.linear(v_x, w_v, bias_v)

        q = q.reshape(N, -1, self.base_attn.num_heads, self.base_attn.head_dim).transpose(1, 2)
        k = k.reshape(N, -1, self.base_attn.num_heads, self.base_attn.head_dim).transpose(1, 2)
        v = v.reshape(N, -1, self.base_attn.num_heads, self.base_attn.head_dim).transpose(1, 2)

        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask.reshape(N, self.base_attn.num_heads, L, L)

        if self.base_attn.logit_scale is not None:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
            logit_scale = torch.clamp(self.base_attn.logit_scale, max=self.base_attn.logit_scale_max).exp()
            attn = attn * logit_scale
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn.masked_fill_(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask.to(dtype=attn.dtype)
            attn = attn.softmax(dim=-1)
            attn = self.base_attn.attn_drop(attn)
            x_out = attn @ v
        else:
            q = self.base_attn.ln_q(q)
            k = self.base_attn.ln_k(k)
            if self.base_attn.use_sdpa:
                x_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.base_attn.attn_drop.p if self.training else 0.,
                    scale=self.base_attn.scale,
                )
            else:
                q = q * self.base_attn.scale
                attn = q @ k.transpose(-1, -2)
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn.masked_fill_(~attn_mask, float("-inf"))
                    else:
                        attn = attn + attn_mask.to(dtype=attn.dtype)
                attn = attn.softmax(dim=-1)
                attn = self.base_attn.attn_drop(attn)
                x_out = attn @ v

        if self.base_attn.head_scale is not None:
            x_out = x_out * self.base_attn.head_scale
        x_out = x_out.transpose(1, 2).reshape(N, -1, C)
        x_out = self.base_attn.ln_inner(x_out)
        x_out = self.base_attn.out_proj(x_out)
        x_out = self.base_attn.out_drop(x_out)
        return x_out


# ============================================================
# APPLY LoRA TO ALL VISION TRANSFORMER BLOCKS
# ============================================================

def apply_fused_qkv_lora(
    model,
    rank=8,
    alpha=16,
):
    """
    Replaces the attention module in every ViT block
    with the LoRA-enabled attention wrapper.
    """

    blocks = model.visual.transformer.resblocks

    count = 0

    for block in blocks:

        original_attn = block.attn

        block.attn = LoRAMultiheadAttention(
            original_attn=original_attn,
            rank=rank,
            alpha=alpha,
        )

        count += 1

    return count


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    total = 0
    trainable = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:
            trainable += parameter.numel()

    return trainable, total


# ============================================================
# COMPLETE LoRA TEST
# ============================================================

def test_lora():

    print("=" * 60)
    print("Testing REAL Fused-QKV LoRA")
    print("=" * 60)

    device = "cpu"

    # --------------------------------------------------------
    # Load OpenCLIP QuickGELU
    # --------------------------------------------------------

    model, _, _ = (
        open_clip.create_model_and_transforms(
            "ViT-L-14-quickgelu",
            pretrained="openai",
        )
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Verify batch_first
    # --------------------------------------------------------

    original_attn = (
        model.visual
        .transformer
        .resblocks[0]
        .attn
    )

    print(
        "\nOriginal batch_first:",
        original_attn.batch_first
    )

    assert original_attn.batch_first is True

    print(
        "batch_first check: PASSED"
    )

    # --------------------------------------------------------
    # Freeze entire model
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Create dummy image
    # --------------------------------------------------------

    image = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    # --------------------------------------------------------
    # BASELINE OUTPUT
    # --------------------------------------------------------

    print(
        "\nRunning original model..."
    )

    model.eval()

    with torch.no_grad():

        baseline_features = (
            model.encode_image(image)
            .detach()
            .clone()
        )

    print(
        "Baseline embedding shape:",
        tuple(baseline_features.shape)
    )

    # --------------------------------------------------------
    # APPLY LoRA
    # --------------------------------------------------------

    print(
        "\nApplying LoRA..."
    )

    num_blocks = apply_fused_qkv_lora(
        model,
        rank=8,
        alpha=16,
    )

    print(
        f"LoRA blocks: {num_blocks}"
    )

    assert num_blocks == 24

    # --------------------------------------------------------
    # ZERO-INIT EQUIVALENCE TEST
    # --------------------------------------------------------

    print(
        "\nRunning zero-init equivalence test..."
    )

    model.eval()

    with torch.no_grad():

        lora_features = (
            model.encode_image(image)
            .detach()
        )

    max_difference = (
        baseline_features - lora_features
    ).abs().max().item()

    mean_difference = (
        baseline_features - lora_features
    ).abs().mean().item()

    print(
        f"Maximum output difference: "
        f"{max_difference:.10f}"
    )

    print(
        f"Mean output difference: "
        f"{mean_difference:.10f}"
    )

    assert torch.allclose(
        baseline_features,
        lora_features,
        atol=1e-5,
        rtol=1e-5,
    ), (
        "LoRA initialization changed the "
        "baseline output!"
    )

    print(
        "Zero-init equivalence test: PASSED"
    )

    # --------------------------------------------------------
    # PARAMETER COUNT
    # --------------------------------------------------------

    trainable, total = count_parameters(
        model
    )

    print(
        f"\nTotal parameters:     {total:,}"
    )

    print(
        f"Trainable parameters: {trainable:,}"
    )

    expected_trainable = 1_179_648

    assert trainable == expected_trainable

    print(
        "Trainable parameter count: PASSED"
    )

    # --------------------------------------------------------
    # TRAINABLE PARAMETER NAMES
    # --------------------------------------------------------

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    print(
        "\nNumber of trainable tensors:",
        len(trainable_names)
    )

    print(
        "\nFirst trainable parameters:"
    )

    for name in trainable_names[:12]:

        print(
            f"  {name}"
        )

    # --------------------------------------------------------
    # REAL FORWARD PASS
    # --------------------------------------------------------

    print(
        "\nRunning real image forward pass..."
    )

    model.train()

    image_features = model.encode_image(
        image
    )

    print(
        "Image embedding shape:",
        tuple(image_features.shape)
    )

    # --------------------------------------------------------
    # REAL BACKWARD PASS
    # --------------------------------------------------------

    print(
        "\nRunning backward pass..."
    )

    loss = image_features.float().mean()

    loss.backward()

    # --------------------------------------------------------
    # INSPECT FIRST LoRA BLOCK
    # --------------------------------------------------------

    first_block = (
        model.visual
        .transformer
        .resblocks[0]
        .attn
    )

    lora = first_block.lora

    # --------------------------------------------------------
    # GRADIENT CHECKS
    # --------------------------------------------------------

    print(
        "\nGradient checks:"
    )

    gradients = {
        "q_A": lora.q_A.grad,
        "q_B": lora.q_B.grad,
        "k_A": lora.k_A.grad,
        "k_B": lora.k_B.grad,
        "v_A": lora.v_A.grad,
        "v_B": lora.v_B.grad,
    }

    for name, gradient in gradients.items():

        print(
            f"{name}: {gradient is not None}"
        )

    # Every LoRA matrix must receive gradients.
    for name, gradient in gradients.items():

        assert gradient is not None, (
            f"No gradient reached {name}"
        )

        assert torch.isfinite(
            gradient
        ).all(), (
            f"Non-finite gradient in {name}"
        )

    print(
        "\nAll LoRA gradients: PASSED"
    )

    # --------------------------------------------------------
    # ORIGINAL QKV MUST REMAIN FROZEN
    # --------------------------------------------------------

    assert (
        first_block.base_attn
        .in_proj_weight
        .requires_grad
        is False
    )

    print(
        "Original QKV remains frozen: PASSED"
    )

    # --------------------------------------------------------
    # OUTPUT PROJECTION MUST REMAIN FROZEN
    # --------------------------------------------------------

    assert (
        first_block.base_attn
        .out_proj.weight
        .requires_grad
        is False
    )

    print(
        "Output projection remains frozen: PASSED"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "ALL LoRA TESTS PASSED"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_lora()