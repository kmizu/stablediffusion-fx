"""Reference attention with separate projections, preserving spatial conditioning hooks.

Equivalent to native SDPA attention over [hidden_states, reference * weight], but projecting
the two spatial grids separately lets ControlLLLite condition both grids before concatenation.
"""


def attend(attn, hidden_states, reference, weight, *, attention_mask=None, temb=None):
    import torch
    import torch.nn.functional as F

    residual = hidden_states
    context = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    shape = hidden_states.shape
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        context = context.flatten(2).transpose(1, 2)
        reference = reference.flatten(2).transpose(1, 2)
    batch = hidden_states.shape[0]
    reference = reference * weight
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    if attn.norm_cross:
        # GroupNorm can reduce across tokens; normalize the combined context before spatially
        # separate projections to preserve the native concatenated-context statistics.
        length = context.shape[1]
        combined = attn.norm_encoder_hidden_states(torch.cat([context, reference], dim=1))
        context, reference = combined[:, :length], combined[:, length:]
    query = attn.to_q(hidden_states)
    key = torch.cat([attn.to_k(context), attn.to_k(reference)], dim=1)
    value = torch.cat([attn.to_v(context), attn.to_v(reference)], dim=1)
    head_dim = key.shape[-1] // attn.heads
    query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    if attention_mask is not None:
        attention_mask = F.pad(attention_mask, (0, reference.shape[1]),
                               value=True if attention_mask.dtype == torch.bool else 0.0)
        attention_mask = attn.prepare_attention_mask(attention_mask, key.shape[2], batch)
        attention_mask = attention_mask.view(batch, attn.heads, -1, attention_mask.shape[-1])
    result = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                            dropout_p=0.0, is_causal=False)
    result = result.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(query.dtype)
    result = attn.to_out[1](attn.to_out[0](result))
    if len(shape) == 4:
        result = result.transpose(1, 2).reshape(shape)
    if attn.residual_connection:
        result = result + residual
    return result / attn.rescale_output_factor
