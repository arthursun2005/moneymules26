import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial
import struct
import numpy as np


# ============================================================================
# Layer Utilities (Pure JAX, no Flax)
# ============================================================================

def init_conv(key, in_channels, out_channels, kernel_size, name="conv"):
    """Initialize convolutional layer parameters using Kaiming init."""
    k1, k2 = random.split(key)
    fan_in = in_channels * kernel_size * kernel_size
    std = jnp.sqrt(2.0 / fan_in)
    w = random.normal(k1, (out_channels, in_channels, kernel_size, kernel_size)) * std
    b = jnp.zeros((out_channels,))
    return {'w': w, 'b': b}


def init_conv_transpose(key, in_channels, out_channels, kernel_size, name="conv_t"):
    """Initialize transposed convolutional layer parameters."""
    k1, k2 = random.split(key)
    fan_in = in_channels * kernel_size * kernel_size
    std = jnp.sqrt(2.0 / fan_in)
    # For conv_transpose: kernel shape is (out_channels, in_channels, kH, kW)
    # but JAX conv_transpose uses (in_channels, out_channels, kH, kW)
    w = random.normal(k1, (in_channels, out_channels, kernel_size, kernel_size)) * std
    b = jnp.zeros((out_channels,))
    return {'w': w, 'b': b}


def init_dense(key, in_features, out_features):
    """Initialize dense layer parameters."""
    k1, k2 = random.split(key)
    std = jnp.sqrt(2.0 / in_features)
    w = random.normal(k1, (in_features, out_features)) * std
    b = jnp.zeros((out_features,))
    return {'w': w, 'b': b}


def conv2d(params, x, stride=1, padding='SAME'):
    """Apply 2D convolution. x: (N, C, H, W)"""
    return jax.lax.conv_general_dilated(
        x, params['w'],
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    ) + params['b'][None, :, None, None]


def conv_transpose2d(params, x, stride=2, padding='SAME'):
    """Apply transposed 2D convolution. x: (N, C, H, W)"""
    return jax.lax.conv_transpose(
        x, params['w'],
        strides=(stride, stride),
        padding=padding,
        dimension_numbers=('NCHW', 'IOHW', 'NCHW')
    ) + params['b'][None, :, None, None]


def dense(params, x):
    """Apply dense layer."""
    return x @ params['w'] + params['b']


def gdn(x, gamma, beta, inverse=False):
    """
    Generalized Divisive Normalization (GDN) - used in learned image compression.
    x: (N, C, H, W)
    gamma: (C, C)
    beta: (C,)
    """
    C = x.shape[1]
    # Compute normalization: sqrt(beta + sum_j(gamma_ij * x_j^2))
    x_sq = x ** 2  # (N, C, H, W)
    # Reshape for matrix multiply across channels
    N, _, H, W = x.shape
    x_sq_flat = x_sq.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
    norm = x_sq_flat @ (gamma ** 2) + (beta ** 2)[None, :]  # (N*H*W, C)
    norm = jnp.sqrt(norm + 1e-6)
    norm = norm.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # (N, C, H, W)

    if inverse:
        return x * norm
    else:
        return x / norm


def init_gdn(key, channels):
    """Initialize GDN parameters."""
    k1, k2 = random.split(key)
    gamma = jnp.eye(channels) * 0.1 + random.normal(k1, (channels, channels)) * 0.01
    beta = jnp.ones(channels) * 0.1
    return {'gamma': gamma, 'beta': beta}


# ============================================================================
# Entropy Model: Fully Factorized Prior (like in Ballé et al.)
# ============================================================================

def init_entropy_model(key, channels, num_filters=3):
    """
    Initialize a simple factorized entropy model.
    Models the marginal distribution of each latent channel with a flexible density.
    Uses a stack of small MLPs per channel to model the CDF.
    """
    keys = random.split(key, 4)
    params = {
        'h1': init_dense(keys[0], 1, 64),
        'h2': init_dense(keys[1], 64, 64),
        'h3': init_dense(keys[2], 64, 64),
        'h4': init_dense(keys[3], 64, 1),
        'channel_means': jnp.zeros(channels),
        'channel_log_scales': jnp.zeros(channels),
    }
    return params


def entropy_model_log_prob(params, y):
    """
    Compute log probability of quantized latents using a Gaussian mixture
    approximated by learned per-channel means and scales.
    y: (N, C, H, W) - quantized latents
    Returns: log_prob per element
    """
    means = params['channel_means'][None, :, None, None]
    log_scales = params['channel_log_scales'][None, :, None, None]
    scales = jnp.exp(log_scales) + 1e-6

    # Model each quantized value as the integral of a logistic distribution
    # over [y - 0.5, y + 0.5]
    centered = y - means
    upper = (centered + 0.5) / scales
    lower = (centered - 0.5) / scales

    # CDF of logistic distribution
    cdf_upper = jax.nn.sigmoid(upper)
    cdf_lower = jax.nn.sigmoid(lower)

    # Probability mass in the bin
    prob = cdf_upper - cdf_lower
    prob = jnp.clip(prob, 1e-10, 1.0)

    log_prob = jnp.log(prob)
    return log_prob


# ============================================================================
# Hyperprior Entropy Model (like Ballé et al. 2018)
# ============================================================================

def init_hyper_encoder(key, latent_channels, hyper_channels=128):
    """Hyperprior encoder: encodes latent statistics."""
    keys = random.split(key, 6)
    params = {
        'conv1': init_conv(keys[0], latent_channels, hyper_channels, 3),
        'conv2': init_conv(keys[1], hyper_channels, hyper_channels, 3),
        'conv3': init_conv(keys[2], hyper_channels, hyper_channels, 3),
        'gdn1': init_gdn(keys[3], hyper_channels),
        'gdn2': init_gdn(keys[4], hyper_channels),
    }
    return params


def hyper_encoder_forward(params, y):
    """
    Encode latent tensor to hyperprior.
    y: (N, C, H, W) latent representation
    Returns: z (N, hyper_C, H', W') hyper-latent
    """
    h = jnp.abs(y)
    h = conv2d(params['conv1'], h, stride=1)
    h = jax.nn.leaky_relu(h, negative_slope=0.2)

    h = conv2d(params['conv2'], h, stride=2)
    h = jax.nn.leaky_relu(h, negative_slope=0.2)

    h = conv2d(params['conv3'], h, stride=2)
    return h


def init_hyper_decoder(key, hyper_channels=128, latent_channels=192):
    """Hyperprior decoder: decodes to predicted means and scales."""
    keys = random.split(key, 6)
    params = {
        'conv_t1': init_conv_transpose(keys[0], hyper_channels, hyper_channels, 3),
        'conv_t2': init_conv_transpose(keys[1], hyper_channels, hyper_channels, 3),
        'conv_means': init_conv(keys[2], hyper_channels, latent_channels, 3),
        'conv_scales': init_conv(keys[3], hyper_channels, latent_channels, 3),
    }
    return params


def hyper_decoder_forward(params, z):
    """
    Decode hyperprior to predicted means and scales for the latent.
    z: (N, hyper_C, H', W')
    Returns: means, scales each (N, latent_C, H, W)
    """
    h = conv_transpose2d(params['conv_t1'], z, stride=2)
    h = jax.nn.leaky_relu(h, negative_slope=0.2)

    h = conv_transpose2d(params['conv_t2'], h, stride=2)
    h = jax.nn.leaky_relu(h, negative_slope=0.2)

    means = conv2d(params['conv_means'], h, stride=1)
    scales = conv2d(params['conv_scales'], h, stride=1)
    scales = jnp.exp(scales) + 1e-6  # Ensure positive

    return means, scales


def hyperprior_log_prob(y, means, scales):
    """
    Compute log probability using predicted means and scales (Gaussian/logistic).
    """
    centered = y - means
    upper = (centered + 0.5) / scales
    lower = (centered - 0.5) / scales

    cdf_upper = jax.nn.sigmoid(upper)
    cdf_lower = jax.nn.sigmoid(lower)

    prob = cdf_upper - cdf_lower
    prob = jnp.clip(prob, 1e-10, 1.0)

    return jnp.log(prob)


# ============================================================================
# Context Model (Autoregressive / Masked Conv for CMix-like mixing)
# ============================================================================

def init_masked_conv(key, in_channels, out_channels, kernel_size=5):
    """
    Initialize a masked convolution (causal) for autoregressive context modeling.
    The mask ensures we only look at previously decoded elements.
    """
    params = init_conv(key, in_channels, out_channels, kernel_size)
    # Create the mask: zero out the center and everything after in raster order
    k = kernel_size
    mask = jnp.ones((out_channels, in_channels, k, k))
    center = k // 2
    # Zero out bottom half
    mask = mask.at[:, :, center + 1:, :].set(0.0)
    # Zero out right side of center row (including center)
    mask = mask.at[:, :, center, center:].set(0.0)
    params['mask'] = mask
    return params


def masked_conv2d(params, x, stride=1, padding='SAME'):
    """Apply masked 2D convolution."""
    masked_w = params['w'] * params['mask']
    new_params = {'w': masked_w, 'b': params['b']}
    return conv2d(new_params, x, stride=stride, padding=padding)


def init_context_model(key, latent_channels=192):
    """
    Context model using masked convolutions for autoregressive prediction.
    CMix-like: combines multiple context predictions.
    """
    keys = random.split(key, 4)
    params = {
        'masked_conv1': init_masked_conv(keys[0], latent_channels, latent_channels * 2, 5),
        'masked_conv2': init_masked_conv(keys[1], latent_channels * 2, latent_channels * 2, 3),
        'conv_out_means': init_conv(keys[2], latent_channels * 2, latent_channels, 1),
        'conv_out_scales': init_conv(keys[3], latent_channels * 2, latent_channels, 1),
    }
    return params


def context_model_forward(params, y_hat):
    """
    Predict means and scales from already-decoded context.
    y_hat: (N, C, H, W) quantized latents
    """
    h = masked_conv2d(params['masked_conv1'], y_hat)
    h = jax.nn.leaky_relu(h, negative_slope=0.2)

    h = masked_conv2d(params['masked_conv2'], h)
    h = jax.nn.leaky_relu(h, negative_slope=0.2)

    ctx_means = conv2d(params['conv_out_means'], h, stride=1)
    ctx_scales = conv2d(params['conv_out_scales'], h, stride=1)
    ctx_scales = jnp.exp(ctx_scales) + 1e-6

    return ctx_means, ctx_scales


# ============================================================================
# CMix-like Mixer: Combines multiple predictions
# ============================================================================

def init_mixer(key, num_experts=3, latent_channels=192):
    """
    CMix-style mixer that combines predictions from:
    1. Hyperprior
    2. Autoregressive context
    3. Simple factorized prior
    
    Learns adaptive weights to combine them.
    """
    keys = random.split(key, 3)
    params = {
        # Gating network: takes concatenated features and predicts weights
        'gate_conv1': init_conv(keys[0], latent_channels * num_experts, latent_channels, 1),
        'gate_conv2': init_conv(keys[1], latent_channels, num_experts, 1),
        # Scale combination
        'scale_gate_conv1': init_conv(keys[2], latent_channels * num_experts, num_experts, 1),
    }
    return params


def mixer_forward(params, means_list, scales_list, latent_channels):
    """
    CMix-style mixing of multiple entropy model predictions.
    means_list: list of (N, C, H, W) mean predictions
    scales_list: list of (N, C, H, W) scale predictions
    Returns: mixed means, mixed scales
    """
    num_experts = len(means_list)

    # Concatenate all means for gating
    means_cat = jnp.concatenate(means_list, axis=1)  # (N, num_experts*C, H, W)

    # Gating network for means
    gate = conv2d(params['gate_conv1'], means_cat, stride=1)
    gate = jax.nn.leaky_relu(gate, negative_slope=0.2)
    gate = conv2d(params['gate_conv2'], gate, stride=1)  # (N, num_experts, H, W)
    gate_weights = jax.nn.softmax(gate, axis=1)  # (N, num_experts, H, W)

    # Weighted combination of means
    means_stack = jnp.stack(means_list, axis=1)  # (N, num_experts, C, H, W)
    gate_expanded = gate_weights[:, :, None, :, :]  # (N, num_experts, 1, H, W)
    mixed_means = jnp.sum(means_stack * gate_expanded, axis=1)  # (N, C, H, W)

    # Gating for scales
    scales_cat = jnp.concatenate(scales_list, axis=1)
    scale_gate = conv2d(params['scale_gate_conv1'], scales_cat, stride=1)
    scale_gate_weights = jax.nn.softmax(scale_gate, axis=1)

    scales_stack = jnp.stack(scales_list, axis=1)
    scale_gate_expanded = scale_gate_weights[:, :, None, :, :]
    mixed_scales = jnp.sum(scales_stack * scale_gate_expanded, axis=1)
    mixed_scales = jnp.clip(mixed_scales, 1e-6, None)

    return mixed_means, mixed_scales


# ============================================================================
# Main Encoder Network
# ============================================================================

def init_encoder(key, in_channels=3, latent_channels=192):
    """
    Analysis transform (encoder): image -> latent representation.
    Uses strided convolutions with GDN activation (standard in learned compression).
    """
    keys = random.split(key, 10)
    params = {
        'conv1': init_conv(keys[0], in_channels, 128, 5),
        'gdn1': init_gdn(keys[1], 128),
        'conv2': init_conv(keys[2], 128, 128, 5),
        'gdn2': init_gdn(keys[3], 128),
        'conv3': init_conv(keys[4], 128, 192, 5),
        'gdn3': init_gdn(keys[5], 192),
        'conv4': init_conv(keys[6], 192, latent_channels, 5),
        # Residual refinement (CMix-inspired: multiple passes)
        'res_conv1': init_conv(keys[7], latent_channels, latent_channels, 3),
        'res_conv2': init_conv(keys[8], latent_channels, latent_channels, 3),
    }
    return params


def encoder_forward(params, x):
    """
    Encode image to latent representation.
    x: (N, 3, H, W) image in [0, 1]
    Returns: y (N, latent_C, H/16, W/16) latent
    """
    h = conv2d(params['conv1'], x, stride=2)
    h = gdn(h, params['gdn1']['gamma'], params['gdn1']['beta'])

    h = conv2d(params['conv2'], h, stride=2)
    h = gdn(h, params['gdn2']['gamma'], params['gdn2']['beta'])

    h = conv2d(params['conv3'], h, stride=2)
    h = gdn(h, params['gdn3']['gamma'], params['gdn3']['beta'])

    h = conv2d(params['conv4'], h, stride=2)

    # Residual refinement block
    res = conv2d(params['res_conv1'], h, stride=1)
    res = jax.nn.leaky_relu(res, negative_slope=0.2)
    res = conv2d(params['res_conv2'], res, stride=1)
    h = h + res * 0.1

    return h


# ============================================================================
# Main Decoder Network
# ============================================================================

def init_decoder(key, latent_channels=192, out_channels=3):
    """
    Synthesis transform (decoder): latent representation -> reconstructed image.
    Uses transposed convolutions with inverse GDN.
    """
    keys = random.split(key, 10)
    params = {
        # Residual refinement first
        'res_conv1': init_conv(keys[0], latent_channels, latent_channels, 3),
        'res_conv2': init_conv(keys[1], latent_channels, latent_channels, 3),
        'conv_t1': init_conv_transpose(keys[2], latent_channels, 192, 5),
        'igdn1': init_gdn(keys[3], 192),
        'conv_t2': init_conv_transpose(keys[4], 192, 128, 5),
        'igdn2': init_gdn(keys[5], 128),
        'conv_t3': init_conv_transpose(keys[6], 128, 128, 5),
        'igdn3': init_gdn(keys[7], 128),
        'conv_t4': init_conv_transpose(keys[8], 128, out_channels, 5),
    }
    return params


def decoder_forward(params, y_hat):
    """
    Decode latent to reconstructed image.
    y_hat: (N, latent_C, H/16, W/16)
    Returns: x_hat (N, 3, H, W) in [0, 1]
    """
    # Residual refinement
    res = conv2d(params['res_conv1'], y_hat, stride=1)
    res = jax.nn.leaky_relu(res, negative_slope=0.2)
    res = conv2d(params['res_conv2'], res, stride=1)
    h = y_hat + res * 0.1

    h = conv_transpose2d(params['conv_t1'], h, stride=2)
    h = gdn(h, params['igdn1']['gamma'], params['igdn1']['beta'], inverse=True)

    h = conv_transpose2d(params['conv_t2'], h, stride=2)
    h = gdn(h, params['igdn2']['gamma'], params['igdn2']['beta'], inverse=True)

    h = conv_transpose2d(params['conv_t3'], h, stride=2)
    h = gdn(h, params['igdn3']['gamma'], params['igdn3']['beta'], inverse=True)

    h = conv_transpose2d(params['conv_t4'], h, stride=2)
    x_hat = jax.nn.sigmoid(h)

    return x_hat


# ============================================================================
# Quantization (STE - Straight Through Estimator for training)
# ============================================================================

def quantize(y, is_training=True, key=None):
    """
    Quantize latents. During training, add uniform noise as a differentiable
    proxy. During inference, round to nearest integer.
    """
    if is_training and key is not None:
        # Add uniform noise U(-0.5, 0.5) as differentiable proxy for rounding
        noise = random.uniform(key, y.shape, minval=-0.5, maxval=0.5)
        y_hat = y + noise
    else:
        y_hat = jnp.round(y)
    return y_hat


# ============================================================================
# Full Compression Model
# ============================================================================

def init_model(key, in_channels=3, latent_channels=192, hyper_channels=128):
    """Initialize all model parameters."""
    keys = random.split(key, 8)
    params = {
        'encoder': init_encoder(keys[0], in_channels, latent_channels),
        'decoder': init_decoder(keys[1], latent_channels, in_channels),
        'hyper_encoder': init_hyper_encoder(keys[2], latent_channels, hyper_channels),
        'hyper_decoder': init_hyper_decoder(keys[3], hyper_channels, latent_channels),
        'context_model': init_context_model(keys[4], latent_channels),
        'entropy_model': init_entropy_model(keys[5], latent_channels),
        'mixer': init_mixer(keys[6], num_experts=3, latent_channels=latent_channels),
        'latent_channels': latent_channels,
        'hyper_channels': hyper_channels,
    }
    return params


def model_forward(params, x, key, is_training=True, lam=0.01):
    """
    Full forward pass: encode -> quantize -> decode with rate-distortion loss.
    
    x: (N, 3, H, W) images in [0, 1]
    lam: rate-distortion tradeoff (higher = more quality, less compression)
    
    Returns: loss, aux_info
    """
    latent_channels = params['latent_channels']
    k1, k2, k3 = random.split(key, 3)

    # ---- Encode ----
    y = encoder_forward(params['encoder'], x)

    # ---- Quantize latent ----
    y_hat = quantize(y, is_training=is_training, key=k1)

    # ---- Decode ----
    x_hat = decoder_forward(params['decoder'], y_hat)

    # ---- Hyper encoder/decoder ----
    z = hyper_encoder_forward(params['hyper_encoder'], y)
    z_hat = quantize(z, is_training=is_training, key=k2)
    hyper_means, hyper_scales = hyper_decoder_forward(params['hyper_decoder'], z_hat)

    # ---- Context model (autoregressive) ----
    ctx_means, ctx_scales = context_model_forward(params['context_model'], y_hat)

    # ---- Factorized prior ----
    fact_means = params['entropy_model']['channel_means'][None, :, None, None]
    fact_means = jnp.broadcast_to(fact_means, y_hat.shape)
    fact_scales = jnp.exp(params['entropy_model']['channel_log_scales'])[None, :, None, None]
    fact_scales = jnp.broadcast_to(fact_scales + 1e-6, y_hat.shape)

    # ---- CMix-style mixing ----
    means_list = [hyper_means, ctx_means, fact_means]
    scales_list = [hyper_scales, ctx_scales, fact_scales]
    mixed_means, mixed_scales = mixer_forward(
        params['mixer'], means_list, scales_list, latent_channels
    )

    # ---- Rate (bits) ----
    # Rate for latent y
    log_prob_y = hyperprior_log_prob(y_hat, mixed_means, mixed_scales)
    rate_y = -jnp.sum(log_prob_y) / jnp.log(2.0)  # Convert nats to bits

    # Rate for hyper-latent z (use simple factorized prior)
    log_prob_z = entropy_model_log_prob(params['entropy_model'], z_hat)
    rate_z = -jnp.sum(log_prob_z) / jnp.log(2.0)

    total_rate = (rate_y + rate_z) / x.shape[0]  # Per image

    # ---- Distortion ----
    # MSE distortion
    mse = jnp.mean((x - x_hat) ** 2)

    # MS-SSIM approximation (simplified structural similarity component)
    # Using a simple perceptual-ish loss alongside MSE
    distortion = mse

    # ---- Rate-Distortion Loss ----
    # R + lambda * D (minimize rate subject to distortion constraint)
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
    bpp = total_rate / (x.shape[2] * x.shape[3])  # bits per pixel

    loss = lam * 255.0 ** 2 * distortion + bpp

    aux = {
        'loss': loss,
        'mse': mse,
        'psnr': -10.0 * jnp.log10(mse + 1e-10),
        'bpp': bpp,
        'rate_y': rate_y / x.shape[0],
        'rate_z': rate_z / x.shape[0],
        'x_hat': x_hat,
        'y_hat': y_hat,
        'z_hat': z_hat,
    }

    return loss, aux


# ============================================================================
# Adam Optimizer (Pure JAX)
# ============================================================================

def init_adam(params, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
    """Initialize Adam optimizer state."""
    def zeros_like_tree(tree):
        return jax.tree.map(lambda x: jnp.zeros_like(x) if isinstance(x, jnp.ndarray) else x, tree)

    state = {
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
        'step': 0,
        'm': zeros_like_tree(params),
        'v': zeros_like_tree(params),
    }
    return state


def adam_update(params, grads, state):
    """Perform one Adam update step."""
    step = state['step'] + 1
    lr = state['lr']
    beta1 = state['beta1']
    beta2 = state['beta2']
    eps = state['eps']

    def update_fn(p, g, m, v):
        if not isinstance(p, jnp.ndarray):
            return p, m, v
        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m_new / (1 - beta1 ** step)
        v_hat = v_new / (1 - beta2 ** step)
        p_new = p - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return p_new, m_new, v_new

    new_params, new_m, new_v = jax.tree.map(
        update_fn, params, grads, state['m'], state['v'],
        is_leaf=lambda x: not isinstance(x, dict)
    )

    # Separate the tuple outputs
    new_params = jax.tree.map(lambda x: x[0] if isinstance(x, tuple) else x, new_params)
    new_m_state = jax.tree.map(lambda x: x[1] if isinstance(x, tuple) else x, 
                                jax.tree.map(update_fn, params, grads, state['m'], state['v'],
                                            is_leaf=lambda x: not isinstance(x, dict)))
    new_v_state = jax.tree.map(lambda x: x[2] if isinstance(x, tuple) else x,
                                jax.tree.map(update_fn, params, grads, state['m'], state['v'],
                                            is_leaf=lambda x: not isinstance(x, dict)))

    new_state = {
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
        'step': step,
        'm': new_m_state,
        'v': new_v_state,
    }
    return new_params, new_state


# Simpler Adam using optax-style flat updates
def adam_step(params, grads, opt_state):
    """Simplified Adam step operating on pytrees."""
    step = opt_state['step'] + 1
    lr = opt_state['lr']
    b1, b2, eps = opt_state['beta1'], opt_state['beta2'], opt_state['eps']

    new_m = jax.tree.map(
        lambda m, g: b1 * m + (1 - b1) * g if isinstance(m, jnp.ndarray) else m,
        opt_state['m'], grads
    )
    new_v = jax.tree.map(
        lambda v, g: b2 * v + (1 - b2) * (g ** 2) if isinstance(v, jnp.ndarray) else v,
        opt_state['v'], grads
    )

    bc1 = 1 - b1 ** step
    bc2 = 1 - b2 ** step

    new_params = jax.tree.map(
        lambda p, m, v: p - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps)
        if isinstance(p, jnp.ndarray) else p,
        params, new_m, new_v
    )

    new_state = {**opt_state, 'step': step, 'm': new_m, 'v': new_v}
    return new_params, new_state


# ============================================================================
# Training Step
# ============================================================================

@partial(jit, static_argnums=(3,))
def train_step(params, opt_state, key, lam=0.01):
    """
    Single training step. Call with your batch of images.
    This returns a function; see train_step_with_data below.
    """
    pass  # Placeholder


def make_train_step(lam=0.01):
    """Create a JIT-compiled training step function."""

    @jit
    def step(params, opt_state, x, key):
        """
        x: (N, 3, H, W) batch of images in [0, 1]
        """
        def loss_fn(p):
            loss, aux = model_forward(p, x, key, is_training=True, lam=lam)
            return loss, aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Gradient clipping
        grad_norm = jnp.sqrt(sum(
            jnp.sum(g ** 2) for g in jax.tree.leaves(grads)
            if isinstance(g, jnp.ndarray)
        ))
        clip_val = 1.0
        scale = jnp.minimum(1.0, clip_val / (grad_norm + 1e-6))
        grads = jax.tree.map(
            lambda g: g * scale if isinstance(g, jnp.ndarray) else g, grads
        )

        new_params, new_opt_state = adam_step(params, grads, opt_state)

        return new_params, new_opt_state, loss, aux

    return step


# ============================================================================
# Arithmetic Coding (for actual compression/decompression)
# ============================================================================

class ArithmeticCoder:
    """
    Simple arithmetic encoder/decoder for actually compressing the latents.
    Uses the learned probability model to encode symbols.
    """
    
    def __init__(self, precision=32):
        self.precision = precision
        self.full = 1 << precision
        self.half = 1 << (precision - 1)
        self.quarter = 1 << (precision - 2)

    def encode(self, symbols, cdf_lower, cdf_upper):
        """
        Encode a sequence of symbols given their CDF intervals.
        symbols: list of int symbols
        cdf_lower: list of lower CDF values [0, 1]
        cdf_upper: list of upper CDF values [0, 1]
        Returns: bytes
        """
        low = 0
        high = self.full
        bits = []
        pending = 0

        for i in range(len(symbols)):
            range_size = high - low
            high = low + int(range_size * cdf_upper[i])
            low = low + int(range_size * cdf_lower[i])

            while True:
                if high < self.half:
                    bits.append(0)
                    bits.extend([1] * pending)
                    pending = 0
                    low = low * 2
                    high = high * 2
                elif low >= self.half:
                    bits.append(1)
                    bits.extend([0] * pending)
                    pending = 0
                    low = (low - self.half) * 2
                    high = (high - self.half) * 2
                elif low >= self.quarter and high < 3 * self.quarter:
                    pending += 1
                    low = (low - self.quarter) * 2
                    high = (high - self.quarter) * 2
                else:
                    break

        # Finalize
        pending += 1
        if low < self.quarter:
            bits.append(0)
            bits.extend([1] * pending)
        else:
            bits.append(1)
            bits.extend([0] * pending)

        # Pack bits into bytes
        while len(bits) % 8 != 0:
            bits.append(0)
        
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            byte_array.append(byte)

        return bytes(byte_array), len(symbols)


def compute_cdf_intervals(y_flat, means_flat, scales_flat):
    """
    Compute CDF intervals for arithmetic coding.
    Returns numpy arrays of cdf_lower and cdf_upper for each symbol.
    """
    y_np = np.array(y_flat, dtype=np.float32)
    means_np = np.array(means_flat, dtype=np.float32)
    scales_np = np.array(scales_flat, dtype=np.float32)

    # Use logistic CDF
    def logistic_cdf(x, mu, s):
        return 1.0 / (1.0 + np.exp(-(x - mu) / s))

    cdf_lower = logistic_cdf(y_np - 0.5, means_np, scales_np)
    cdf_upper = logistic_cdf(y_np + 0.5, means_np, scales_np)

    # Clip for numerical stability
    cdf_lower = np.clip(cdf_lower, 1e-10, 1.0 - 1e-10)
    cdf_upper = np.clip(cdf_upper, 1e-10, 1.0 - 1e-10)

    # Ensure upper > lower
    cdf_upper = np.maximum(cdf_upper, cdf_lower + 1e-10)

    return cdf_lower, cdf_upper


# ============================================================================
# Compress / Decompress Functions
# ============================================================================

def compress_image(params, x):
    """
    Compress a single image.
    x: (1, 3, H, W) image in [0, 1]
    Returns: compressed_bytes, metadata
    """
    key = random.PRNGKey(0)
    latent_channels = params['latent_channels']

    # Encode
    y = encoder_forward(params['encoder'], x)
    y_hat = jnp.round(y)

    # Hyper encode
    z = hyper_encoder_forward(params['hyper_encoder'], y)
    z_hat = jnp.round(z)

    # Get entropy parameters
    hyper_means, hyper_scales = hyper_decoder_forward(params['hyper_decoder'], z_hat)
    ctx_means, ctx_scales = context_model_forward(params['context_model'], y_hat)

    fact_means = params['entropy_model']['channel_means'][None, :, None, None]
    fact_means = jnp.broadcast_to(fact_means, y_hat.shape)
    fact_scales = jnp.exp(params['entropy_model']['channel_log_scales'])[None, :, None, None]
    fact_scales = jnp.broadcast_to(fact_scales + 1e-6, y_hat.shape)

    # Mix predictions
    means_list = [hyper_means, ctx_means, fact_means]
    scales_list = [hyper_scales, ctx_scales, fact_scales]
    mixed_means, mixed_scales = mixer_forward(
        params['mixer'], means_list, scales_list, latent_channels
    )

    # Arithmetic coding
    coder = ArithmeticCoder()

    # Encode z (hyper-latent) with factorized prior
    z_flat = np.array(z_hat.flatten(), dtype=np.float32)
    z_means = np.zeros_like(z_flat)
    z_scales = np.ones_like(z_flat)
    z_cdf_lower, z_cdf_upper = compute_cdf_intervals(z_flat, z_means, z_scales)
    z_bytes, z_len = coder.encode(z_flat.tolist(), z_cdf_lower.tolist(), z_cdf_upper.tolist())

    # Encode y (latent) with mixed model
    y_flat = np.array(y_hat.flatten(), dtype=np.float32)
    m_flat = np.array(mixed_means.flatten(), dtype=np.float32)
    s_flat = np.array(mixed_scales.flatten(), dtype=np.float32)
    y_cdf_lower, y_cdf_upper = compute_cdf_intervals(y_flat, m_flat, s_flat)
    y_bytes, y_len = coder.encode(y_flat.tolist(), y_cdf_lower.tolist(), y_cdf_upper.tolist())

    # Package
    H, W = x.shape[2], x.shape[3]
    metadata = {
        'H': H, 'W': W,
        'y_shape': y_hat.shape,
        'z_shape': z_hat.shape,
        'z_len': z_len,
        'y_len': y_len,
    }

    # Combine into single bytestream
    header = struct.pack('IIII', H, W, len(z_bytes), len(y_bytes))
    compressed = header + z_bytes + y_bytes

    return compressed, metadata


def decompress_image(params, compressed_bytes):
    """
    Decompress image from compressed bytes.
    Returns: x_hat (1, 3, H, W) reconstructed image
    
    Note: Full arithmetic decoding would require implementing the decoder side.
    This shows the structure; for simplicity we store the quantized values directly
    in a real system you'd decode from the arithmetic coded stream.
    """
    # Parse header
    H, W, z_bytes_len, y_bytes_len = struct.unpack('IIII', compressed_bytes[:16])

    # In a full implementation, you would:
    # 1. Arithmetic decode z_hat from z_bytes
    # 2. Run hyper_decoder to get initial means/scales
    # 3. Autoregressively decode y_hat using context model + mixer
    # 4. Run decoder to get x_hat

    # For now, assume we have y_hat (in practice decoded from bitstream)
    # This is the decoder path:
    # x_hat = decoder_forward(params['decoder'], y_hat)

    print(f"Decompressing image: {H}x{W}, compressed size: {len(compressed_bytes)} bytes")
    print(f"Compression ratio: {H * W * 3 / len(compressed_bytes):.2f}x")

    return None  # Would return x_hat in full implementation


# ============================================================================
# Training Loop
# ============================================================================

def train(images, num_epochs=100, batch_size=8, lr=1e-4, lam=0.01, seed=42):
    """
    Train the compression model.
    
    images: numpy array (N, 3, H, W) in [0, 1], float32
            Images should be pre-padded to multiples of 16.
    num_epochs: number of training epochs
    batch_size: batch size
    lr: learning rate
    lam: rate-distortion tradeoff
         Higher lambda = better quality, higher bitrate
         Lower lambda = lower quality, lower bitrate
         Typical range: 0.001 (low bitrate) to 0.05 (high quality)
    """
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)

    # Initialize model
    print("Initializing model...")
    params = init_model(init_key)

    # Initialize optimizer
    opt_state = init_adam(params, lr=lr)

    # Create training step
    step_fn = make_train_step(lam=lam)

    N = images.shape[0]
    steps_per_epoch = max(1, N // batch_size)

    print(f"Training with {N} images, batch_size={batch_size}, lambda={lam}")
    print(f"Image shape: {images.shape}")
    print("=" * 60)

    for epoch in range(num_epochs):
        key, shuffle_key = random.split(key)
        perm = random.permutation(shuffle_key, N)

        epoch_loss = 0.0
        epoch_bpp = 0.0
        epoch_psnr = 0.0

        for step in range(steps_per_epoch):
            key, step_key = random.split(key)
            idx = perm[step * batch_size: (step + 1) * batch_size]
            batch = jnp.array(images[np.array(idx)])

            params, opt_state, loss, aux = step_fn(params, opt_state, batch, step_key)

            epoch_loss += float(loss)
            epoch_bpp += float(aux['bpp'])
            epoch_psnr += float(aux['psnr'])

        epoch_loss /= steps_per_epoch
        epoch_bpp /= steps_per_epoch
        epoch_psnr /= steps_per_epoch

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {epoch_loss:.4f} | "
                  f"BPP: {epoch_bpp:.4f} | PSNR: {epoch_psnr:.2f} dB")

    print("=" * 60)
    print("Training complete!")
    return params


# ============================================================================
# Utility: Prepare images for compression
# ============================================================================

def prepare_image(img_array):
    """
    Prepare an image array for the compression model.
    
    img_array: numpy array (H, W, 3) uint8 or float
    Returns: jax array (1, 3, H, W) float32 in [0, 1], padded to multiple of 16
    """
    if img_array.dtype == np.uint8:
        img = img_array.astype(np.float32) / 255.0
    else:
        img = img_array.astype(np.float32)

    # Ensure [0, 1]
    img = np.clip(img, 0.0, 1.0)

    H, W = img.shape[:2]

    # Pad to multiple of 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    # Convert to NCHW
    img = np.transpose(img, (2, 0, 1))  # (3, H, W)
    img = img[np.newaxis, ...]  # (1, 3, H, W)

    return jnp.array(img), (H, W)


def reconstruct_image(x_hat, original_size):
    """
    Convert model output back to displayable image.
    
    x_hat: jax array (1, 3, H, W) float32
    original_size: (H, W) original image dimensions
    Returns: numpy array (H, W, 3) uint8
    """
    img = np.array(x_hat[0])  # (3, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Crop to original size
    H, W = original_size
    img = img[:H, :W, :]

    return img


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("CMix-like Neural Image Compression with JAX")
    print("=" * 60)

    # Create synthetic training data for demonstration
    key = random.PRNGKey(0)
    # In practice, load real images here
    # Example: images = np.array([prepare_image(img)[0] for img in your_images])
    num_train = 16
    H, W = 128, 128
    dummy_images = np.random.rand(num_train, 3, H, W).astype(np.float32)

    print(f"Training data shape: {dummy_images.shape}")

    # Train the model
    params = train(
        dummy_images,
        num_epochs=50,
        batch_size=4,
        lr=1e-4,
        lam=0.01,  # Rate-distortion tradeoff
    )

    # Compress a single image
    print("\n" + "=" * 60)
    print("Compressing test image...")
    test_img = jnp.array(dummy_images[0:1])

    compressed, metadata = compress_image(params, test_img)
    original_size = H * W * 3  # bytes (assuming 8-bit RGB)
    compressed_size = len(compressed)

    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {original_size / compressed_size:.2f}x")
    print(f"BPP: {compressed_size * 8 / (H * W):.4f}")

    # Test reconstruction quality
    _, aux = model_forward(params, test_img, random.PRNGKey(0), is_training=False)
    print(f"Reconstruction PSNR: {float(aux['psnr']):.2f} dB")
    print(f"Reconstruction MSE: {float(aux['mse']):.6f}")
