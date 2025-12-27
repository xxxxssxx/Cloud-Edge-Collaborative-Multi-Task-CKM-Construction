import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        """
        We pretend we are 'fused_bias_act' backward. In the original fused version,
        they stored 'out' and used it to figure out which elements had been negative
        or positive, etc. We'll do the same sign checks in pure PyTorch.
        """
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        # Derivative of y = LeakyRelu(x)*scale is scale*(1 or negative_slope).
        # We can figure that out from 'out': if out>0 => slope=scale, else=scale*negative_slope.
        mask = (out > 0).float()
        grad_input = grad_output * (mask + (1 - mask) * negative_slope) * scale

        # The bias gradient is just a sum over spatial dimensions and batch.
        # For a 2D conv-like tensor, sum across N,H,W (keeping channel dimension).
        # We'll treat the rest of dims generally:
        dims_to_sum = list(range(grad_input.ndim))
        # remove channel dim (1)
        if grad_input.ndim > 1:
            # By convention, we assume channel is dim=1 (N,C,H,W).
            dims_to_sum.remove(1)
        grad_bias = grad_input.sum(dim=dims_to_sum)

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        """
        This is the "grad of grad" pass. The original fused code re-invokes 'fused_bias_act'
        in a certain mode. We can replicate that logic, or simply rely on standard derivatives.
        If we want to replicate exactly, we can do so, but it's simpler just to do autograd here.
        """
        (out,) = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        scale = ctx.scale

        # Derivative with respect to the second derivative is basically the same sign mask logic.
        mask = (out > 0).float()
        gradgrad_out = gradgrad_input * (mask + (1 - mask) * negative_slope) * scale

        # If gradgrad_bias is not None, we just broadcast-add it. But typically you won't see
        # higher-order grads for bias in normal usage. We'll just ignore it or broadcast it.
        if gradgrad_bias is not None and torch.numel(gradgrad_bias) != 0:
            # shape [C], need to broadcast over NxCxHxW
            shape = [1, -1] + [1]*(out.ndim-2)
            gradgrad_out = gradgrad_out + gradgrad_bias.view(*shape) * 0  # or no-op

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        """
        The forward pass that replaces fused_bias_act(input, bias, ...).
        We will add 'bias' to 'input' and apply scaled LeakyReLU, storing
        what we need for the backward.
        """
        # Expand bias to match input shape. We assume 'bias' is shape [C],
        # and input is NCHW. So do (1, C, 1, 1).
        # (If your code deals in 2D or 1D, just adapt as needed.)
        if input.ndim == 2:
            # e.g. shape [N, C], then just do add bias
            x = input + bias
        elif input.ndim >= 2:
            shape = [1, -1] + [1]*(input.ndim - 2)
            x = input + bias.view(*shape)
        else:
            # fallback
            x = input + bias

        out = F.leaky_relu(x, negative_slope=negative_slope) * scale

        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        scale = ctx.scale

        # The fused extension calls a separate backward function, so let's do the same:
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, negative_slope, scale
        )

        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=(2**0.5)):
    """
    Drop-in replacement for the fused_leaky_relu() function that used to call the extension.
    We'll simply do everything in native PyTorch. We'll switch based on CPU/GPU if you like,
    but there's no special CUDA kernel now.
    """
    # The original code does a direct call to FusedLeakyReLUFunction if device != CPU.
    # On CPU, it does a simple fallback. We can just unify and always call the custom function
    # (which itself is pure-PyTorch).
    if bias is None:
        # If for some reason bias is None, just treat it as zero
        bias = input.new_zeros(input.shape[1])
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)


class FusedLeakyReLU(nn.Module):
    """
    Same module signature as before:
        FusedLeakyReLU(channel, negative_slope=0.2, scale=2**0.5)
    Returns fused_leaky_relu(x, self.bias, negative_slope, scale).
    """
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(
            input, 
            self.bias, 
            negative_slope=self.negative_slope, 
            scale=self.scale
        )
