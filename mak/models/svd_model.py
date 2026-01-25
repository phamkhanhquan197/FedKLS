from torch import nn
import torch.nn.functional as F
import torch

class SVDAdapter(nn.Module):
    def __init__(self, W_res, A, B, alpha, rank, original_bias=None):
        super().__init__()
        self.A = nn.Parameter(A.clone().detach()) # Trainable
        self.B = nn.Parameter(B.clone().detach()) # Trainable
        self.alpha = alpha # LoRA scaling factor
        self.rank = rank
        self.scaling = alpha/rank
        self.bias = None if original_bias is None else nn.Parameter(original_bias.clone().detach())
        self.W_res = W_res.cuda()
        self.W_res.requires_grad = False

    def forward(self, x):
        """
        Performs the forward pass of the SVDAdapter.

        Convention: A(r, in), B(out, r) matching PEFT.
        The computation is equivalent to:
        Output = x @ (W_res + scaling * B @ A)^T + bias
               = F.linear(x, W_res + scaling * B @ A, bias)

        Args:
            x (torch.Tensor): Input tensor.
                              Expected shape: [batch_size, ..., in_features]
                              where in_features must match self.W_res.shape[1],
                              self.A.shape[1].

        Returns:
            torch.Tensor: Output tensor.
                          Shape: [batch_size, ..., out_features]
                          where out_features is self.W_res.shape[0], self.B.shape[0].
        """
        effective_weight = self.W_res + self.scaling * (self.B @ self.A)
        output = F.linear(x, effective_weight, bias=self.bias if self.bias is not None else None)
        return output
    
    def __repr__(self):
        bias_info = f", bias=None" if self.bias is None else f", bias={list(self.bias.shape)} (trainable: {self.bias.requires_grad})"
        return (
            f"{self.__class__.__name__}("
            f"W_res: {list(self.W_res.shape)} (buffer, frozen), "
            f"A: {list(self.A.shape)} (trainable: {self.A.requires_grad}), "
            f"B: {list(self.B.shape)} (trainable: {self.B.requires_grad}), "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
            f"{bias_info})"
        )

class ConvAdapter(nn.Module):
    """Adapter for Conv2d layers using LoRA, without bias."""
    def __init__(self, original_conv, W_res, A, B, alpha, rank):
        super().__init__()
        self.W_res = W_res.cuda()
        self.W_res.requires_grad = False
        self.A = nn.Parameter(A.clone().detach())  # Trainable
        self.B = nn.Parameter(B.clone().detach())  # Trainable
        self.lora_scale = alpha / rank if rank > 0 else 0.0
        self.out_channels = original_conv.out_channels
        self.in_channels = original_conv.in_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
         # B [Cout, r] @ A [r, Cin*k1*k2] = [Cout, Cin*k1*k2]
        delta_w_flat = torch.matmul(self.B, self.A) * self.lora_scale
        # Reshape back to [Cout, Cin, k1, k2]
        delta_w = delta_w_flat.view(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        )
        # Effective weight = W_res (frozen) + ΔW (trainable)
        effective_weight = self.W_res + delta_w

        return F.conv2d(x, effective_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

    def __repr__(self):
        return (f"ConvAdapter(W_res: {list(self.W_res.shape)} (buffer, frozen), "
                f"A: {list(self.A.shape)} (trainable: {self.A.requires_grad}), "
                f"B: {list(self.B.shape)} (trainable: {self.B.requires_grad}), "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.lora_scale:.4f}, "
                f"bias=None (trainable: False))")
