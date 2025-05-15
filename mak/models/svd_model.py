from torch import nn
import torch.nn.functional as F

class SVDAdapter(nn.Module):
    def __init__(self, W_res, A, B, alpha, rank, original_bias=None):
        super().__init__()
        self.A = nn.Parameter(A.clone().detach()) # Trainable
        self.B = nn.Parameter(B.clone().detach()) # Trainable
        self.alpha = alpha # LoRA scaling factor
        self.rank = rank
        self.scaling = alpha/rank
        self.bias = nn.Parameter(original_bias.clone().detach())
        self.W_res = W_res.cuda()
        self.W_res.require_grad = False #Freeze the residual matrix

    def forward(self, x):
        """
        Performs the forward pass of the SVDAdapter.

        The computation is equivalent to:
        Output = x @ (W_res + scaling * A @ B)^T + bias
               = F.linear(x, W_res + scaling * A @ B, bias)

        Args:
            x (torch.Tensor): Input tensor.
                              Expected shape: [batch_size, ..., in_features]
                              where in_features must match self.W_res.shape[1],
                              self.B.shape[1].

        Returns:
            torch.Tensor: Output tensor.
                          Shape: [batch_size, ..., out_features]
                          where out_features is self.W_res.shape[0], self.A.shape[0].
        """
        effective_weight = self.W_res + (self.alpha/self.rank) * (self.A @ self.B)
        output = F.linear(x, effective_weight, self.bias)
        return output
    

    def __repr__(self):
        bias_shape = list(self.bias.shape)
        bias_trainable = self.bias.requires_grad
        bias_info = f", bias={bias_shape} (trainable: {bias_trainable})"

            
        return (
            f"{self.__class__.__name__}("
            f"W_res: {list(self.W_res.shape)} (buffer, frozen), "
            f"A: {list(self.A.shape)} (trainable: {self.A.requires_grad}), "
            f"B: {list(self.B.shape)} (trainable: {self.B.requires_grad}), "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
            f"{bias_info})"
        )

