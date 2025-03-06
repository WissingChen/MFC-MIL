import torch
from torch import nn
import math
from torch.nn import functional as F


class FeatureFreProcessing(nn.Module):
    def __init__(self):
        super(FeatureFreProcessing,self).__init__()

    def torch_hilbert(self,x, N=None, axis=-1):
        x = torch.as_tensor(x, dtype=torch.float32)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        if N is None:
            N = x.shape[axis]
        if N <= 0:
            raise ValueError("N must be positive.")

        Xf = torch.fft.fft(x, n=N, dim=axis)
        h = torch.zeros(N, dtype=torch.float32)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2

        if x.ndim > 1:
            ind = [None] * x.ndim
            ind[axis] = slice(None)
            h = h[tuple(ind)]
        x = torch.fft.ifft(Xf * h.to(Xf.device), dim=axis).real
        return x

    def forward(self, feature):
        feature = self.torch_hilbert(feature)
        return feature

class FeatureFreProcessing(nn.Module):
    def __init__(self, input_size=512):
        super(FeatureFreProcessing,self).__init__()
        self.in_proj = nn.Linear(input_size, 512)
        self.out_proj = nn.Linear(512, input_size)

    def torch_hilbert(self,x, N=None, axis=-1):
        # x = torch.as_tensor(x, dtype=torch.float32)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        if N is None:
            N = x.shape[axis]
        if N <= 0:
            raise ValueError("N must be positive.")

        Xf = torch.fft.fft(x, n=N, dim=axis)
        h = torch.zeros(N, dtype=torch.float32)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2

        if x.ndim > 1:
            ind = [None] * x.ndim
            ind[axis] = slice(None)
            h = h[tuple(ind)]
        x = torch.fft.ifft(Xf * h.to(Xf.device), dim=axis).real
        return x

    def torch_only_phase(self, x, N=None, axis=-1):
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        if N is None:
            N = x.shape[axis]
        if N <= 0:
            raise ValueError("N must be positive.")
        
        Xf = torch.fft.fft(x)
        Xm = torch.abs(Xf)
        Xp = torch.angle(Xf)

        h = Xm.mean() * math.e ** (1j*Xp)
        h = torch.abs(torch.fft.ifft(h))
        return h

    def torch_dct(self, x):
        x = x.float()

        N, D = x.shape
        
        # Pad the input to apply FFT
        x_v2 = torch.cat([x, x.flip(dims=[1])], dim=1)
        
        # Apply FFT
        X_fft = torch.fft.rfft(x_v2, dim=1)
        
        # Take the real part (as DCT outputs real values)
        dct_output = X_fft.real[:, :D]

        return dct_output
    
    def torch_dwt(x, wavelet='haar'):
        """
        Perform 1D Discrete Wavelet Transform (DWT) on an input tensor using convolution for speed.
        Args:
            x (torch.Tensor): Input tensor of shape [N, D], where N is the number of patches and D is the feature dimension.
            wavelet (str): The type of wavelet to use for DWT, default is 'haar'.
        Returns:
            torch.Tensor: Low-frequency (approximation) components of shape [N, D//2].
            torch.Tensor: High-frequency (detail) components of shape [N, D//2].
        """
        if wavelet == 'haar':
            # Haar wavelet filter coefficients (low-pass and high-pass)
            low_pass_filter = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), 1 / torch.sqrt(torch.tensor(2.0))], device=x.device)
            high_pass_filter = torch.tensor([-1 / torch.sqrt(torch.tensor(2.0)), 1 / torch.sqrt(torch.tensor(2.0))], device=x.device)
        else:
            raise NotImplementedError(f"Wavelet {wavelet} is not implemented.")

        # Reshape filters for convolution
        low_pass_filter = low_pass_filter.view(1, 1, -1)
        high_pass_filter = high_pass_filter.view(1, 1, -1)

        # Perform convolution with low-pass filter
        approx = F.conv1d(x.unsqueeze(1), low_pass_filter, stride=2).squeeze(1)

        # Perform convolution with high-pass filter
        detail = F.conv1d(x.unsqueeze(1), high_pass_filter, stride=2).squeeze(1)

        out = torch.cat([approx, detail], dim=1)
        return out

    def forward(self, x):
        ori_x = x
        x = self.in_proj(x)
        x = self.torch_hilbert(x)
        # x = self.torch_only_phase(x)
        # x = self.torch_dct(x)
        # x = self.dwt(x)
        x = self.out_proj(x) + ori_x
        return x
