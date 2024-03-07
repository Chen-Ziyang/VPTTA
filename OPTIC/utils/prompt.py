import torch
import torch.nn as nn
import torch.nn.functional as F


class Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.01, image_size=512):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data

    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        # obtain the low frequency amplitude part
        prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous()

        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)

        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_
