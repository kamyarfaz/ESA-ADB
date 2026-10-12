"""Parameter-matched AE with channel attention restricted to the same channel."""
from .models import MultivariateAE
import torch


class ChannelIsolatedAE(MultivariateAE):
    """Same parameters/initialization, diagonal channel attention at inference.

    Temporal attention remains unrestricted within each channel. Parameters are
    shared across channels; outputs cannot consume another channel's input.
    Training losses and parameter updates still combine all channels.
    """
    def _mask(self, device):
        return ~torch.eye(self.C, dtype=torch.bool, device=device)

    def _encode(self, tok):
        b, ch, npatches, d = tok.shape
        t = self.temporal_encoder(tok.reshape(b*ch, npatches, d)).reshape(b,ch,npatches,d)
        t = t.permute(0,2,1,3).reshape(b*npatches,ch,d)
        t = self.channel_encoder(t, mask=self._mask(tok.device))
        return t.reshape(b,npatches,ch,d).permute(0,2,1,3)

    def _decode(self, tok, z):
        b,ch,npatches,d = tok.shape
        t = self.temporal_decoder(tok.reshape(b*ch,npatches,d), z.reshape(b*ch,npatches,d)).reshape(b,ch,npatches,d)
        tq=t.permute(0,2,1,3).reshape(b*npatches,ch,d)
        zq=z.permute(0,2,1,3).reshape(b*npatches,ch,d)
        mask=self._mask(tok.device)
        t=self.channel_decoder(tq,zq,tgt_mask=mask,memory_mask=mask)
        return t.reshape(b,npatches,ch,d).permute(0,2,1,3)
