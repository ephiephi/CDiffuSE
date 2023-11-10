import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import matplotlib.pyplot as plt
import torchaudio.functional as F
from IPython.display import Audio

path = "/data/ephraim/datasets_16k/noisy_testset_wav/p232_013.wav"
WAVEFORM_DISTORTED, SAMPLE_RATE_NMR = torchaudio.load(path)

###objective:
objective_model = SQUIM_OBJECTIVE.get_model()

stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(WAVEFORM_DISTORTED[0:1, :])
print(f"STOI: {stoi_hyp[0]}")
print(f"PESQ: {pesq_hyp[0]}")
print(f"SI-SDR: {si_sdr_hyp[0]}\n")


x = WAVEFORM_DISTORTED
with torch.enable_grad():
    x_in = x.detach().requires_grad_(True)
    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(x_in[0:1, :])
    gradient = torch.autograd.grad(stoi_hyp, x_in)[0]
    print("gradient")
    print(gradient)
    print(gradient.shape)
    print(x_in.shape)

    # x_in = x.detach().requires_grad_(True)
    # logits = classifier(x_in, t)
    # log_probs = F.log_softmax(logits, dim=-1)
    # selected = log_probs[range(len(logits)), y.view(-1)]
    # gradient = (torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale )
    print(objective_model)
