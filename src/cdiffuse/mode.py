import argparse
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from joblib import Parallel, delayed

import soundfile as sf


""" Model """


class Expert(nn.Module):
    def __init__(self, output_size=257):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(16640, 500), nn.BatchNorm1d(500), nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500, output_size), nn.BatchNorm1d(output_size), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(self.flatten(x))
        x = self.fc2(x)
        return x


class Gate(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(2570, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, num_experts), nn.BatchNorm1d(num_experts), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MODE(nn.Module):
    def __init__(self, num_experts=3, output_size=257, context=10):
        super().__init__()
        self.experts = nn.ModuleList([Expert(output_size) for _ in range(num_experts)])
        self.gate = Gate(num_experts)
        self.num_experts = num_experts
        self.output_size = output_size
        self.context = context

    def forward(self, x):
        experts_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_out = self.gate(x.view(-1, self.output_size * self.context))
        out = experts_out * gate_out.unsqueeze(-1)
        out = out.sum(1)
        return out, (experts_out, gate_out)


""" Data pipeline """


class CevaAudioSegment2Wave(object):
    """
    pydub AudioSegment to numpy array.

    Args:
            byte_to_float (bool, deafault=False): normalize from binary to float (divide by \(2^{bytes*8-1}\)
    """

    def __init__(self, byte_to_float=False, **kwargs):
        self._b2f = byte_to_float

    def __call__(self, segment):
        sig = np.array(segment.get_array_of_samples())
        if self._b2f:
            bytes_per_sample = segment.sample_width
            sig = sig.astype(np.float32)
            sig /= 2.0 ** (bytes_per_sample * 8 - 1)
        return sig


class CevaToTensor(object):
    """
    transform features from numpy array to torch Tensor.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, sig):
        return torch.tensor(sig)


class CevaSpectogram(object):
    """
    Crate a spectogram from tensor audio signal shape (batch, time).
    Args:
            samplerate (int, default=16000): Hz.
            nfft (int, default=512): number of bins in STFT.
            winlen (float, default=0.025): window len in sec.
            winstep (float, default=0.01): window stride in sec.
            window_func (str, optional): A function to create a window tensor ('hann', 'hamming' or None).
    Returns:
            Power of STFT signal tensor shape (batch, freq, time) where freq is `nfft // 2 + 1`.
    """

    def __init__(
        self,
        samplerate=16000,
        nfft=512,
        winlen=0.032,
        winstep=0.008,
        window_func="hann",
        **kwargs,
    ):
        self._nfft = nfft
        self._sr = samplerate
        # window length in samples
        self._winlen_samp = int(winlen * samplerate)
        # window step in samples
        self._winstep_samp = int(winstep * samplerate)
        if window_func == "hann":
            self._window_func = torch.hann_window(int(samplerate * winlen))
        elif window_func == "hamming":
            self._window_func = torch.hamming_window(int(samplerate * winlen))
        else:
            self._window_func = None

    def __call__(self, sig):
        magnitude_specturm = stft(
            sig,
            self._nfft,
            self._winlen_samp,
            self._winstep_samp,
            window=self._window_func,
            return_complex=False,
        )
        return magnitude_specturm


def stft(input, nfft, win_length, win_step, window=None, return_complex=False):
    """
    The STFT computes the Fourier transform of short overlapping windows of the input.
    This giving frequency components of the signal as they change over time.

    Args:
            input (Tensor): 1D time sequence signal tensor.
            nfft (int): size of Fourier transform.
            win_length (int): the size of window frame and STFT filter.
            win_step (int): the distance between neighboring sliding window frames.
            window (Tensor, optional): the optional window function.
            return_complex (bool, default=False): Whether to return magnitude or complex tensor.

    Returns:
            A tensor containing the STFT result (complex or magnitude).
    """
    from torch.fft import rfft

    wsig = input.unfold(0, win_length, win_step)

    if window is not None:
        wsig = torch.mul(wsig, window)

    padded_wsig = F.pad(wsig, [0, nfft - win_length], mode="constant", value=0)
    padded_wsig = padded_wsig.view(padded_wsig.shape[0], 1, padded_wsig.shape[1])
    fft_complex = rfft(padded_wsig)
    if return_complex:
        return fft_complex
    return fft_complex.abs()


class CevaLogMel(object):
    """
    Apply log on mel filterbanks tensor.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, mel_filter):
        log_mel_filter = torch.log10(mel_filter + 1e-10)
        return log_mel_filter


""" Enhancement """


@torch.no_grad()
def enhance_signal(
    input,
    estimate_irm,
    synt_win,
    do_vad=True,
    vad_thr=10,
    vad_attenuation=0.03,
    beta=0.04,
    **stft_args,
):
    mask = estimate_irm.T
    stft_for_enhance = input.T
    # mask[mask > 0.8] = 1  # originally from sholmi's code
    if do_vad:
        vad = np.abs(mask.sum(axis=0)) < vad_thr
    else:
        vad = np.zeros(mask.shape[1], dtype=np.bool)
    mask[0:2, :] = 0
    mask[:, vad == 1] = mask[:, vad == 1] * vad_attenuation

    est_s = enhancement(
        mask, stft_for_enhance, synt_win=synt_win, beta=beta, **stft_args
    )
    est_s = torch.tensor(est_s)
    est_stft = stft(est_s, return_complex=False, **stft_args)
    return est_s, est_stft


def enhancement(
    irm, noisy_stft, nfft, win_length, win_step, synt_win, beta=0.04, **kwargs
):
    sig_length = win_step * irm.shape[1] + win_length
    s_est = np.zeros((sig_length, 1))
    stft_size = nfft // 2 + 1

    for seg in np.arange(1, irm.shape[1]):
        time_cal = (
            np.arange((seg - 1) * win_step + 1, (seg - 1) * win_step + win_length + 1)
            - 1
        )
        time_cal = time_cal.astype("int64")

        stft_abs = np.abs(noisy_stft[:, seg - 1]).reshape(stft_size, 1)
        stft_phase = np.angle(noisy_stft[:, seg - 1]).reshape(stft_size, 1)

        rho = irm[:, seg - 1].reshape(stft_size, 1)
        if beta == 0:
            a_hat = rho * stft_abs  # eq (2)
        else:
            a_hat = (stft_abs**rho) * (
                (beta * stft_abs) ** (1 - rho)
            )  # adaptation of eq (3)

        a_hat[0:3] = a_hat[0:3] * 0.001
        a_hat_inv = a_hat[::-1]
        a_hat_inv = a_hat_inv[1 : stft_size - 1]
        a_hat_full = np.append(a_hat, a_hat_inv).reshape(win_length, 1)

        p_inv = stft_phase[::-1]
        p_inv = p_inv[1 : stft_size - 1]
        p_full = np.append(stft_phase, -p_inv).reshape(win_length, 1)

        estimated_stft = a_hat_full * np.exp(1j * p_full)
        r1 = np.real(np.fft.ifft(estimated_stft.T))

        r2 = r1.T * synt_win
        s_est[time_cal] = s_est[time_cal] + r2
    s_est = s_est.flatten() / 1.1 / np.max(np.abs(s_est))
    s_est = np.trim_zeros(s_est, trim="b")  # remove trailing zeros
    return s_est


@torch.no_grad()
def _process_file(model, noisy_file, output_path, transform, **params):
    stft_args = {
        "nfft": params["nfft"],
        "win_length": params["nfft"],
        "win_step": int(params["nfft"] * (1 - params["overlap"])),
        "window": torch.hann_window(params["nfft"]),
    }
    out_dict = {}
    # read file
    print("read wav file")
    if noisy_file.suffix == ".wav":
        sig = AudioSegment.from_file(
            noisy_file,
            sample_width=params["sample_width"],
            frame_rate=params["frame_rate"],
            channels=params["channels"],
        )
    else:
        sig = AudioSegment.from_raw(
            noisy_file,
            format="s16le",
            bitrate="16k",
            sample_width=params["sample_width"],
            frame_rate=params["frame_rate"],
            channels=params["channels"],
        )

    noisy_stft = transform(sig)
    noisy_stft = noisy_stft.squeeze()
    # reshape to get context
    context_noisy_stft = F.pad(noisy_stft, [0, 0] + [5, 4])
    context_noisy_stft = context_noisy_stft.unfold(dimension=0, size=10, step=1)

    noisy_full_stft = stft(
        torch.tensor(sig.get_array_of_samples()) / 2**15,
        return_complex=True,
        **stft_args,
    ).squeeze()

    # get IRM
    print("start evaluation")
    out, _ = model(context_noisy_stft.unsqueeze(1).contiguous())
    out = out.detach().cpu().numpy()
    out_dict["irm"] = out

    if not params["input_is_dir"]:
        current_output_path = output_path / noisy_file.name
    else:
        current_output_path = (
            output_path / noisy_file.parent.parent.name / noisy_file.parent.name
        )
        current_output_path.mkdir(parents=True, exist_ok=True)
        current_output_path = current_output_path / noisy_file.name

    # enhance
    print("start enhancing")
    enhance_sig, enhance_stft = enhance_signal(
        noisy_full_stft.detach().cpu().numpy(),
        estimate_irm=out,
        synt_win=params["synt_win"],
        vad_thr=params["spp_threshold"],
        vad_attenuation=params["spp_vad_attenuation_factor"],
        beta=params["beta"],
        **stft_args,
    )

    out = (enhance_sig.cpu().numpy().flatten() * 2**15).astype(np.int16)
    # out = (enhance_sig.cpu().numpy().flatten() * 2**15).astype(np.float32)
    # make sure data length is a multiple of '(sample_width * channels)'
    if len(out) % 2 != 0:
        out = out[: -(len(out) % 2)]

    # write wav file
    print("write wav")
    sound = AudioSegment(out, channels=1, sample_width=2, frame_rate=16000)
    _ = sound.export(f"{current_output_path}", format="wav")

    # sf.write(current_output_path, out, 16000)


def run_mode():
    print("start mode")
    # parse arguments
    input = Path("/data/ephraim/datasets_16k/noisy_testset_wav/")
    output = "/data/ephraim/output/Enhanced/mode_output_float/"
    beta = 0.04
    experts = 3
    model_path = "/data/ephraim/mode/mode3-epoch150-no_norm-new.pth"
    workers = 1
    print("load window")
    synt_win = loadmat("/data/ephraim/mode/synt_win.mat")["synt_win"]
    print("done")

    params = {
        # signal params
        "frame_rate": 16000,
        "sample_width": 2,
        "channels": 1,
        # spp params
        "spp_threshold": 10,  # 10 - no infuance on VAD; 100 - VAD has influnce
        "spp_vad_attenuation_factor": 0.5,  # 0.03 - default value ;
        "beta": float(beta),
        # stft params
        "nfft": 512,
        "overlap": 0.75,
        "eps": 1e-6,
        "synt_win": synt_win,
    }
    print("get files")
    if input.is_dir():
        input_path = input
        noisy_files = list(input.glob("**/*.wav"))
        params["input_is_dir"] = True
    elif input.is_file() and input.suffix in [".pcm", ".wav"]:
        input_path = input.parent
        noisy_files = [input]
        params["input_is_dir"] = False
    else:
        raise ValueError("input argument should be either a pcm/wav file or a folder")
    output_path = output if output is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    print(noisy_files)
    transforms = Compose(
        [
            CevaAudioSegment2Wave(byte_to_float=True),
            CevaToTensor(),
            CevaSpectogram(
                params["frame_rate"],
                params["nfft"],
                params["nfft"] / params["frame_rate"],
                (params["nfft"] * (1 - params["overlap"])) / params["frame_rate"],
                "hann",
            ),
            CevaLogMel(),
        ]
    )
    model = MODE(
        num_experts=experts, output_size=params["nfft"] // 2 + 1, context=10
    )
    print("start loading weights")
    model.load_state_dict(torch.load(model_path, map_location="cpu")["state_dict"])
    print("done")
    model.eval()

    res = Parallel(n_jobs=workers)(
        delayed(_process_file)(model, noisy_file, output_path, transforms, **params)
        for noisy_file in tqdm(noisy_files)
    )


def main():
    print("start")
    # parse arguments
    parser = argparse.ArgumentParser(description="BIU SPP evaluation")
    parser.add_argument(
        "-i",
        "--input",
        default="/data/ephraim/datasets_16k/noisy_testset_wav/",
        type=Path,
        help="input file or folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/data/ephraim/output/Enhanced/mode_output_float/",
        type=Path,
        help="output path, if None the output path will be as the input directory",
    )
    parser.add_argument(
        "--beta", default=0.04, help="beta value for BIU enhancement algorithm"
    )
    parser.add_argument("--experts", type=int, default=3, help="number of experts")
    parser.add_argument(
        "--model_path",
        type=Path,
        default="mode3-epoch150-no_norm-new.pth",
        help="model path",
    )
    parser.add_argument("--workers", type=int, default=1, help="number of workers")
    args = parser.parse_args()
    print("load window")
    synt_win = loadmat("synt_win.mat")["synt_win"]
    print("done")

    params = {
        # signal params
        "frame_rate": 16000,
        "sample_width": 2,
        "channels": 1,
        # spp params
        "spp_threshold": 10,  # 10 - no infuance on VAD; 100 - VAD has influnce
        "spp_vad_attenuation_factor": 0.5,  # 0.03 - default value ;
        "beta": float(args.beta),
        # stft params
        "nfft": 512,
        "overlap": 0.75,
        "eps": 1e-6,
        "synt_win": synt_win,
    }
    print("get files")
    if args.input.is_dir():
        input_path = args.input
        noisy_files = list(args.input.glob("**/*.wav"))
        params["input_is_dir"] = True
    elif args.input.is_file() and args.input.suffix in [".pcm", ".wav"]:
        input_path = args.input.parent
        noisy_files = [args.input]
        params["input_is_dir"] = False
    else:
        raise ValueError("input argument should be either a pcm/wav file or a folder")
    output_path = args.output if args.output is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    print(noisy_files)
    transforms = Compose(
        [
            CevaAudioSegment2Wave(byte_to_float=True),
            CevaToTensor(),
            CevaSpectogram(
                params["frame_rate"],
                params["nfft"],
                params["nfft"] / params["frame_rate"],
                (params["nfft"] * (1 - params["overlap"])) / params["frame_rate"],
                "hann",
            ),
            CevaLogMel(),
        ]
    )
    model = MODE(
        num_experts=args.experts, output_size=params["nfft"] // 2 + 1, context=10
    )
    print("start loading weights")
    model.load_state_dict(torch.load(args.model_path, map_location="cpu")["state_dict"])
    print("done")
    model.eval()

    res = Parallel(n_jobs=args.workers)(
        delayed(_process_file)(model, noisy_file, output_path, transforms, **params)
        for noisy_file in tqdm(noisy_files)
    )


if __name__ == "__main__":
    main()
