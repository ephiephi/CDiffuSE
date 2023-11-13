from dataclasses import dataclass


# from torchaudio.models import squim_objective_base, squim_subjective_base, SquimObjective, SquimSubjective

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# from __future__ import absolute_import, division, print_function, unicode_literals
import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import torch
import warnings
import zipfile

from urllib.request import urlopen
from urllib.parse import urlparse  # noqa: F401

try:
    from tqdm.auto import (
        tqdm,
    )  # automatically select proper tqdm submodule if available
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        # fake tqdm if it's not installed
        class tqdm(object):  # type: ignore
            def __init__(
                self,
                total=None,
                disable=False,
                unit=None,
                unit_scale=None,
                unit_divisor=None,
            ):
                self.total = total
                self.disable = disable
                self.n = 0
                # ignore unit, unit_scale, unit_divisor; they're just for real tqdm

            def update(self, n):
                if self.disable:
                    return

                self.n += n
                if self.total is None:
                    sys.stderr.write("\r{0:.1f} bytes".format(self.n))
                else:
                    sys.stderr.write(
                        "\r{0:.1f}%".format(100 * self.n / float(self.total))
                    )
                sys.stderr.flush()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.disable:
                    return

                sys.stderr.write("\n")


# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")

MASTER_BRANCH = "master"
ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
VAR_DEPENDENCY = "dependencies"
MODULE_HUBCONF = "hubconf.py"
READ_DATA_CHUNK = 8192
hub_dir = None


# Copied from tools/shared/module_loader to be included in torch package
def import_module(name, path):
    if sys.version_info >= (3, 5):
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        from importlib.machinery import SourceFileLoader

        return SourceFileLoader(name, path).load_module()


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _git_archive_link(repo_owner, repo_name, branch):
    return "https://github.com/{}/{}/archive/{}.zip".format(
        repo_owner, repo_name, branch
    )


def _load_attr_from_module(module, func_name):
    # Check if callable is defined in the module
    if func_name not in dir(module):
        return None
    return getattr(module, func_name)


def _get_torch_home():
    torch_home = hub_dir
    if torch_home is None:
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
            )
        )
    return torch_home


def _setup_hubdir():
    global hub_dir
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_HUB"):
        warnings.warn("TORCH_HUB is deprecated, please use env TORCH_HOME instead")

    if hub_dir is None:
        torch_home = _get_torch_home()
        hub_dir = os.path.join(torch_home, "hub")

    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)


def _parse_repo_info(github):
    branch = MASTER_BRANCH
    if ":" in github:
        repo_info, branch = github.split(":")
    else:
        repo_info = github
    repo_owner, repo_name = repo_info.split("/")
    return repo_owner, repo_name, branch


def _get_cache_or_reload(github, force_reload, verbose=True):
    # Parse github repo information
    repo_owner, repo_name, branch = _parse_repo_info(github)
    # Github allows branch name with slash '/',
    # this causes confusion with path on both Linux and Windows.
    # Backslash is not allowed in Github branch name so no need to
    # to worry about it.
    normalized_br = branch.replace("/", "_")
    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    repo_dir = os.path.join(hub_dir, "_".join([repo_owner, repo_name, normalized_br]))

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        if verbose:
            sys.stderr.write("Using cache found in {}\n".format(repo_dir))
    else:
        cached_file = os.path.join(hub_dir, normalized_br + ".zip")
        _remove_if_exists(cached_file)

        url = _git_archive_link(repo_owner, repo_name, branch)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, progress=False)

        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            # Unzip the code and rename the base folder
            cached_zipfile.extractall(hub_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    return repo_dir


def _check_module_exists(name):
    if sys.version_info >= (3, 4):
        import importlib.util

        return importlib.util.find_spec(name) is not None
    elif sys.version_info >= (3, 3):
        # Special case for python3.3
        import importlib.find_loader

        return importlib.find_loader(name) is not None
    else:
        # NB: Python2.7 imp.find_module() doesn't respect PEP 302,
        #     it cannot find a package installed as .egg(zip) file.
        #     Here we use workaround from:
        #     https://stackoverflow.com/questions/28962344/imp-find-module-which-supports-zipped-eggs?lq=1
        #     Also imp doesn't handle hierarchical module names (names contains dots).
        try:
            # 1. Try imp.find_module(), which searches sys.path, but does
            # not respect PEP 302 import hooks.
            import imp

            result = imp.find_module(name)
            if result:
                return True
        except ImportError:
            pass
        path = sys.path
        for item in path:
            # 2. Scan path for import hooks. sys.path_importer_cache maps
            # path items to optional "importer" objects, that implement
            # find_module() etc.  Note that path must be a subset of
            # sys.path for this to work.
            importer = sys.path_importer_cache.get(item)
            if importer:
                try:
                    result = importer.find_module(name, [item])
                    if result:
                        return True
                except ImportError:
                    pass
        return False


def _check_dependencies(m):
    dependencies = _load_attr_from_module(m, VAR_DEPENDENCY)

    if dependencies is not None:
        missing_deps = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if len(missing_deps):
            raise RuntimeError(
                "Missing dependencies: {}".format(", ".join(missing_deps))
            )


def _load_entry_from_hubconf(m, model):
    if not isinstance(model, str):
        raise ValueError("Invalid input: model should be a string of function name")

    # Note that if a missing dependency is imported at top level of hubconf, it will
    # throw before this function. It's a chicken and egg situation where we have to
    # load hubconf to know what're the dependencies, but to import hubconf it requires
    # a missing package. This is fine, Python will throw proper error message for users.
    _check_dependencies(m)

    func = _load_attr_from_module(m, model)

    if func is None or not callable(func):
        raise RuntimeError("Cannot find callable {} in hubconf".format(model))

    return func


def set_dir(d):
    r"""
    Optionally set hub_dir to a local dir to save downloaded models & weights.

    If ``set_dir`` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if the environment
    variable is not set.


    Args:
        d (string): path to a local folder to save downloaded models & weights.
    """
    global hub_dir
    hub_dir = d


def list(github, force_reload=False):
    r"""
    List all entrypoints available in `github` hubconf.

    Args:
        github (string): a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        >>> entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    """
    # Setup hub_dir to save downloaded files
    _setup_hubdir()

    repo_dir = _get_cache_or_reload(github, force_reload, True)

    sys.path.insert(0, repo_dir)

    hub_module = import_module(MODULE_HUBCONF, repo_dir + "/" + MODULE_HUBCONF)

    sys.path.remove(repo_dir)

    # We take functions starts with '_' as internal helper functions
    entrypoints = [
        f
        for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith("_")
    ]

    return entrypoints


def help(github, model, force_reload=False):
    r"""
    Show the docstring of entrypoint `model`.

    Args:
        github (string): a string with format <repo_owner/repo_name[:tag_name]> with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model (string): a string of entrypoint name defined in repo's hubconf.py
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Example:
        >>> print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
    """
    # Setup hub_dir to save downloaded files
    _setup_hubdir()

    repo_dir = _get_cache_or_reload(github, force_reload, True)

    sys.path.insert(0, repo_dir)

    hub_module = import_module(MODULE_HUBCONF, repo_dir + "/" + MODULE_HUBCONF)

    sys.path.remove(repo_dir)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry.__doc__


# Ideally this should be `def load(github, model, *args, forece_reload=False, **kwargs):`,
# but Python2 complains syntax error for it. We have to skip force_reload in function
# signature here but detect it in kwargs instead.
# TODO: fix it after Python2 EOL
def load(github, model, *args, **kwargs):
    r"""
    Load a model from a github repo, with pretrained weights.

    Args:
        github (string): a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model (string): a string of entrypoint name defined in repo's hubconf.py
        *args (optional): the corresponding args for callable `model`.
        force_reload (bool, optional): whether to force a fresh download of github repo unconditionally.
            Default is `False`.
        verbose (bool, optional): If False, mute messages about hitting local caches. Note that the message
            about first download is cannot be muted.
            Default is `True`.
        **kwargs (optional): the corresponding kwargs for callable `model`.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    """
    # Setup hub_dir to save downloaded files
    _setup_hubdir()

    force_reload = kwargs.get("force_reload", False)
    kwargs.pop("force_reload", None)
    verbose = kwargs.get("verbose", True)
    kwargs.pop("verbose", None)

    repo_dir = _get_cache_or_reload(github, force_reload, verbose)

    sys.path.insert(0, repo_dir)

    hub_module = import_module(MODULE_HUBCONF, repo_dir + "/" + MODULE_HUBCONF)

    entry = _load_entry_from_hubconf(hub_module, model)

    model = entry(*args, **kwargs)

    sys.path.remove(repo_dir)

    return model


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest
                    )
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def _download_url_to_file(url, dst, hash_prefix=None, progress=True):
    warnings.warn(
        "torch.hub._download_url_to_file has been renamed to\
            torch.hub.download_url_to_file to be a public API,\
            _download_url_to_file will be removed in after 1.3 release"
    )
    download_url_to_file(url, dst, hash_prefix, progress)


def load_state_dict_from_url(
    url, model_dir=None, map_location=None, progress=True, check_hash=False
):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1) if check_hash else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    if zipfile.is_zipfile(cached_file):
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            members = cached_zipfile.infolist()
            if len(members) != 1:
                raise RuntimeError("Only one file(not dir) is allowed in the zipfile")
            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            cached_file = os.path.join(model_dir, extraced_name)

    return torch.load(cached_file, map_location=map_location)


def transform_wb_pesq_range(x: float) -> float:
    """The metric defined by ITU-T P.862 is often called 'PESQ score', which is defined
    for narrow-band signals and has a value range of [-0.5, 4.5] exactly. Here, we use the metric
    defined by ITU-T P.862.2, commonly known as 'wide-band PESQ' and will be referred to as "PESQ score".

    Args:
        x (float): Narrow-band PESQ score.

    Returns:
        (float): Wide-band PESQ score.
    """
    return 0.999 + (4.999 - 0.999) / (1 + math.exp(-1.3669 * x + 3.8224))


PESQRange: Tuple[float, float] = (
    1.0,  # P.862.2 uses a different input filter than P.862, and the lower bound of
    # the raw score is not -0.5 anymore. It's hard to figure out the true lower bound.
    # We are using 1.0 as a reasonable approximation.
    transform_wb_pesq_range(4.5),
)


class RangeSigmoid(nn.Module):
    def __init__(self, val_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        super(RangeSigmoid, self).__init__()
        assert isinstance(val_range, tuple) and len(val_range) == 2
        self.val_range: Tuple[float, float] = val_range
        self.sigmoid: nn.modules.Module = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (
            self.sigmoid(x) * (self.val_range[1] - self.val_range[0])
            + self.val_range[0]
        )
        return out


class Encoder(nn.Module):
    """Encoder module that transform 1D waveform to 2D representations.

    Args:
        feat_dim (int, optional): The feature dimension after Encoder module. (Default: 512)
        win_len (int, optional): kernel size in the Conv1D layer. (Default: 32)
    """

    def __init__(self, feat_dim: int = 512, win_len: int = 32) -> None:
        super(Encoder, self).__init__()

        self.conv1d = nn.Conv1d(1, feat_dim, win_len, stride=win_len // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply waveforms to convolutional layer and ReLU layer.

        Args:
            x (torch.Tensor): Input waveforms. Tensor with dimensions `(batch, time)`.

        Returns:
            (torch,Tensor): Feature Tensor with dimensions `(batch, channel, frame)`.
        """
        out = x.unsqueeze(dim=1)
        out = F.relu(self.conv1d(out))
        return out


class SingleRNN(nn.Module):
    def __init__(
        self, rnn_type: str, input_size: int, hidden_size: int, dropout: float = 0.0
    ) -> None:
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn: nn.modules.Module = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: batch, seq, dim
        out, _ = self.rnn(x)
        out = self.proj(out)
        return out


class DPRNN(nn.Module):
    """*Dual-path recurrent neural networks (DPRNN)* :cite:`luo2020dual`.

    Args:
        feat_dim (int, optional): The feature dimension after Encoder module. (Default: 64)
        hidden_dim (int, optional): Hidden dimension in the RNN layer of DPRNN. (Default: 128)
        num_blocks (int, optional): Number of DPRNN layers. (Default: 6)
        rnn_type (str, optional): Type of RNN in DPRNN. Valid options are ["RNN", "LSTM", "GRU"]. (Default: "LSTM")
        d_model (int, optional): The number of expected features in the input. (Default: 256)
        chunk_size (int, optional): Chunk size of input for DPRNN. (Default: 100)
        chunk_stride (int, optional): Stride of chunk input for DPRNN. (Default: 50)
    """

    def __init__(
        self,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 6,
        rnn_type: str = "LSTM",
        d_model: int = 256,
        chunk_size: int = 100,
        chunk_stride: int = 50,
    ) -> None:
        super(DPRNN, self).__init__()

        self.num_blocks = num_blocks

        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for _ in range(num_blocks):
            self.row_rnn.append(SingleRNN(rnn_type, feat_dim, hidden_dim))
            self.col_rnn.append(SingleRNN(rnn_type, feat_dim, hidden_dim))
            self.row_norm.append(nn.GroupNorm(1, feat_dim, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, feat_dim, eps=1e-8))
        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim, d_model, 1),
            nn.PReLU(),
        )
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride

    def pad_chunk(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # input shape: (B, N, T)
        seq_len = x.shape[-1]

        rest = (
            self.chunk_size
            - (self.chunk_stride + seq_len % self.chunk_size) % self.chunk_size
        )
        out = F.pad(x, [self.chunk_stride, rest + self.chunk_stride])

        return out, rest

    def chunking(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        out, rest = self.pad_chunk(x)
        batch_size, feat_dim, seq_len = out.shape

        segments1 = (
            out[:, :, : -self.chunk_stride]
            .contiguous()
            .view(batch_size, feat_dim, -1, self.chunk_size)
        )
        segments2 = (
            out[:, :, self.chunk_stride :]
            .contiguous()
            .view(batch_size, feat_dim, -1, self.chunk_size)
        )
        out = torch.cat([segments1, segments2], dim=3)
        out = (
            out.view(batch_size, feat_dim, -1, self.chunk_size)
            .transpose(2, 3)
            .contiguous()
        )

        return out, rest

    def merging(self, x: torch.Tensor, rest: int) -> torch.Tensor:
        batch_size, dim, _, _ = x.shape
        out = (
            x.transpose(2, 3)
            .contiguous()
            .view(batch_size, dim, -1, self.chunk_size * 2)
        )
        out1 = (
            out[:, :, :, : self.chunk_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, self.chunk_stride :]
        )
        out2 = (
            out[:, :, :, self.chunk_size :]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, : -self.chunk_stride]
        )
        out = out1 + out2
        if rest > 0:
            out = out[:, :, :-rest]
        out = out.contiguous()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, rest = self.chunking(x)
        batch_size, _, dim1, dim2 = x.shape
        out = x
        for row_rnn, row_norm, col_rnn, col_norm in zip(
            self.row_rnn, self.row_norm, self.col_rnn, self.col_norm
        ):
            row_in = (
                out.permute(0, 3, 2, 1)
                .contiguous()
                .view(batch_size * dim2, dim1, -1)
                .contiguous()
            )
            row_out = row_rnn(row_in)
            row_out = (
                row_out.view(batch_size, dim2, dim1, -1)
                .permute(0, 3, 2, 1)
                .contiguous()
            )
            row_out = row_norm(row_out)
            out = out + row_out

            col_in = (
                out.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * dim1, dim2, -1)
                .contiguous()
            )
            col_out = col_rnn(col_in)
            col_out = (
                col_out.view(batch_size, dim1, dim2, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            col_out = col_norm(col_out)
            out = out + col_out
        out = self.conv(out)
        out = self.merging(out, rest)
        out = out.transpose(1, 2).contiguous()
        return out


class AutoPool(nn.Module):
    def __init__(self, pool_dim: int = 1) -> None:
        super(AutoPool, self).__init__()
        self.pool_dim: int = pool_dim
        self.softmax: nn.modules.Module = nn.Softmax(dim=pool_dim)
        self.register_parameter("alpha", nn.Parameter(torch.ones(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.softmax(torch.mul(x, self.alpha))
        out = torch.sum(torch.mul(x, weight), dim=self.pool_dim)
        return out


class SquimObjective(nn.Module):
    """Speech Quality and Intelligibility Measures (SQUIM) model that predicts **objective** metric scores
    for speech enhancement (e.g., STOI, PESQ, and SI-SDR).

    Args:
        encoder (torch.nn.Module): Encoder module to transform 1D waveform to 2D feature representation.
        dprnn (torch.nn.Module): DPRNN module to model sequential feature.
        branches (torch.nn.ModuleList): Transformer branches in which each branch estimate one objective metirc score.
    """

    def __init__(
        self,
        encoder: nn.Module,
        dprnn: nn.Module,
        branches: nn.ModuleList,
    ):
        super(SquimObjective, self).__init__()
        self.encoder = encoder
        self.dprnn = dprnn
        self.branches = branches

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input waveforms. Tensor with dimensions `(batch, time)`.

        Returns:
            List(torch.Tensor): List of score Tenosrs. Each Tensor is with dimension `(batch,)`.
        """
        if x.ndim != 2:
            raise ValueError(
                f"The input must be a 2D Tensor. Found dimension {x.ndim}."
            )
        x = x / (torch.mean(x**2, dim=1, keepdim=True) ** 0.5 * 20)
        out = self.encoder(x)
        out = self.dprnn(out)
        scores = []
        for branch in self.branches:
            scores.append(branch(out).squeeze(dim=1))
        return scores


def _create_branch(d_model: int, nhead: int, metric: str) -> nn.modules.Module:
    """Create branch module after DPRNN model for predicting metric score.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): Number of heads in the multi-head attention model.
        metric (str): The metric name to predict.

    Returns:
        (nn.Module): Returned module to predict corresponding metric score.
    """
    layer1 = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=0.0)
    layer2 = AutoPool()
    if metric == "stoi":
        layer3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, 1),
            RangeSigmoid(),
        )
    elif metric == "pesq":
        layer3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, 1),
            RangeSigmoid(val_range=PESQRange),
        )
    else:
        layer3: nn.modules.Module = nn.Sequential(
            nn.Linear(d_model, d_model), nn.PReLU(), nn.Linear(d_model, 1)
        )
    return nn.Sequential(layer1, layer2, layer3)


def squim_objective_model(
    feat_dim: int,
    win_len: int,
    d_model: int,
    nhead: int,
    hidden_dim: int,
    num_blocks: int,
    rnn_type: str,
    chunk_size: int,
    chunk_stride: Optional[int] = None,
) -> SquimObjective:
    """Build a custome :class:`torchaudio.prototype.models.SquimObjective` model.

    Args:
        feat_dim (int, optional): The feature dimension after Encoder module.
        win_len (int): Kernel size in the Encoder module.
        d_model (int): The number of expected features in the input.
        nhead (int): Number of heads in the multi-head attention model.
        hidden_dim (int): Hidden dimension in the RNN layer of DPRNN.
        num_blocks (int): Number of DPRNN layers.
        rnn_type (str): Type of RNN in DPRNN. Valid options are ["RNN", "LSTM", "GRU"].
        chunk_size (int): Chunk size of input for DPRNN.
        chunk_stride (int or None, optional): Stride of chunk input for DPRNN.
    """
    if chunk_stride is None:
        chunk_stride = chunk_size // 2
    encoder = Encoder(feat_dim, win_len)
    dprnn = DPRNN(
        feat_dim, hidden_dim, num_blocks, rnn_type, d_model, chunk_size, chunk_stride
    )
    branches = nn.ModuleList(
        [
            _create_branch(d_model, nhead, "stoi"),
            _create_branch(d_model, nhead, "pesq"),
            _create_branch(d_model, nhead, "sisdr"),
        ]
    )
    return SquimObjective(encoder, dprnn, branches)


def squim_objective_base() -> SquimObjective:
    """Build :class:`torchaudio.prototype.models.SquimObjective` model with default arguments."""
    return squim_objective_model(
        feat_dim=256,
        win_len=64,
        d_model=256,
        nhead=4,
        hidden_dim=256,
        num_blocks=2,
        rnn_type="LSTM",
        chunk_size=71,
    )


@dataclass
class SquimObjectiveBundle:
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.models.SquimObjective` model.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    This bundle can estimate objective metric scores for speech enhancement, such as STOI, PESQ, Si-SDR.
    A typical use case would be a flow like `waveform -> list of scores`. Please see below for the code example.

    Example: Estimate the objective metric scores for the input waveform.
        >>> import torch
        >>> import torchaudio
        >>> from torchaudio.pipelines import SQUIM_OBJECTIVE as bundle
        >>>
        >>> # Load the SquimObjective bundle
        >>> model = bundle.get_model()
        Downloading: "https://download.pytorch.org/torchaudio/models/squim_objective_dns2020.pth"
        100%|████████████| 28.2M/28.2M [00:03<00:00, 9.24MB/s]
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Estimate objective metric scores
        >>> scores = model(waveform)
        >>> print(f"STOI: {scores[0].item()},  PESQ: {scores[1].item()}, SI-SDR: {scores[2].item()}.")
    """  # noqa: E501

    _path: str
    _sample_rate: float

    def _get_state_dict(self, dl_kwargs):
        url = f"https://download.pytorch.org/torchaudio/models/{self._path}"
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        print("url:", url)
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        return state_dict

    def get_model(self, *, dl_kwargs=None) -> SquimObjective:
        """Construct the SquimObjective model, and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.models.SquimObjective`.
        """
        model = squim_objective_base()
        # model.load_state_dict(self._get_state_dict(dl_kwargs))
        model.load_state_dict(torch.load("model1.pth"))
        model.eval()
        return model

    @property
    def sample_rate(self):
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate


SQUIM_OBJECTIVE = SquimObjectiveBundle(
    "squim_objective_dns2020.pth",
    _sample_rate=16000,
)
SQUIM_OBJECTIVE.__doc__ = """SquimObjective pipeline trained using approach described in
    :cite:`kumar2023torchaudio` on the *DNS 2020 Dataset* :cite:`reddy2020interspeech`.

    The underlying model is constructed by :py:func:`torchaudio.models.squim_objective_base`.
    The weights are under `Creative Commons Attribution 4.0 International License
    <https://github.com/microsoft/DNS-Challenge/blob/interspeech2020/master/LICENSE>`__.

    Please refer to :py:class:`SquimObjectiveBundle` for usage instructions.
    """


# @dataclass
# class SquimSubjectiveBundle:
#     """Data class that bundles associated information to use pretrained
#     :py:class:`~torchaudio.models.SquimSubjective` model.

#     This class provides interfaces for instantiating the pretrained model along with
#     the information necessary to retrieve pretrained weights and additional data
#     to be used with the model.

#     Torchaudio library instantiates objects of this class, each of which represents
#     a different pretrained model. Client code should access pretrained models via these
#     instances.

#     This bundle can estimate subjective metric scores for speech enhancement, such as MOS.
#     A typical use case would be a flow like `waveform -> score`. Please see below for the code example.

#     Example: Estimate the subjective metric scores for the input waveform.
#         >>> import torch
#         >>> import torchaudio
#         >>> from torchaudio.pipelines import SQUIM_SUBJECTIVE as bundle
#         >>>
#         >>> # Load the SquimSubjective bundle
#         >>> model = bundle.get_model()
#         Downloading: "https://download.pytorch.org/torchaudio/models/squim_subjective_bvcc_daps.pth"
#         100%|████████████| 360M/360M [00:09<00:00, 41.1MB/s]
#         >>>
#         >>> # Resample audio to the expected sampling rate
#         >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
#         >>> # Use a clean reference (doesn't need to be the reference for the waveform) as the second input
#         >>> reference = torchaudio.functional.resample(reference, sample_rate, bundle.sample_rate)
#         >>>
#         >>> # Estimate subjective metric scores
#         >>> score = model(waveform, reference)
#         >>> print(f"MOS: {score}.")
#     """  # noqa: E501

#     _path: str
#     _sample_rate: float

#     def _get_state_dict(self, dl_kwargs):
#         url = f"https://download.pytorch.org/torchaudio/models/{self._path}"
#         dl_kwargs = {} if dl_kwargs is None else dl_kwargs
#         state_dict = load_state_dict_from_url(url, **dl_kwargs)
#         return state_dict

#     def get_model(self, *, dl_kwargs=None) -> SquimSubjective:
#         """Construct the SquimSubjective model, and load the pretrained weight.

#         The weight file is downloaded from the internet and cached with
#         :func:`torch.hub.load_state_dict_from_url`

#         Args:
#             dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

#         Returns:
#             Variation of :py:class:`~torchaudio.models.SquimObjective`.
#         """
#         model = squim_subjective_base()
#         model.load_state_dict(self._get_state_dict(dl_kwargs))
#         model.eval()
#         return model

#     @property
#     def sample_rate(self):
#         """Sample rate of the audio that the model is trained on.

#         :type: float
#         """
#         return self._sample_rate


# SQUIM_SUBJECTIVE = SquimSubjectiveBundle(
#     "squim_subjective_bvcc_daps.pth",
#     _sample_rate=16000,
# )
# SQUIM_SUBJECTIVE.__doc__ = """SquimSubjective pipeline trained
#     as described in :cite:`manocha2022speech` and :cite:`kumar2023torchaudio`
#     on the *BVCC* :cite:`cooper2021voices` and *DAPS* :cite:`mysore2014can` datasets.

#     The underlying model is constructed by :py:func:`torchaudio.models.squim_subjective_base`.
#     The weights are under `Creative Commons Attribution Non Commercial 4.0 International
#     <https://zenodo.org/record/4660670#.ZBtWPOxuerN>`SQUIM_SUBJECTIVE.

#     Please refer to :py:class:`SquimSubjectiveBundle` for usage instructions.
#     """


if __name__ == "__main__":
    objective_model = SQUIM_OBJECTIVE.get_model()
    print(objective_model)
