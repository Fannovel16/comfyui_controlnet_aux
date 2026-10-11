"""Regression test for issue #631.

``LPIPS.load_from_pretrained()`` used to pass a CWD-relative root
(``"taming/modules/autoencoder/lpips"``) to ``get_ckpt_path()``, so every
Diffusion Edge model load downloaded ``vgg.pth`` from heibox.uni-heidelberg.de
into the current working directory -- raising and leaving a stray directory
behind when that host is unreachable -- instead of using the identical file
bundled in the source tree at
``src/custom_controlnet_aux/diffusion_edge/taming/modules/autoencoder/lpips/vgg.pth``.

The harness below loads the real ``losses/lpips.py`` (whose relative import
resolves to the real ``losses/util.py``) via importlib; only
``custom_controlnet_aux.util.custom_torch_download`` is stubbed out, to avoid
pulling in the heavy ``custom_controlnet_aux.util`` dependency chain. It is
the VGG16 backbone downloader and is never called on the path under test,
which is why these tests run fully offline.

Written in unittest style so it can run either as
``python -m unittest tests.test_lpips_bundled_ckpt`` or under pytest.
"""

import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

SRC = Path(__file__).resolve().parents[1] / "src"
LOSSES = SRC / "custom_controlnet_aux" / "diffusion_edge" / "taming" / "modules" / "losses"
LPIPS_PY = LOSSES / "lpips.py"
BUNDLED = LOSSES.parent / "autoencoder" / "lpips" / "vgg.pth"
PKG = "custom_controlnet_aux.diffusion_edge.taming.modules"


def _load_losses_util():
    spec = importlib.util.spec_from_file_location("lpips_regression_losses_util", LOSSES / "util.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_lpips_module():
    """Load the real losses/lpips.py under its real package name so its
    relative ``from .util import get_ckpt_path`` resolves to the real
    losses/util.py. Returns the module; the caller must restore sys.modules
    (see TestLpipsUsesBundledCkpt.tearDown)."""
    register = types.ModuleType
    ca = register("custom_controlnet_aux")
    ca.__path__ = []
    cu = register("custom_controlnet_aux.util")
    cu.custom_torch_download = lambda filename: str(BUNDLED.parent / filename)
    sys.modules["custom_controlnet_aux"] = ca
    sys.modules["custom_controlnet_aux.util"] = cu

    name = "custom_controlnet_aux"
    for sub, d in [
        ("diffusion_edge", SRC / "custom_controlnet_aux/diffusion_edge"),
        ("taming", SRC / "custom_controlnet_aux/diffusion_edge/taming"),
        ("modules", SRC / "custom_controlnet_aux/diffusion_edge/taming/modules"),
        ("losses", LOSSES),
    ]:
        name = f"{name}.{sub}"
        m = register(name)
        m.__path__ = [str(d)]
        sys.modules[name] = m

    mod = register(PKG + ".losses.lpips")
    mod.__file__, mod.__package__ = str(LPIPS_PY), PKG + ".losses"
    sys.modules[mod.__name__] = mod
    exec(compile(LPIPS_PY.read_text(), str(LPIPS_PY), "exec"), mod.__dict__)
    return mod


@unittest.skipUnless(torch is not None, "torch is required")
class TestLpipsUsesBundledCkpt(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {k: sys.modules[k] for k in list(sys.modules) if k.startswith("custom_controlnet_aux")}
        self._prev_cwd = os.getcwd()
        self.cwd = tempfile.mkdtemp(prefix="lpips631-cwd-")
        os.chdir(self.cwd)
        self.lpips = _load_lpips_module()

    def tearDown(self):
        os.chdir(self._prev_cwd)
        shutil.rmtree(self.cwd, ignore_errors=True)
        for k in [k for k in list(sys.modules) if k.startswith("custom_controlnet_aux")]:
            del sys.modules[k]
        sys.modules.update(self._saved_modules)

    def test_load_from_pretrained_uses_bundled_file(self):
        """load_from_pretrained() must load the bundled vgg.pth and leave the
        CWD untouched (no heibox download, no stray taming/ directory)."""
        self.assertTrue(BUNDLED.exists(), f"bundled checkpoint missing: {BUNDLED}")
        holder = torch.nn.Module()  # bare module: exercises only load_from_pretrained
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.lpips.LPIPS.load_from_pretrained(holder)
        self.assertIn(f"loaded pretrained LPIPS loss from {BUNDLED}", buf.getvalue())
        self.assertEqual(sorted(str(p) for p in Path(self.cwd).rglob("*")), [],
                         "CWD was polluted by a checkpoint download")

    def test_bundled_vgg_pth_md5_matches_md5_map(self):
        """The bundled file is the exact artifact MD5_MAP['vgg_lpips'] pins."""
        self.assertTrue(BUNDLED.exists(), f"bundled checkpoint missing: {BUNDLED}")
        util = _load_losses_util()
        self.assertEqual(util.md5_hash(str(BUNDLED)), util.MD5_MAP["vgg_lpips"])


if __name__ == "__main__":
    unittest.main()
