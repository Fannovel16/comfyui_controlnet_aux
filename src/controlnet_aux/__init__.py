__version__ = "0.0.6"

from .hed import HEDdetector
from .leres import LeresDetector
from .lineart import LineartDetector
from .lineart_anime import LineartAnimeDetector
from .manga_line import LineartMangaDetector
from .midas import MidasDetector
from .mlsd import MLSDdetector
from .normalbae import NormalBaeDetector
from .open_pose import OpenposeDetector
from .pidi import PidiNetDetector
from .zoe import ZoeDetector
from .oneformer import OneformerSegmentor

from .canny import CannyDetector
from .mediapipe_face import MediapipeFaceDetector
from .segment_anything import SamDetector
from .shuffle import ContentShuffleDetector
from .picky_scribble import PickyScribble
from .binary import BinaryDetector
from .dwpose import DwposeDetector
