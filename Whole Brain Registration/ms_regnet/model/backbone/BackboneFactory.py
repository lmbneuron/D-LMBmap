from .voxelmorph import VoxelMorph
from .bezier import Bezier


def create_backone(mode):
    if mode == "voxelmorph":
        return VoxelMorph
    elif mode == 'bezier':
        return Bezier
    else:
        raise AssertionError
