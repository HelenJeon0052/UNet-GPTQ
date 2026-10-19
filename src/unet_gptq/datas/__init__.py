from .fastmri_dataset import fft2c, fft2_unshifted, ifft2c, ifft_unshifted, image_to_4ch, _crop_shape, FastMRISlicePairs
from .fastmri_plus import boxes_to_roi_mask
# These optional local utilities are not shipped in the public repository.
# Resolve them only on explicit access, so importing the data/reconstruction
# package does not depend on unpublished inspection or plotting modules.
def __getattr__(name):
    from importlib import import_module
    modules = {"inspect_pair": ".inspect_fastmri", "arranged_img": ".visualize_input"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

__all__ = ["fft2c", "fft2_unshifted", "ifft2c", "ifft_unshifted", "image_to_4ch", "_crop_shape", "FastMRISlicePairs", "boxes_to_roi_mask", "inspect_pair", "arranged_img"]
__version__ = '0.1.0'
