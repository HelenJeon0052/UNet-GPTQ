from .fastmri_dataset import fft2c, fft2_unshifted, ifft2c, ifft_unshifted, image_to_4ch, _crop_shape, FastMRISlicePairs
from .fastmri_plus import boxes_to_roi_mask


def __getattr__(name):
    """Load optional inspection and plotting utilities only when requested."""
    from importlib import import_module

    locations = {
        "inspect_pair": ("unet_gptq.train.inspect_fastmri", "inspect_pair"),
        "arranged_img": ("unet_gptq.datas.visualize_input", "arranged_img"),
    }
    if name not in locations:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = locations[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value

__all__ = ["fft2c", "fft2_unshifted", "ifft2c", "ifft_unshifted", "image_to_4ch", "_crop_shape", "FastMRISlicePairs", "boxes_to_roi_mask", "inspect_pair", "arranged_img"]
__version__ = '0.1.0'
