import logging
import subprocess
from subprocess import Popen
import os
from typing import Iterable

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

from .logging_utils import get_logger

default_window_center = "540"
default_window_width = "580"


def apply_windowing(image, center, width, bit_depth=16, voi_type='LINEAR'):
    """Windowing function to transform image pixels for presentation.
    Must be run after a DICOM modality LUT is applied to the image.
    Windowing algorithm defined in DICOM standard:
    http://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2
    Reference implementation:
    https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L460

    Args:
        image (ndarray): Numpy image array
        center (float): Window center (or level)
        width (float): Window width
        bit_depth (int): Max bit size of pixel
    Returns:
        ndarray: Numpy array of transformed images
    """
    y_min = 0
    y_max = (2**bit_depth - 1)
    y_range = y_max - y_min

    if voi_type == 'LINEAR':
        c = center - 0.5
        w = width - 1.0

        below = image <= (c - w / 2)  # pixels to be set as black
        above = image > (c + w / 2)  # pixels to be set as white
        between = np.logical_and(~below, ~above)

        image[below] = y_min
        image[above] = y_max

        if between.any():
            image[between] = (
                    ((image[between] - c) / w + 0.5) * y_range + y_min
            )
    elif voi_type == 'SIGMOID':
        image = y_range / (1 + np.exp(-4 * (image - center) / width)) + y_min
    else:
        raise ValueError("Invalid/Unknown VOI LUT type: {}".format(voi_type))

    return image


def is_dcmtk_installed():
    try:
        result = subprocess.check_output(["dcmj2pnm"], stderr=subprocess.STDOUT)
        return "Convert DICOM".lower() in result.decode('utf-8').lower()
    except subprocess.CalledProcessError as e:
        return False
    except FileNotFoundError:
        return False


def dicom_to_image_dcmtk(dicom_path, image_path, pillow=False):
    """Converts a dicom image to a grayscale 16-bit png image using dcmtk.

    Convert DICOM to PNG using dcmj2pnm (support.dcmtk.org/docs/dcmj2pnm.html)
    from dcmtk library (dicom.offis.de/dcmtk.php.en)

    Arguments:
        dicom_path(str): The path to the dicom file.
        image_path(str): The path where the image will be saved.
    """
    logger = get_logger()

    dcm_file = pydicom.dcmread(dicom_path)
    manufacturer = getattr(dcm_file, 'Manufacturer', 'Unknown Manufacturer')
    voi_lut_exists = (0x0028, 0x3010) in dcm_file and len(dcm_file[(0x0028, 0x3010)].value) > 0

    # SeriesDescription is not a required attribute, see
    #   https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.3.html#table_C.7-5a
    if hasattr(dcm_file, 'SeriesDescription'):
        ser_desc = dcm_file.SeriesDescription
    else:
        ser_desc = ''

    # https://support.dcmtk.org/docs/dcmj2pnm.html
    if 'GE' in manufacturer and voi_lut_exists:
        logger.debug("Manufacturer is GE and VOI LUT exists, using VOI LUT")
        args = ['dcmj2pnm', '+on2', '--use-voi-lut', '1', '--grayscale', dicom_path, image_path]
    elif 'C-View' in ser_desc and voi_lut_exists:
        logger.debug("SeriesDescription contains C-View and VOI LUT exists, using VOI LUT")
        args = ['dcmj2pnm', '+on2', '--grayscale', '+Ww', default_window_center, default_window_width, dicom_path, image_path]
    else:
        logger.debug("Manufacturer not GE or C-View/VOI LUT doesn't exist, defaulting to min-max window algorithm")
        args = ['dcmj2pnm', '+on2', '--min-max-window', '--grayscale', dicom_path, image_path]

    level_name = logging.getLevelName(logger.level).lower().replace("warning", "warn")
    if level_name in {"fatal", "error", "warn", "info", "debug", "trace"}:
        args += ['--log-level', level_name]
    args = list(map(str, args))

    logger.debug(f"Running command: {' '.join(args)}")
    output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if output.stderr:
        logger.debug(output.stderr.decode('utf-8'))

    image = Image.open(image_path).convert('I')
    if pillow:
        image = np.array(image).astype(np.int32)
        if image.shape[-1] in {3, 4}:
            image = image.mean(axis=-1, dtype=np.int32)
        return Image.fromarray(image, mode='I')
    else:
        return image


def dicom_to_arr(dicom, window_method='minmax', index=0, pillow=False, overlay=False):
    logger = get_logger()
    image = apply_modality_lut(dicom.pixel_array, dicom)

    if (0x0028, 0x1056) in dicom:
        voi_type = dicom[0x0028, 0x1056].value
    else:
        voi_type = 'LINEAR'

    manufacturer = getattr(dicom, 'Manufacturer', 'Unknown Manufacturer')
    if 'GE' in manufacturer and (0x0028, 0x3010) in dicom:
        logger.debug('GE dicom_to_arr conversion')
        image = apply_voi_lut(image.astype(np.uint16), dicom, index=index)
        num_bits = dicom[0x0028, 0x3010].value[index][0x0028, 0x3002].value[2]
        image *= 2**(16 - num_bits)
    elif window_method == 'auto':
        logger.debug('auto dicom_to_arr conversion')
        window_center = -600
        window_width = 1500
        # Use the window center and width from the DICOM header if available
        if (0x0028, 0x1050) in dicom:
            window_center = dicom[0x0028, 0x1050].value
            window_width = dicom[0x0028, 0x1051].value

        logger.debug(f"auto window center: {window_center}, window width: {window_width}")

        image = apply_windowing(image, window_center, window_width, voi_type=voi_type)
    elif window_method == 'minmax':
        logger.debug('minmax dicom_to_arr conversion')
        min_pixel = np.min(image)
        max_pixel = np.max(image)
        window_center = (min_pixel + max_pixel + 1) / 2
        window_width = max_pixel - min_pixel + 1
        logger.debug(f"minmax window center: {window_center}, window width: {window_width}")

        image = apply_windowing(image, window_center, window_width, voi_type=voi_type)
    else:
        raise ValueError(f"Invalid window_method: {window_method}")

    image = image.astype(np.uint16)

    if overlay:
        arr = dicom.overlay_array(0x6000)
        old_shape = arr.shape
        arr = arr.flatten()

        arr = np.append(arr, np.array([0] * 4))
        arr = arr.reshape((len(arr) // 16, 16))

        for i in range(arr.shape[0]):
            if 1 not in arr[i]:
                continue
            arr[i] = np.roll(arr[i], 8)

        arr = arr.flatten()

        arr = arr[:-4].reshape(old_shape)
        num_bits = dicom[0x0028, 0x3010].value[index][0x0028, 0x3002].value[2]
        image[arr == 1] = 2 ** 16 - 1

    if pillow:
        image = image.astype(np.int32)
        if image.shape[-1] in {3, 4}:
            image = image.mean(axis=-1, dtype=np.int32)
        return Image.fromarray(image, mode='I')
    else:
        return image


def get_dicom_info(dicom: pydicom.Dataset):
    """Return tags for View Position and Image Laterality.

    Args:
        dicom (pydicom.Dataset): Dataset object containing DICOM tags

    Returns:
        int: binary integer 0 or 1 corresponding to the type of View Position
        int: binary integer 0 or 1 corresponding to the type of Image Laterality
    """

    # Some cases (FUJIFILM) have cases where view position is in
    # Acquisition Device Processing Description (0018, 1400)
    # rather than standard DICOM tag
    if not hasattr(dicom, 'ViewPosition'):
        if 'CC' in dicom[0x0018, 0x1400].value:  # Acquisition Device Processing Description
            view_str = 'CC'
        elif 'MLO' in dicom[0x0018, 0x1400].value:
            view_str = 'MLO'
        else:
            raise AttributeError('ViewPosition does not exist in DICOM metadata')
    else:
        view_str = dicom.ViewPosition

    # Have seen cases where ImageLaterality is not present in DICOM metadata,
    # and the relevant information is in the ViewPosition tag. Check for this.
    if not hasattr(dicom, 'ImageLaterality'):
        if "RIGHT" in view_str.upper():
            side_str = 'R'
        elif "LEFT" in view_str.upper():
            side_str = 'L'
        else:
            raise AttributeError('ImageLaterality does not exist in DICOM metadata')
    else:
        side_str = dicom.ImageLaterality

    view_str = view_str.upper().replace("RIGHT", "").replace("LEFT", "").strip()

    valid_view = ['CC', 'MLO', 'ML']
    valid_side = ['R', 'L']

    if view_str not in valid_view:
        raise ValueError("Invalid View Position `{}`: must be in {}".format(view_str, valid_view))
    if side_str not in valid_side:
        raise ValueError("Invalid Image Laterality `{}`: must be in {}".format(side_str, valid_side))

    view = 0 if view_str == 'CC' else 1
    side = 0 if side_str == 'R' else 1

    return view, side
