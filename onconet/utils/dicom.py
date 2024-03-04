import logging
from collections import Iterable
from subprocess import Popen

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

logger = logging.getLogger('onconet.utils.dicom')


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

    return image


def read_dicoms(dicom_list, limit=None):
    """Reads in a list DICOM files or file paths.

    Args:
        dicom_list (Iterable): List of file objects or file paths
        limit (int, optional): Limit number of dicoms to be read

    Returns:
        list: List of pydicom Datasets
    """
    dicoms = []
    for f in dicom_list:
        try:
            dicom = pydicom.dcmread(f)
        except Exception as e:
            logger.warning(e)
            continue

        dicoms.append(dicom)

        if limit is not None and len(dicoms) >= limit:
            logger.debug("Limit of DICOM input reached: {}".format(limit))
            break

    return dicoms


def dicom_to_image_dcmtk(dicom_path, image_path):
    """Converts a dicom image to a grayscale 16-bit png image using dcmtk.

    Convert DICOM to PNG using dcmj2pnm (support.dcmtk.org/docs/dcmj2pnm.html)
    from dcmtk library (dicom.offis.de/dcmtk.php.en)

    Arguments:
        dicom_path(str): The path to the dicom file.
        image_path(str): The path where the image will be saved.
    """
    default_window_level = "540"
    default_window_width = "580"

    dcm_file = pydicom.dcmread(dicom_path)
    manufacturer = dcm_file.Manufacturer
    voi_lut_exists = (0x0028, 0x3010) in dcm_file and len(dcm_file[(0x0028, 0x3010)].value) > 0

    # SeriesDescription is not a required attribute, see
    #   https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.3.html#table_C.7-5a
    if hasattr(dcm_file, 'SeriesDescription'):
        ser_desc = dcm_file.SeriesDescription
    else:
        ser_desc = ''

    if 'GE' in manufacturer and voi_lut_exists:
        Popen(['dcmj2pnm', '+on2', '--use-voi-lut', '1', dicom_path, image_path]).wait()
    elif 'C-View' in ser_desc and voi_lut_exists:
        Popen(['dcmj2pnm', '+on2', '+Ww', default_window_level, default_window_width, dicom_path, image_path]).wait()
    else:
        logger.warning("Manufacturer not GE or C-View/VOI LUT doesn't exist, defaulting to min-max window algorithm")
        Popen(['dcmj2pnm', '+on2', '--min-max-window', dicom_path, image_path]).wait()

    return Image.open(image_path)


def dicom_to_arr(dicom, auto=True, index=0, pillow=False, overlay=False):
    image = apply_modality_lut(dicom.pixel_array, dicom)

    if (0x0028, 0x1056) in dicom:
        voi_type = dicom[0x0028, 0x1056].value
    else:
        voi_type = 'LINEAR'

    if 'GE' in dicom.Manufacturer:
        logger.debug('GE dicom_to_arr conversion')
        image = apply_voi_lut(image.astype(np.uint16), dicom, index=index)

        num_bits = dicom[0x0028, 0x3010].value[index][0x0028, 0x3002].value[2]
        image *= 2**(16 - num_bits)
    elif auto:
        logger.debug('auto dicom_to_arr conversion')
        window_center = -600
        window_width = 1500

        image = apply_windowing(image, window_center, window_width, voi_type=voi_type)
    else:
        logger.debug('minmax')
        min_pixel = np.min(image)
        max_pixel = np.max(image)
        window_center = (min_pixel + max_pixel + 1) / 2
        window_width = max_pixel - min_pixel + 1

        image = apply_windowing(image, window_center, window_width, voi_type=voi_type)

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
        return Image.fromarray(image.astype(np.int32), mode='I')
    else:
        return image


def get_dicom_info(dicom):
    """Return tags for View Position and Image Laterality.
    # TODO: This may be Mirai specific, move as needed.

    Args:
        dicom (pydicom.Dataset): Dataset object containing DICOM tags

    Returns:
        int: binary integer 0 or 1 corresponding to the type of View Position
        int: binary integer 0 or 1 corresponding to the type of Image Laterality
    """
    if not hasattr(dicom, 'ViewPosition'):
        raise AttributeError('ViewPosition does not exist in DICOM metadata')
    if not hasattr(dicom, 'ImageLaterality'):
        raise AttributeError('ImageLaterality does not exist in DICOM metadata')

    view_str = dicom.ViewPosition
    side_str = dicom.ImageLaterality

    valid_view = ['CC', 'MLO']
    valid_side = ['R', 'L']

    if view_str not in valid_view:
        raise ValueError("Invalid View Position `{}`: must be in {}".format(view_str, valid_view))
    if side_str not in valid_side:
        raise ValueError("Invalid Image Laterality `{}`: must be in {}".format(side_str, valid_side))

    view = 0 if view_str == 'CC' else 1
    side = 0 if side_str == 'R' else 1

    return view, side
