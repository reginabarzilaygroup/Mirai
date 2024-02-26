import io
import logging
import math
import os
import pdb
import pickle
import tempfile
from typing import List, BinaryIO
import zipfile

import numpy as np
import pydicom
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from onconet import __version__ as onconet_version
from onconet.models.factory import load_model, RegisterModel, get_model_by_name
import onconet.transformers.factory as transformer_factory
from onconet.models.factory import get_model
from onconet.transformers.basic import ComposeTrans
import onconet.utils.dicom
from onconet.utils import parsing

logger = logging.getLogger('mirai_full')


@RegisterModel("mirai_full")
class MiraiFull(nn.Module):

    def __init__(self, args):
        super(MiraiFull, self).__init__()
        self.args = args
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            self.image_encoder = get_model_by_name('custom_resnet', False, args)

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        if args.transformer_snapshot is not None:
            self.transformer = load_model(args.transformer_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = get_model_by_name('transformer', False, args)
        args.img_only_dim = self.transformer.args.transfomer_hidden_dim

    def forward(self, x, risk_factors=None, batch=None):
        B, C, N, H, W = x.size()
        x = x.transpose(1,2).contiguous().view(B*N, C, H, W)
        risk_factors_per_img =  (lambda N, risk_factors: [factor.expand( [N, *factor.size()]).contiguous().view([-1, factor.size()[-1]]).contiguous() for factor in risk_factors])(N, risk_factors) if risk_factors is not None else None
        _, img_x, _ = self.image_encoder(x, risk_factors_per_img, batch)
        img_x = img_x.view(B, N, -1)
        img_x = img_x[:,:,: self.image_repr_dim]
        logit, transformer_hidden, activ_dict = self.transformer(img_x, risk_factors, batch)
        return logit, transformer_hidden, activ_dict


def download_file(url, destination):
    import urllib.request

    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as e:
        logging.getLogger("mirai_full").error(f"An error occurred while downloading from {url} to {destination}: {e}")
        raise e


class MiraiModel:
    """
    Represents a trained Mirai model. Useful for predictions on individual exams.
    """
    def __init__(self, config_obj):
        super().__init__()
        self.args = config_obj
        self.__version__ = onconet_version

    def load_model(self):
        logger.debug("Loading model...")
        self.args.cuda = self.args.cuda and torch.cuda.is_available()

        self.download_if_needed(self.args)
        if self.args.model_name == 'mirai_full':
            model = get_model(self.args)
        else:
            model = torch.load(self.args.snapshot, map_location='cpu')

        # Unpack models that were trained as data parallel
        if isinstance(model, nn.DataParallel):
            model = model.module

        # Add use precomputed hiddens for models trained before it was introduced.
        # Assumes a resnet WHybase backbone
        try:
            model._model.args.use_precomputed_hiddens = self.args.use_precomputed_hiddens
            model._model.args.cuda = self.args.cuda
        except Exception as e:
            logger.debug("Exception caught, skipping precomputed hiddens")
            pass

        return model

    def load_callibrator(self):
        logger.debug("Loading calibrator...")

        # Load callibrator if desired
        if self.args.callibrator_path is not None:
            callibrator = pickle.load(open(self.args.callibrator_path, 'rb'))
        else:
            callibrator = None

        return callibrator

    def process_image_joint(self, batch, model, callibrator, risk_factor_vector=None):
        logger.debug("Getting predictions...")

        # Apply transformers
        x = batch['x']
        risk_factors = autograd.Variable(risk_factor_vector.unsqueeze(0)) if risk_factor_vector is not None else None

        if self.args.cuda:
            x = x.cuda()
            model = model.cuda()
            logger.debug("Inference with GPU")
        else:
            model = model.cpu()
            logger.debug("Inference with CPU")

        # Index 0 to toss batch dimension
        logit, _, _ = model(x, risk_factors, batch)
        probs = F.sigmoid(logit).cpu().data.numpy()
        pred_y = np.zeros(probs.shape[1])

        if callibrator is not None:
            logger.debug("Raw probs: {}".format(probs))

            for i in callibrator.keys():
                pred_y[i] = callibrator[i].predict_proba(probs[0, i].reshape(-1, 1))[0, 1]

        return pred_y.tolist()

    def process_exam(self, images, risk_factor_vector):
        logger.debug("Processing images...")

        test_image_transformers = parsing.parse_transformers(self.args.test_image_transformers)
        test_tensor_transformers = parsing.parse_transformers(self.args.test_tensor_transformers)
        test_transformers = transformer_factory.get_transformers(test_image_transformers, test_tensor_transformers, self.args)
        transforms = ComposeTrans(test_transformers)

        batch = self.collate_batch(images, transforms)
        model = self.load_model()
        callibrator = self.load_callibrator()

        y = self.process_image_joint(batch, model, callibrator, risk_factor_vector)

        return y

    def collate_batch(self, images, transforms):
        logger.debug("Collating batches...")

        batch = {}
        batch['side_seq'] = torch.cat([torch.tensor(b['side_seq']).unsqueeze(0) for b in images], dim=0).unsqueeze(0)
        batch['view_seq'] = torch.cat([torch.tensor(b['view_seq']).unsqueeze(0) for b in images], dim=0).unsqueeze(0)
        batch['time_seq'] = torch.zeros_like(batch['view_seq'])

        batch['x'] = torch.cat(
            (lambda imgs: [transforms(b['x']).unsqueeze(0) for b in imgs])(images), dim=0
        ).unsqueeze(0).transpose(1, 2)

        return batch

    def run_model(self, dicom_files: List[BinaryIO], payload=None):
        if payload is None:
            payload = {
                'dcmtk': True
            }
        elif 'dcmtk' not in payload:
            payload['dcmtk'] = True

        images = []

        if payload['dcmtk']:
            logger.info('Using dcmtk')
        else:
            logger.info('Using pydicom')

        dicom_info = {}

        for dicom in dicom_files:
            try:
                view, side = onconet.utils.dicom.get_dicom_info(pydicom.dcmread(dicom))

                if (view, side) in dicom_info:
                    prev_dicom = dicom_info[(view, side)]
                    prev = int(prev_dicom[0x0008, 0x0023].value + prev_dicom[0x0008, 0x0033].value)
                    cur = int(dicom[0x0008, 0x0023].value + dicom[0x0008, 0x0033].value)

                    if cur > prev:
                        dicom_info[(view, side)] = dicom
                else:
                    dicom_info[(view, side)] = dicom
            except Exception as e:
                logger.warning("{}: {}".format(type(e).__name__, e))

        for k in dicom_info:
            try:
                dicom_path = tempfile.NamedTemporaryFile(suffix='.dcm').name
                image_path = tempfile.NamedTemporaryFile(suffix='.png').name
                logger.debug("Temp DICOM path: {}".format(dicom_path))
                logger.debug("Temp image path: {}".format(image_path))

                dicom = dicom_info[k]
                dicom.seek(0)
                with open(dicom_path, 'wb') as f:
                    f.write(dicom.read())

                view, side = k

                if payload['dcmtk']:
                    image = onconet.utils.dicom.dicom_to_image_dcmtk(dicom_path, image_path)
                    logger.debug('Image mode: {}'.format(image.mode))
                    images.append({'x': image, 'side_seq': side, 'view_seq': view})
                else:
                    dicom = pydicom.dcmread(dicom_path)
                    image = onconet.utils.dicom.dicom_to_arr(dicom, pillow=True)
                    images.append({'x': image, 'side_seq': side, 'view_seq': view})
            except Exception as e:
                logger.warning(f"{type(e).__name__}: {e}")

        risk_factor_vector = None

        y = self.process_exam(images, risk_factor_vector)
        logging.debug(f'Raw Predictions: {y}')

        y = {'Year {}'.format(i+1): round(p, 4) for i, p in enumerate(y)}
        report = {'predictions': y}

        return report

    @staticmethod
    def download_if_needed(args, cache_dir='./.cache'):
        if args.model_name == 'mirai_full':
            if os.path.exists(args.img_encoder_snapshot) and os.path.exists(args.transformer_snapshot):
                return
        else:
            if os.path.exists(args.snapshot):
                return

        if getattr(args, 'remote_snapshot_uri', None) is None:
            return

        logger.info(f"Local models not found, downloading snapshot from remote URI: {args.remote_snapshot_uri}")
        os.makedirs(cache_dir, exist_ok=True)
        tmp_zip_path = os.path.join(cache_dir, "snapshots.zip")
        download_file(args.remote_snapshot_uri, tmp_zip_path)

        dest_dir = os.path.dirname(args.img_encoder_snapshot) if args.model_name == 'mirai_full' else os.path.dirname(args.snapshot)

        # Unzip file
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
