"""Script to export network predictions on a target dataset."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Union
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader
import transformers
import tqdm
import torch

from src import MonoDepthModule, MonoDepthEvaluator, predict_depth_single
from src.paths import find_model_file
from src.tools import ops, parsers
from src.utils.io import load_yaml, write_yaml
"""Script to evaluate network predictions on a target dataset."""

from src.typing import Metrics

def save_metrics(file: Path, metrics: Sequence[Metrics]):
    """Helper to save metrics. If any strings are present, save metrics separately. Otherwise save means."""
    print(f'\n-> Saving results to "{file}"...')
    file.parent.mkdir(exist_ok=True, parents=True)
    use_mean = all((isinstance(v, float) for v in metrics[0].values()))
    if use_mean: metrics = {k: float(np.array([m[k] for m in metrics]).mean()) for k in metrics[0]}
    write_yaml(file, metrics, mkdir=True)    

def compute_eval_metrics(preds: NDArray, mode: str, cfg_file: Path) -> Sequence[Metrics]:
    """Compute evaluation metrics from network predictions.
    Predictions must be unscaled (see `compute_eval_preds`).

    :param preds: (NDArray) (b, h, w) Precomputed unscaled network predictions.
    :param mode: (str) Evaluation mode, which determines prediction scaling. {stereo, mono}
    :param cfg_file: (Path) Path to YAML config file.
    :return: (list[Metrics]) Metrics computed for each dataset item.
    """
    cfg = load_yaml(cfg_file)
    cfg_ds, cfg_args = cfg['dataset'], cfg['args']

    target_stem = cfg_ds.pop('target_stem', f'targets_{cfg.get("mode", "test")}')
    ds = parsers.get_ds(cfg_ds)
    target_file = ds.split_file.parent/f'{target_stem}.npz'
    print(f'\n-> Loading targets from "{target_file}"...')
    data = np.load(target_file, allow_pickle=True)

    evaluator = MonoDepthEvaluator(mode=mode, **cfg_args)
    metrics = evaluator.run(preds, data)
    return metrics

@torch.no_grad()
def compute_eval_preds(ckpt_file: Union[str, Path], cfg: dict, mode: str, overwrite: bool = True) -> list[Metrics]:
    """Compute network predictions required for evaluation.

    The confing in `cfg_dataset` is equivalent to that used by the `Trainer`.
    Note that in most cases, additional outputs, such as depth or edges can be omitted.
    Furthermore, image `size` is determined by the pretrained checkpoint.

    The config stored in `ckpt_file` is used to automatically determine:
        - Image size for network input.
        - Initial disparity scaling range.

    NOTE: The output disparities are NOT in metric depth. They are just scaled to the range expected by the network
    during training. We still need to apply fixed scaling (stereo) or median scaling (mono). This is done in the
    evaluation script by the `DepthEvaluator`.

    :param ckpt_file: (Path) Path to pretrained model checkpoint. Path can be absolute or relative to `MODEL_ROOTS`.
    :param cfg: (dict) Loaded YAML dataset config.
    :return: (ndarray) (b, h, w) Array containing unscaled network predictions for each dataset item.
    """
    device = ops.get_device()

    # ckpt_file = find_model_file(ckpt_file)
    # if not (ckpt_file.parent/'finished').is_file() and not overwrite:
    #     print(f'-> Training for "{ckpt_file}" has not finished...')
    #     print('-> Set `--overwrite 1` to run this evaluation anyway...')
    #     exit()

    # hparams_file = str(ckpt_file.parents[1] / 'hparams.yaml')
    # print(f'\n-> Loading model weights from "{ckpt_file}"...')
    # mod = MonoDepthModule.load_from_checkpoint(ckpt_file, hparams_file=hparams_file, strict=False).eval()
    # mod.freeze()

    cfg.update({
        # 'size': mod.cfg['dataset']['size'],
        'as_torch': True,
        'use_aug': False,
        'log_time': False,
    })
    ds = parsers.get_ds(cfg)
    dl = DataLoader(ds, batch_size=12, num_workers=4, collate_fn=ds.collate_fn, pin_memory=True)
    cfg_args: dict = cfg.get('args', {})

    # model = mod.nets['depth'].to(device)

    image_processor = transformers.AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", use_fast=True, ensure_multiple_of=14)
    model = transformers.AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", device_map='auto')
    evaluator = MonoDepthEvaluator(mode=mode, **cfg_args)

    metrics = []
    for data in tqdm.tqdm(dl):
        x, y, m = data
        img = x['imgs'].to(device)
        # inputs = image_processor(images=img, return_tensors='pt')
        preds = model(pixel_values=img).predicted_depth
        preds = preds.cpu().numpy()
        y = {k: v.cpu().numpy() for k, v in y.items()}
        for e, i in enumerate(preds):
            i = (i - i.min()) / (i.max() - i.min())
            x = Image.fromarray(np.uint8(255 * i))
            x.save(f'artifacts/preds_{e:04}.png')
        metrics.extend(evaluator.run(preds, y, pred_type='disparity'))
        print(metrics)
        break
    return metrics

if __name__ == '__main__':
    parser = ArgumentParser(description='Script to evaluate network predictions on a target dataset.')
    parser.add_argument('--mode', required=True, choices={'stereo', 'mono'}, help='Evaluation mode, determining the prediction scaling.')
    parser.add_argument('--ckpt-file', default=None, type=Path, help='Optional path to model ckpt to compute predictions.')
    parser.add_argument('--cfg-file', required=True, type=Path, help='Path to YAML eval config.')
    parser.add_argument('--save-file', default=None, type=Path, help='Path to YAML file to save evaluation metrics.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing metrics files.')
    args = parser.parse_args()

    if args.save_file and args.save_file.is_file() and not args.overwrite:
        print(f'-> Evaluation file already exists "{args.save_file}"...')
        print('-> Set `--overwrite 1` to run this evaluation anyway...')
        exit()

    if not args.ckpt_file:
        raise ValueError('Must provide either a `--pred-file` with precomputed predictions '
                         'or a `--ckpt-file to compute predictions from!')
    cfg = load_yaml(args.cfg_file)['dataset']
    # cfg.pop('target_stem')
    preds = compute_eval_preds(args.ckpt_file, cfg, args.overwrite)

    metrics = compute_eval_metrics(preds, args.mode, args.cfg_file)
    if args.save_file: save_metrics(args.save_file, metrics)
