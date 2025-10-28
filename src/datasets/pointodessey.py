from pathlib import Path
from typing import Optional, Literal, Any

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from src import register
from src.tools import ops, viz
from src.typing import Axes, BatchData
from src.utils import io
from . import BaseDataset

__all__ = ["PointOdesseyDataset"]

def get_images(root_path: Path, mode: Literal['train', 'val', 'test', 'sample']) -> list[tuple[Path, Path]]:
    scenes: list[tuple[Path, Path]] = []
    for scene in (root_path / mode).glob("*/"):
        # open info.npz and get number of the frames in the scene
        info_file = np.load(scene / 'info.npz')
        num_scenes: int = info_file['valids'][0]
        # TODO: read other info reg. disparity?
        # we do not read all files directly as this is cheaper on the fs
        scenes.extend([
            (scene / 'rgbs' / f'rgb_{idx:05d}.jpg',
             scene / 'depths' / f'depth_{idx:05d}.png') for idx in range(num_scenes)
        ])

    return scenes

@register("point_odessey")
class PointOdesseyDataset(BaseDataset):
    """
    PointOdessey dataset based on https://arxiv.org/abs/2307.15055.

    Attributes:
    :param mode: (str) split mode to load. {train, val, test, sample}
    :param size: (Sequence[int]) Target image training size as (w, h).
    :param data_root: (Path) Path to root directory of data
    :param as_torch: (bool) If `True`, convert (x, y, meta) to torch.
    :param use_aug: (bool) If `True`, call 'self.augment' during __getitem__.
    :param log_time: (bool) If `True`, log time taken to load/augment each item.
    """
    def __init__(
            self,
            mode: Literal['train', 'val', 'test', 'sample'],
            size: tuple[int, int] = (924, 518),
            data_root: Path = Path('/scratch/mde/pointodessey'),
            **kwargs: dict[Any, Any]
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.w, self.h = self.size = size
        if not (mode == 'train') and self.use_aug:
            raise ValueError('using augmentation while not training is not recommended!')

        self.imgs = get_images(data_root, mode)


    def __len__(self) -> int:
        return len(self.imgs)

    def load(self, item: int, x: dict, y: dict, m: dict) -> BatchData:
        """
        Load single item in dataset.

        NOTE: Items in each dict will be converted into `torch.Tensor`s if `self.as_torch=True`.

        :param item: (int) Dataset item to load.
        :param x: {
            imgs: (ndarray) (h, w, 3) Target image for depth estimation.
        }
        :param y: {
            images: (ndarray) (h, w, 3) x['imgs'] (NO AUGMENTATIONS)
            depth: (ndarray) (h, w, 1) Ground truth depth
        }
        :param m:
            items: (str) Loaded dataset item
            path: (tuple[Path, Path]) Loaded dataset item path
            aug (str): Augmentations applied to current item.
            errors: (List[str]): List of errors when loading previous items.
            data_timer (MultiLevelTimer): Timing information for current item.
        """

        img, depth = self.imgs[item]
        # m['path'] = img

        with self.timer('Image'):
            img, img_res = self.load_image(img)
            x['imgs'] = np.array(img_res, dtype=np.float32)
            y['imgs'] = np.array(img_res, dtype=np.float32)

        with self.timer('Depth'):
            y['depth'] = np.array(self.load_depth(depth), dtype=np.float32)
        
        return x, y, m

    def load_image(self, data: Path) -> tuple[Image.Image, Image.Image]:
        img = Image.open(data).convert('RGB')
        img_res = img.resize(self.size, resample=Image.BILINEAR)
        return img, img_res

    def load_depth(self, data: Path):
        img = Image.open(data)
        return img

    def load_edges(self, data: tuple[str, str]) -> Image:
        """Load a single depth edge map.

        :param data: (str, str) Data representing the item's scene and file number.
        :return: (Image) (self.full_w, self.full_h) Loaded PIL edge map.
        """
        return ValueError("no edges for pointodessey!")

    def transform(self, x: dict, y: dict, m: dict) -> BatchData:
        """Apply ImageNet standarization to the images processed by the network `x`."""
        x['imgs'] = ops.standardize(x['imgs'])
        return x, y, m

    def create_axs(self) -> Axes:
        """Create the axis structure required for plotting."""
        _, axs = plt.subplots(1 + self.use_depth + (self.edges_dir is not None))
        if isinstance(axs, plt.Axes): axs = np.array([axs])
        plt.tight_layout()
        return axs

    def show(self, x: dict, y: dict, m: dict, axs: Optional[Axes] = None) -> None:
        """Show a single dataset item."""
        axs = self.create_axs() if axs is None else axs
