# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from nanotron.data.utils import Split, log_single_rank, normalize

logger = logging.getLogger(__name__)


# TODO edit docs
# TODO MOVER
@dataclass
class GPTDatasetConfig:
    """Configuration object for megatron-core blended and megatron datasets
    
    Attributes:
        is_built_on_rank (Callable): A callable which returns True if the dataset should be built
        on the current rank. It should be Megatron Core parallelism aware i.e. global rank, group
        rank, and virtual rank may inform its return value.

        random_seed (int): The seed for all RNG during dataset creation.

        sequence_length (int): The sequence length.

        blend (Optional[List[str]]): The blend string, consisting of either a single dataset or a
        flattened sequential sequence of weight-dataset pairs. For exampe, ["dataset-path1"] and
        ["50", "dataset-path1", "50", "dataset-path2"] are both valid. Not to be used with
        'blend_per_split'. Defaults to None.

        blend_per_split (blend_per_split: Optional[List[Optional[List[str]]]]): A set of blend
        strings, as defined above, one for each split distribution. Not to be used with 'blend'.
        Defauls to None.

        split (Optional[str]): The split string, a comma separated weighting for the dataset splits
        when drawing samples from a single distribution. Not to be used with 'blend_per_split'.
        Defaults to None.

        split_vector: (Optional[List[float]]): The split string, parsed and normalized post-
        initialization. Not to be passed to the constructor.

        path_to_cache (str): Where all re-useable dataset indices are to be cached.
    """

    random_seed: int

    sequence_length: int

    data_path: Optional[List[str]] = None

    split: Optional[str] = None

    split_vector: Optional[List[float]] = field(init=False, default=None)

    # TODO We add the cache because it need its somewhere, take a look later
    path_to_cache: str = None

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization. See
        https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        self.split_vector = _parse_and_normalize_split(self.split)


def _parse_and_normalize_split(split: str) -> List[float]:
    """Parse the dataset split ratios from a string

    Args:
        split (str): The train valid test split string e.g. "99,1,0"

    Returns:
        List[float]: The trian valid test split ratios e.g. [99.0, 1.0, 0.0]
    """
    split = list(map(float, re.findall(r"[.0-9]+", split)))
    split = split + [0.0 for _ in range(len(Split) - len(split))]

    assert len(split) == len(Split)
    assert all(map(lambda _: _ >= 0.0, split))

    split = normalize(split)

    return split
