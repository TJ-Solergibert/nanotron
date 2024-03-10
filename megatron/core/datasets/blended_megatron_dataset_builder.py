# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from typing import Any, List, Optional, Tuple, Type, Union

import numpy
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split, normalize

logger = logging.getLogger(__name__)

DistributedDataset = Union[BlendedDataset, MegatronDataset, MMapIndexedDataset]


class BlendedMegatronDatasetBuilder(object):
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset

        sizes (List[int]): The minimum number of total samples to draw from each split, varies
        with data_path

        config (GPTDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self, cls: Type[MegatronDataset], sizes: List[int], config: GPTDatasetConfig,
    ):
        self.cls = cls
        self.sizes = sizes
        self.config = config

    def build(self) -> List[Optional[Union[BlendedDataset, MegatronDataset]]]:
        """Build all dataset splits according to the provided data_path(s)
        
        This method is distributed-aware and must be called on all ranks.
        
        The dataset splits returned can vary according to the config. Supply config.data_path and
        config.split to build BlendedDataset and/or MegatronDataset splits from the same
        distribution. Supply config.data_path_per_split to build BlendedDataset and/or MegatronDataset
        splits from separate distributions.

        Returns:
            List[Optional[Union[BlendedDataset, MegatronDataset]]]: A list of either
            MegatronDataset or BlendedDataset (or None) per split
        """
        return self._build_blended_dataset_splits()

    def _build_blended_dataset_splits(
        self,
    ) -> List[Optional[Union[BlendedDataset, MegatronDataset]]]:
        """Build all dataset splits according to the provided data_path(s)
        
        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[Union[BlendedDataset, MegatronDataset]]]: A list of either
            MegatronDataset or BlendedDataset (or None) per split
        """

        # TODO Remove this function and move to init? or at least to build?
        data_path = getattr(self.config, "data_path")
        split = getattr(self.config, "split_vector")

        # data_path consists of a single prefix
        # TODO: Refer to Megatron to check how to work with multiple data_paths datasets. For now we don't support it
        data_path = [data_path]
        if len(data_path) == 1:
            return self._build_megatron_dataset_splits(data_path[0], split, self.sizes)

    def _build_megatron_dataset_splits(
        self, path_prefix: str, split: List[float], sizes: List[int],
    ) -> List[Optional[MegatronDataset]]:
        """Build each MegatronDataset split from a single MMapIndexedDataset

        Args:
            path_prefix (str): The MMapIndexedDataset .bin and .idx file prefix

            split (List[float]): The dataset split ratios (must sum to 1.00)

            sizes (List[int]): The number of total samples to draw from each split

        Returns:
            List[Optional[MegatronDataset]]: The MegatronDatset (or None) per split
        """
        # NOTE self.cls.is_multimodal() in GPTDataset always False
        indexed_dataset = self._build_generic_dataset(
            MMapIndexedDataset, path_prefix, False
        )

        # TODO Refractor this if because in Nanotron all processes create the dataset so its always True
        if indexed_dataset is not None:
            # NOTE if self.cls.is_split_by_sequence(): in GPTDataset always True
            split_idx_bounds = _get_split_indices(
                split, indexed_dataset.sequence_lengths.shape[0]
            )
            
            split_indices = [
                numpy.arange(
                    start=split_idx_bounds[i],
                    stop=split_idx_bounds[i + 1],
                    step=1,
                    dtype=numpy.int32,
                )
                for i, _ in enumerate(Split)
            ]
        # TODO refer to L97 TODO
        else:
            split_indices = [None for _ in Split]

        megatron_datasets = []
        for i, _split in enumerate(Split):
            if split[i] == 0.0:
                megatron_datasets.append(None)
            else:
                megatron_datasets.append(
                    self._build_generic_dataset(
                        self.cls, indexed_dataset, split_indices[i], sizes[i], _split, self.config
                    )
                )

        return megatron_datasets

    def _build_generic_dataset(
        self, cls: Type[DistributedDataset], *args: Any,
    ) -> Optional[DistributedDataset]:
        """Build the DistributedDataset

        Return None if and only if the underlying MegatronDataset class is not built on the current
        rank and torch.distributed is initialized.

        Args:
            cls (Type[DistributedDataset]): The DistributedDataset class to be built

            args (Tuple[Any]): The positional arguments used to build the provided
            DistributedDataset class

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            Optional[DistributedDataset]: The DistributedDataset instantion or None
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

            dataset = None

            # First, build on rank 0
            if rank == 0:
                try:
                    dataset = cls(*args)
                except OSError as err:
                    log = (
                        f"Failed to write dataset materials to the data cache directory. "
                        + f"Please supply a directory to which you have write access via "
                        + f"the path_to_cache attribute in GPTDatasetConfig and "
                        + f"retry. Refer to the preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            torch.distributed.barrier()

            # After, build on other ranks
            # TODO Change for else
            if rank != 0:
                dataset = cls(*args)

            return dataset

        return cls(*args)


def _get_split_indices(split: List[float], num_elements: int) -> List[int]:
    """Determine the document index bounds per split

    Args:
        split (List[float]): The dataset split ratios (must sum to 1.00)

        num_elements (int): The number of elements, e.g. sequences or documents, available for
        the split

    Returns:
        List[int]: The indices for all three splits e.g. [0, 900, 990, 1000] for a 1000-document
        set and a [90.0, 9.0, 1.0] split
    """
    split_indices = [0]
    for split_pct in split:
        split_indices.append(split_indices[-1] + int(round(split_pct * float(num_elements))))
    split_indices[1:] = list(
        map(lambda _: _ - (split_indices[-1] - num_elements), split_indices[1:])
    )

    assert len(split_indices) == len(split) + 1
    assert split_indices[-1] == num_elements

    return split_indices


def _get_prefixes_weights_and_sizes_for_data_path(
    data_path: List[str], target_num_samples_per_split: List[int]
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits
    
    Args:
        data_path (List[str]): e.g. ["30", "path/to/dataset_1_prefix", "70", 
        "path/to/dataset_2_prefix"]

        target_num_samples_per_split (List[int]): The number of samples to target for each
        BlendedDataset split

    Returns:
        Tuple[List[str], List[float], List[List[int]]]: The prefix strings e.g.
        ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], the normalized weights e.g.
        [0.3, 0.7], and the number of samples to request per MegatronDataset per split
    """
    weights, prefixes = zip(
        *[(float(data_path[i]), data_path[i + 1].strip()) for i in range(0, len(data_path), 2)]
    )

    weights = normalize(weights)

    # Use 0.5% target margin to ensure we satiate the network
    sizes_per_dataset = [
        [
            int(math.ceil(target_num_samples * weight * 1.005))
            for target_num_samples in target_num_samples_per_split
        ]
        for weight in weights
    ]

    return prefixes, weights, sizes_per_dataset
