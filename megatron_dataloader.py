from nanotron import logging
import dataclasses
from typing import Dict, Generator, Iterator, List, Optional, Union

import numpy as np
import torch
from nanotron.parallel import ParallelContext
from torch.utils.data import BatchSampler, DataLoader
from nanotron import distributed as dist
from nanotron.dataloader import EmptyInfiniteDataset, _get_train_sampler, get_dataloader_worker_init
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.logging import log_rank

# Megatron imports
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
# from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig
from utils import GPTDatasetConfig # TODO mover de aqui
from megatron.core.datasets.gpt_dataset import GPTDataset

logger = logging.get_logger(__name__)

@dataclasses.dataclass
class DataCollatorForCLM:
    """
    Data collator used for causal language modeling.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        # Make sure we load only what's necessary, ie we only load a `input_ids` column.
        assert all(list(example.keys()) == ["input_ids"] for example in examples)

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        assert (
            expanded_input_length == self.sequence_length + 1
        ), f"Samples should be of length {self.sequence_length + 1} (seq_len+1), but got {expanded_input_length}"

        # Process inputs: last token is the label
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]
            result["input_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

        if isinstance(result["input_ids"], torch.Tensor) and result["input_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )
        if isinstance(result["label_ids"], torch.Tensor) and result["label_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )

        # Cast np.array to torch.Tensor
        result = {k: v if isinstance(v, TensorPointer) else torch.from_numpy(v) for k, v in result.items()}
        return result
    

def get_megatron_train_dataloader(
    train_dataset,
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    consumed_train_samples: int,
    dataloader_num_workers: int,
    seed_worker: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_loop_to_round_batch_size: bool = False,
) -> DataLoader:
    #if not isinstance(train_dataset, datasets.Dataset): TODO: Poner bien esto
    #    raise ValueError(f"training requires a datasets.Dataset, but got {type(train_dataset)}")

    # Case of ranks requiring data
    if dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        train_dataset = train_dataset.with_format(type="numpy", columns=["input_ids"], output_all_columns=True)

    # Case of ranks not requiring data. We give them an infinite dummy dataloader # TODO: No es un dataloader, es un dataset, que luego con el collator se transforma. No engañeis a mi padre
    else:
        #
        # TODO: Solamente se tiene que coger la length del dataset y hacer el empty 
        assert train_dataset.column_names == ["input_ids"], (
            f"Dataset has to have a single column, with `input_ids` as the column name. "
            f"Current dataset: {train_dataset}"
        )
        dataset_length = len(train_dataset)
        train_dataset = train_dataset.remove_columns(column_names="input_ids")
        assert (
            len(train_dataset) == 0
        ), f"Dataset has to be empty after removing the `input_ids` column. Current dataset: {train_dataset}"
        # HACK as if we remove the last column of a train_dataset, it becomes empty and it's number of rows becomes empty.
        # TODO: Este hack solo quiere decir que se elimina el dataset
        train_dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0

    data_collator = DataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
    )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()

    # TODO @nouamanetazi: Remove unused columns: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L852
    # TODO @nouamanetazi: Support torch.utils.data.IterableDataset: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L855-L872

    train_sampler = _get_train_sampler(
        train_dataset=train_dataset,
        dl_ranks_size=dp_ranks_size,
        dl_rank=dp_rank,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
    )

    return DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
        # TODO @thomasw21: I'm not sure but this doesn't seem to work at all.
        # pin_memory_device="cuda",
    )

def core_gpt_dataset_config_from_args(args):
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        return_document_ids=args.retro_return_doc_ids
    )

def gpt_train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    # args = get_args()

    log_rank("> building train, validation, and test datasets for GPT ...", logger=logger, level=logging.INFO, rank=0)

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        core_gpt_dataset_config_from_args(args)
    ).build()

    log_rank("> finished creating GPT datasets ...", logger=logger, level=logging.INFO, rank=0)

    return train_ds, valid_ds, test_ds


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""

    #args = get_args()

    train_iters = 500000
    global_batch_size = 8
    eval_interval = 500

    # Number of train/valid/test samples.
    #if args.train_samples:
    #    train_samples = args.train_samples
    # else:
    if True:
        train_samples = train_iters * global_batch_size
    eval_iters = (train_iters // eval_interval + 1) * \
                 eval_iters
    test_iters = eval_iters
    train_val_test_num_samples = [train_samples,
                                  eval_iters * global_batch_size,
                                  test_iters * global_batch_size]
    
    log_rank(" > datasets target sizes (minimum size):", logger=logger, level=logging.INFO, rank=0)
    log_rank("    train:      {}".format(train_val_test_num_samples[0]), logger=logger, level=logging.INFO, rank=0)
    log_rank("    validation: {}".format(train_val_test_num_samples[1]), logger=logger, level=logging.INFO, rank=0)
    log_rank("    test:       {}".format(train_val_test_num_samples[2]), logger=logger, level=logging.INFO, rank=0)

    # log_rank("", logger=logger, level=logging.INFO, rank=0)

    # Build the datasets.
    return build_train_valid_test_datasets_provider(train_val_test_num_samples)

def build_megatron_dataloaders(trainer, meg_data_path): # TODO Add DistributedTrainer
    """Returns train, valid and test dataloaders"""

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    log_rank("Using megatron!!!", logger=logger, level=logging.INFO, rank=0)

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(gpt_train_valid_test_datasets_provider)

    # We load the processed dataset on the ranks requiring it
    dataloader = get_train_dataloader(
        train_dataset=train_dataset,
        sequence_length=trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        consumed_train_samples=trainer.consumed_train_samples,
        dataloader_num_workers=trainer.config.data.num_loading_workers,
        seed_worker=trainer.config.data.seed,
        dataloader_drop_last=True,
    )
    # Check if we have enough samples for train_steps
    total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
    num_tokens_needed_for_training = (
        (trainer.config.tokens.train_steps - trainer.start_iteration_step)
        * trainer.global_batch_size
        * trainer.sequence_length
    )
    assert num_tokens_needed_for_training <= total_tokens_dataset, (
        f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
        f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.start_iteration_step}"
    )

    return dataloader