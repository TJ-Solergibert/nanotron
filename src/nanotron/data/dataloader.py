from nanotron import logging
from nanotron.logging import log_rank

import torch.distributed as dist # TODO change for nanotron function. Just used to check for rank 0 to build the helpers
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from nanotron.dataloader import SkipBatchSampler, get_dataloader_worker_init

from nanotron.data.blended_megatron_dataset_config import GPTDatasetConfig
from nanotron.data.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from nanotron.data.gpt_dataset import GPTDataset

from dataclasses import dataclass
import torch
import numpy as np
from typing import Dict, List, Union, Optional
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel import ParallelContext
from nanotron.dataloader import EmptyInfiniteDataset

logger = logging.get_logger(__name__)

def build_megatron_dataloader(
        dataset,
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

    # Case of ranks not requiring data. We give them a dummy dataset, then the collator will do his job
    # TODO Actually give dummy dataset, for now we are just passing the same dataset. In all processes we access the dataset despite its useless


    # Case of ranks requiring data
    if not dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        dataset_length = len(dataset)
        train_dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0
    
    data_collator = MegatronDataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
    )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()
    

    # TODO Only the training dataset can have a skip sampler?
    sampler = get_sampler(
        dataset=dataset,
        dl_ranks_size=dp_ranks_size,
        dl_rank=dp_rank,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
    )

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
        # TODO @thomasw21: I'm not sure but this doesn't seem to work at all.
        # pin_memory_device="cuda",
    )

def build_megatron_datasets(
        seed: int,
        sequence_length: int,
        data_path: str,
        split: str,
        train_iters: int,
        eval_interval: int,
        eval_iters: int,
        global_batch_size: int
): 
    # Install helpers. Only performed by 1 process
    if dist.get_rank() == 0:
        log_rank("Compiling dataset index builder ...", logger=logger, level=logging.INFO, rank=0)
        from nanotron.data.utils import compile_helpers

        compile_helpers()
        log_rank("Done with dataset index builder.", logger=logger, level=logging.INFO, rank=0)
    
    # Create GPT Dataset config 
    # TODO Change for config args of trainer, they will belong to the .yaml file
    gpt_config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=sequence_length,
        data_path=data_path,
        split=split,
    )

    # TODO compute train_val_test_num_samples!!!!!!!!!! 
    # Añadir los prints de megatron de cuantos tokens ya, cuantas batches y tokens?
    # Hay las iters y las samples. Primero se calcula el numero total de iters y se multiplica por la 
    # Global batch size para tener el numero de samples
    # Vale a ver: Train iters es un numero, el numero de samples sera train_iter * global batch size
    # Valid es primero calcular el numero de iters (12 trainig iters y eval-interval de 5 hara eval en 5, 10 y 12)
    # que con 5 eval iters hara un total de 15 iters y un total de samples de 15 iters * global batch size
    train_val_test_num_samples = compute_datasets_num_samples(train_iters=train_iters,
                                                            eval_interval=eval_interval, 
                                                            eval_iters=eval_iters,
                                                            global_batch_size=global_batch_size)
    train_dataset, valid_dataset, test_dataset = BlendedMegatronDatasetBuilder(GPTDataset, train_val_test_num_samples, gpt_config).build()

    return train_dataset, valid_dataset, test_dataset

@dataclass
class MegatronDataCollatorForCLM:
    # TODO refractor docs
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
        
        # TODO Acordarse que en Megatron dataset no hay collator... Recibimos directamente un dict con un tensor de [micro_batch_size, seq_len+1]...
        # TODO Probar primero con dataloader a ver que nos llega claro
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        
        
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            # assert all(len(example) == 0 for example in examples)
            # TODO This assert is because the tricky thing of the empty dataset, but as we keep the dataset in all ranks we quit the assertion (THE HACKY THING)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        # Make sure we load only what's necessary, ie we only load a `input_ids` column.
        # TODO Megatron datasets key is "text"
        assert all(list(example.keys()) == ["text"] for example in examples)

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = np.vstack([examples[i]["text"] for i in range(len(examples))])  # (b, s)
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

def get_sampler(
    dl_ranks_size: int,
    dl_rank: int,
    dataset, # TODO introduce type int dataset etc
    seed: int,
    use_loop_to_round_batch_size: bool,
    consumed_train_samples: int,
    micro_batch_size: Optional[int] = None,
    drop_last: Optional[bool] = True,
) -> Optional[torch.utils.data.Sampler]:
    """returns sampler that restricts data loading to a subset of the dataset proper to the DP rank"""

    # Build the sampler.
    # TODO @nouamanetazi: Support group_by_length: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L783-L810
    # TODO Take a look on this
    """
    if use_loop_to_round_batch_size:
        assert micro_batch_size is not None
        # loops at the end back to the beginning of the shuffled samples to make each process have a round multiple of batch_size samples.
        sampler = DistributedSamplerWithLoop(
            train_dataset,
            batch_size=micro_batch_size,
            num_replicas=dl_ranks_size,
            rank=dl_rank,
            seed=seed,
            drop_last=drop_last,
        )
    else:
        sampler = DistributedSampler(
            train_dataset, num_replicas=dl_ranks_size, rank=dl_rank, seed=seed, drop_last=drop_last
        )
    """
    sampler = DistributedSampler(
            dataset, num_replicas=dl_ranks_size, rank=dl_rank, seed=seed, drop_last=drop_last
        )

    if consumed_train_samples > 0:
        sampler = SkipBatchSampler(sampler, skip_batches=consumed_train_samples, dp_size=dl_ranks_size)

    return sampler

# TODO Move from here! To utils?
def compute_datasets_num_samples(train_iters, eval_interval, eval_iters,global_batch_size):
    
    train_samples = train_iters * global_batch_size
    eval_iters = (train_iters // eval_interval + 1) * eval_iters
    test_iters = eval_iters

    datasets_num_samples = [train_samples,
                            eval_iters * global_batch_size,
                            test_iters * global_batch_size]
    
    log_rank(" > Datasets target sizes (minimum size):", logger=logger, level=logging.INFO, rank=0)
    log_rank("    Train:      {}".format(datasets_num_samples[0]), logger=logger, level=logging.INFO, rank=0)
    log_rank("    Validation: {}".format(datasets_num_samples[1]), logger=logger, level=logging.INFO, rank=0)
    log_rank("    Test:       {}".format(datasets_num_samples[2]), logger=logger, level=logging.INFO, rank=0)
    
    return datasets_num_samples