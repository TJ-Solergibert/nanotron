"""
import os

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = '127.0.0.1'
os.environ["MASTER_PORT"] = '29500'
"""
import os
# Collator?
from dataclasses import dataclass
import torch
import numpy as np
from typing import Dict, List, Union, Optional
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer

import torch.distributed as dist
import time

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset

from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from nanotron.dataloader import SkipBatchSampler, get_dataloader_worker_init

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group("gloo", rank=rank, world_size=world_size)

def _get_train_sampler(
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
    # TODO We use summy pp_rank for debug
    #parallel_context: ParallelContext
    pp_rank: int

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        
        # TODO Acordarse que en Megatron dataset no hay collator... Recibimos directamente un dict con un tensor de [micro_batch_size, seq_len+1]...
        # TODO Probar primero con dataloader a ver que nos llega claro
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        
        # TODO debug 
        #current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        current_pp_rank = self.pp_rank
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            # assert all(len(example) == 0 for example in examples)
            # TODO This assert is because the tricky thing of the empty dataset, but as we keep the dataset in all ranks we quit the assertion
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

def get_megatron_dataloader(
        dataset,
        sequence_length: int,
        #parallel_context: ParallelContext,
        dp_rank,
        dp_size,
        pp_rank,
        ########
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
    # TODO Actually give dummy dataset, for now we are just passing the same dataset
    #print(f"[{os.environ['RANK']}]: {pp_rank}")
    if pp_rank != 0:
        print(f"[{os.environ['RANK']}]: Setting Dummy dataset")
    
    data_collator = MegatronDataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        #parallel_context=parallel_context,
        pp_rank = pp_rank

    )

    # Compute size and rank of dataloader workers
    # dp_ranks_size = parallel_context.dp_pg.size()
    # dp_rank = parallel_context.dp_pg.rank()
    dp_ranks_size = dp_size
    #dp_rank = dp_rank

    # TODO Only the training dataset can have a skip sampler?
    train_sampler = _get_train_sampler(
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
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
        # TODO @thomasw21: I'm not sure but this doesn't seem to work at all.
        # pin_memory_device="cuda",
    )

##################
##################
##################
DP_SIZE = 2
PP_SIZE = 2

sequence_length = 1024
micro_batch_size = 4
jump_n_batches = 10
consumed_train_samples = micro_batch_size * jump_n_batches * DP_SIZE # micro batch size * n batches * DP_SIZE
num_loading_workers = 1
seed = 1234
##################
##################
##################


data_path = "/mloscratch/homes/solergib/s-ai/nanotron/datasets/europarl-gpt_text_document" 
split = "949,50,1"
gpt_config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=sequence_length,
        data_path=data_path,
        split=split,
    )

train_val_test_num_samples = [4000000, 80080, 80]

# Helpers!!!
if dist.get_rank() == 0:
    start_time = time.time()
    print("> compiling dataset index builder ...")
    from megatron.core.datasets.utils import compile_helpers

    compile_helpers()
    print(
        ">>> done with dataset index builder. Compilation time: {:.3f} "
        "seconds".format(time.time() - start_time),
        flush=True,
    )



train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(GPTDataset, train_val_test_num_samples, gpt_config).build()

train_dataloader = get_megatron_dataloader(
    train_ds,
    sequence_length,
    #parallel_context=trainer.parallel_context,
    dp_rank=int(rank//DP_SIZE),
    dp_size=DP_SIZE,
    pp_rank=int(rank%PP_SIZE),
    input_pp_rank=0,
    output_pp_rank=int(PP_SIZE-1),
    micro_batch_size=micro_batch_size,
    consumed_train_samples=consumed_train_samples,
    dataloader_num_workers=num_loading_workers,
    seed_worker=seed,
    dataloader_drop_last=True,
)

#print(f"[{os.environ['RANK']}]: Dataloader.dataset: {train_dataloader.dataset}")
i = 0 # start

for data in train_dataloader: # Imprime con microbatchsize 4 el resultado que queremos ver dentro del engine. NO conseguimos replicar lo del iter y los next que hace, ni idea, pero pasamos a megatron
   
    if rank == 0:
        #print(f"[{os.environ['RANK']}]: batch [{i}]: {data}")
        print(f"[{os.environ['RANK']}]: BATCH [{i}] input_ids[0][0]: {data['input_ids'][0][0]}")
        #print(type(data))
        #print(data['input_ids'].shape)
    i += 1
    if i == 12:
        break