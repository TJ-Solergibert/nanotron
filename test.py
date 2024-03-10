import os

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = '127.0.0.1'
os.environ["MASTER_PORT"] = '29500'

import torch.distributed as dist
import time

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset

dist.init_process_group("gloo", rank=0, world_size=1)

data_path = "/mloscratch/homes/solergib/s-ai/nanotron/datasets/europarl-gpt_text_document" 
split = "949,50,1"
gpt_config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
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