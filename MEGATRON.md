# Megatron Dataloader integration to Nanotron
# 1. How does the iterator output look like? (Dimensions)
## nanotron
Nanotron gets the batch from the dataloader with the following statement: `batch=(next(dataloader) for _ in range(self.n_micro_batches_per_batch))`. It's a generator object, can't be pickled.

Then we do `batch = iter(batch)`. Nothing happens.

L#276 engine.py: `for micro_batch in batch:`
micro_batch (micro_batch.pkl) looks like:
- type: Dict: 'input_ids', 'input_mask', 'label_ids', 'label_mask' 
- shape: [micro_batch_size, sequence_length]
- label_ids are next tokens of input_ids
- masks are just `True`

## Megatron
In get_batch the next(data_iterator) we get a dict containing a "text" field with a tensor of shape [micro_batch_size, seq_len+1]. The tokens_ is just the tensor itself without the dict. 

So, the iterator gives us a tensor of [micro_batch_size, seq_len+1], we need to craft the 'input_ids', 'input_mask', 'label_ids', 'label_mask'. # TODO Esto es lo que hace el DataCollatorForCLM que a su vez se encarga de los putos tensor pointers y tal!!!! hay que hacer que el dataset de megatron sea  compatible con el distributed sampler, el skip batch y el dataloader et voila

Hay 2 cosas: Dataset y dataloading. Para no liarla mucho, mejor hacer que el dataset de megatron funcione con el dataloader de nanotron, que justamente es el de torch con el distributed sampler y las skip batches incorporado. Asi nos dejamos de lios e historias con el data parallellism y tensor parallelism que hacer y tal

Acordarse que en Megatron el data iterator lo tienen solo 1 proceso de los que hayan de data parallel... ojo con esto, porque en nanotron lo deben de tener todos...

En nanotron para el distributed sampler: Todos lo crean, solo que los procesos del mismo tensor parallel rank CREAN el mismo, por lo que leen lo mismo. Perfecto, tenemos mas lecturas eso si pero nos ahorramos el broadcast.

# 2. How are the batches build?
## nanotron (build masks and labels_ids?)
## Megatron (build masks and labels_ids?) (def get_batch()...)

# 3. How is the 3D parallelism handled?
## Data parallelism
### nanotron
A `DistributedSampler` is created with the DP size as world size (num_replicas) and DP rank as rank. 
### Megatron
Creo que solo el rank 0 del tensor parallel group crea el iterator; El resto espera al broadcast
## Tensor parallelism
### nanotron
All the processes from the same TP group create the same `DistributedSampler`.
### Megatron
Only 1 rank from the tensor parallel group gets the iterator; then it broadcast the data to the other processes from the tensor parallel rank
## Pipeline parallelism
### nanotron
It's handled via the `DataCollatorForCLM`. In every process, all the batch tensors ('input_ids', 'input_mask', 'label_ids', 'label_mask') have to be either the proper tensor OR a TensorPointer to the rank within the pipeline which contains the Tensor. From the docs:

IMPORTANT NOTE: When preparing your dataloader, make sure every tensor lives on a single rank, and other ranks must have TensorPointer to that rank. This is a requirement for the pipeline engine to work.

### Megatron
If the process it's not the first or the last pipeline stage, get_batch returns None, None, etc.


1. How Megatron .bin files are builded? (Does it concatenates several texts? What are "documents"?)
- Necessary to include vocab file and merges.txt
1. Proposed solution
2. Runs with 1 GPU (debug_llama_1GPU.yaml)
3. Runs with 4 GPUs (Llama 7B)
4. Other considerations
Take into consideration that only the first processes of the pipeline get the data, the others get the dummy dataloader! Port also the dummy dataloader to Nanotron