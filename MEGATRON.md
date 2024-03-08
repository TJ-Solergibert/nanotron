# Megatron integration to Nanotron
1. How do Nanotron inputs look like? (Dimensions)

Nanotron gets the batch from the dataloader with the following statement: `batch=(next(dataloader) for _ in range(self.n_micro_batches_per_batch))`. It's a generator object, can't be pickled.

Then we do `batch = iter(batch)`. Nothing happens.

L#276 engine.py: `for micro_batch in batch:`
micro_batch (micro_batch.pkl) looks like: 
- type: Dict: 'input_ids', 'input_mask', 'label_ids', 'label_mask' 
- [micro_batch_size, sequence_length]
- label_ids are next tokens of input_ids

2. How do Megatron batches look like? (Dimensions)
   
2. How Megatron batches are builded? (Does it concatenates several texts? What are "documents"?)
3. How does Megatron serve batches? (def get_batch?)
4. Proposed solution
5. Runs with 1 GPU (debug_llama_1GPU.yaml)
6. Runs with 4 GPUs (Llama 7B)
7. Other considerations
Take into consideration that only the first processes of the pipeline get the data, the others get the dummy dataloader! Port also the dummy dataloader to Nanotron