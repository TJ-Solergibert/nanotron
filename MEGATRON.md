# Megatron integration to Nanotron
1. How do Nanotron inputs look like? (Dimensions)

Nanotron gets the batch from the dataloader with the following statement: `batch=(next(dataloader) for _ in range(self.n_micro_batches_per_batch))`. It's a generator object, can't be pickled.

Then we do `batch = iter(batch)`. Nothing happens.

L#276 engine.py: `for micro_batch in batch:`
micro_batch (micro_batch.pkl) looks like:
- type: Dict: 'input_ids', 'input_mask', 'label_ids', 'label_mask' 
- shape: [micro_batch_size, sequence_length]
- label_ids are next tokens of input_ids

2. What does the iterator output look like? (Dimensions)
In get_batch the next(data_iterator) we get a dict containing a "text" field with a tensor of shape [micro_batch_size, seq_len+1]. The tokens_ is just the tensor itself without the dict. 

So, the iterator gives us a tensor of [micro_batch_size, seq_len+1], we need to craft the 'input_ids', 'input_mask', 'label_ids', 'label_mask'.
3. How does Megatron serve batches? (build masks and labels_ids?)
 
4. How Megatron .bin files are builded? (Does it concatenates several texts? What are "documents"?)
- Necessary to include vocab file and merges.txt
5. Proposed solution
6. Runs with 1 GPU (debug_llama_1GPU.yaml)
7. Runs with 4 GPUs (Llama 7B)
8. Other considerations
Take into consideration that only the first processes of the pipeline get the data, the others get the dummy dataloader! Port also the dummy dataloader to Nanotron