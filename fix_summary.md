# Fix Summary: Tensor Parallelism Issue in Qwen3 Reranker

## Problem
When running Qwen3-Reranker-4B with `tensor_parallel_size=2`, the model loading failed with:
```
RuntimeError: The size of tensor a (1280) must match the size of tensor b (2560) at non-singleton dimension 1
```

This error occurred in the `load_weights_using_from_2_way_softmax` function when trying to copy a full-size weight vector to a sharded score layer weight.

## Root Cause
The issue was in `vllm/model_executor/models/adapters.py` in two functions:
1. `load_weights_using_from_2_way_softmax` (line 355)
2. `load_weights_no_post_processing` (line 411)

The `score` layer in sequence classification models uses `RowParallelLinear`, which shards the input dimension across tensor parallel ranks. However, the weight loading functions were trying to copy full-size weights without accounting for this sharding.

## Solution
Added tensor parallelism awareness to both functions:

1. **Single weight vector case** (`load_weights_using_from_2_way_softmax`):
   - Extract the appropriate shard: `weight[start_idx:start_idx + shard_size]`
   - Where `shard_size = hidden_size // tp_size`

2. **Weight matrix case** (`load_weights_no_post_processing`):
   - Extract the appropriate shard: `score_weight[:, start_idx:start_idx + shard_size]`
   - Shard along the last dimension (hidden dimension)

3. **Added safety checks**:
   - Assert that `hidden_size % tp_size == 0` to ensure proper divisibility

## Changes Made
- Modified `vllm/model_executor/models/adapters.py`:
  - Added tensor parallel rank and size detection
  - Added weight sharding logic for both functions
  - Added divisibility assertions for safety
- Added unit test in `tests/models/language/pooling/test_qwen3_reranker.py`:
  - Tests tensor parallelism weight loading without dependencies
  - Validates that no tensor size mismatch errors occur

## Testing the Fix
The original failing command should now work:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Reranker-4B \
  --task score \
  --enforce_eager True \
  --served_model_name Qwen/Qwen3-Reranker-4B-30k \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.97
```

## Impact
- Fixes tensor parallelism support for Qwen3 reranker models
- No impact on single-GPU usage (tp_size=1)
- Minimal performance impact (only adds a few tensor operations during model loading)
- Backward compatible - no API changes