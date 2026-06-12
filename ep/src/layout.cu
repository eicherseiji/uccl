#include "ep_configs.cuh"
#include "ep_launch.cuh"
#include "exception.cuh"
#include "layout.hpp"

namespace uccl {
namespace layout {

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM,
          int kNumNVLPeers>
__global__ void get_dispatch_layout(int64_t const* topk_idx,
                                    int* num_tokens_per_rank,
                                    int* num_tokens_per_rdma_rank,
                                    int* num_tokens_per_expert,
                                    bool* is_token_in_rank, int num_tokens,
                                    int num_topk, int num_ranks,
                                    int num_experts) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x);

  // Count expert statistics
  __shared__ int num_tokens_per_expert_per_thread[kNumThreads]
                                                 [kNumExpertsPerSM];
  int expert_begin_idx = sm_id * kNumExpertsPerSM,
      expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
  if (expert_begin_idx < expert_end_idx) {
// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumExpertsPerSM; ++i)
      num_tokens_per_expert_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
#pragma unroll
      for (int j = 0, expert_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
          ++num_tokens_per_expert_per_thread[thread_id]
                                            [expert_idx - expert_begin_idx];
      }
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads,
                     "Too many experts per SM");
    if (expert_begin_idx + thread_id < expert_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_expert_per_thread[i][thread_id];
      num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
    }
    return;
  }

  if (num_tokens_per_rdma_rank != nullptr)
    EP_DEVICE_ASSERT(num_ranks % kNumNVLPeers == 0 and
                     num_ranks > kNumNVLPeers);

  // Count rank statistics
  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / kNumNVLPeers;
  __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
  __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads]
                                                    [kNumRDMARanksPerSM];
  auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
  int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM,
      rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
  int rdma_rank_begin_idx = rank_begin_idx / kNumNVLPeers,
      rdma_rank_end_idx =
          (rank_end_idx + kNumNVLPeers - 1) / kNumNVLPeers;
  if (rank_begin_idx < rank_end_idx) {
    auto const num_expert_per_rank = num_experts / num_ranks;
    auto expert_begin = rank_begin_idx * num_expert_per_rank;
    auto expert_end = rank_end_idx * num_expert_per_rank;

// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumRanksPerSM; ++i)
      num_tokens_per_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = 0; i < kNumRDMARanksPerSM; ++i)
      num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
      int is_in_rank[kNumRanksPerSM] = {0},
          is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll
      for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin <= expert_idx and expert_idx < expert_end) {
          // Count single rank
          rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
          is_in_rank[rank_idx]++,
              is_in_rdma_rank[rank_idx / kNumNVLPeers]++;
        }
      }

      auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll
      for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
        num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
      }

#pragma unroll
      for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
        num_tokens_per_rdma_rank_per_thread[thread_id][j] +=
            (is_in_rdma_rank[j] > 0);
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
    if (rank_begin_idx + thread_id < rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_rank_per_thread[i][thread_id];
      num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
    }

    if (num_tokens_per_rdma_rank != nullptr and
        rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
      num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
    }
  }
}

void get_dispatch_layout(int64_t const* topk_idx, int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank,
                         int num_tokens, int num_topk, int num_ranks,
                         int num_nvl_ranks, int num_experts,
                         cudaStream_t stream) {
#define DISPATCH_LAYOUT_LAUNCH_CASE(num_nvl_ranks_value)                  \
  do {                                                                    \
    LAUNCH_KERNEL(&cfg,                                                   \
                  (get_dispatch_layout<kNumThreads, kNumExpertsPerSM,     \
                                       kNumRanksPerSM,                    \
                                       num_nvl_ranks_value>),             \
                  topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank, \
                  num_tokens_per_expert, is_token_in_rank, num_tokens,     \
                  num_topk, num_ranks, num_experts);                      \
  } while (false)

  constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
  int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
                (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
  EP_HOST_ASSERT(num_nvl_ranks > 0 && num_nvl_ranks <= NUM_MAX_NVL_PEERS);
  EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
  EP_HOST_ASSERT(kNumRanksPerSM % num_nvl_ranks == 0);

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  switch (num_nvl_ranks) {
    case 1:
      DISPATCH_LAYOUT_LAUNCH_CASE(1);
      break;
    case 2:
      DISPATCH_LAYOUT_LAUNCH_CASE(2);
      break;
    case 4:
      DISPATCH_LAYOUT_LAUNCH_CASE(4);
      break;
    case 8:
      DISPATCH_LAYOUT_LAUNCH_CASE(8);
      break;
    default:
      EP_HOST_ASSERT(false && "Unsupported NVL peers");
  }
#undef DISPATCH_LAYOUT_LAUNCH_CASE
}

}  // namespace layout
}  // namespace uccl
