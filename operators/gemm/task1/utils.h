
#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

// ==========================================
// Task 1: Common Utilities
// ==========================================
constexpr unsigned int CHASE_UNROLL = 16;

// Generate a random pointer-chasing cycle
void createRandomChain(std::vector<unsigned int>& buf)
{
    const size_t N = buf.size();

    std::vector<unsigned int> perm(N);

    std::iota(perm.begin(), perm.end(), 0u);

    std::mt19937 rng(12345);
    std::shuffle(perm.begin(), perm.end(), rng);

    for(size_t i = 0; i + 1 < N; ++i)
    {
        buf[perm[i]] = perm[i + 1];
    }

    buf[perm[N - 1]] = perm[0];
}

// Return median
float getMedian(std::vector<float>& results)
{
    std::sort(results.begin(), results.end());

    return results[results.size() / 2];
}
