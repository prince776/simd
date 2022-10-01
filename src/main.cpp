#pragma GCC target("avx2")
#pragma GCC optimize("O3")

#include <iostream>
#ifdef __x86_64__
#include <immintrin.h>
#else
#include "sse2neon.h"
#endif

using namespace std;

// void cpuSupport()
// {
//     cout << __builtin_cpu_supports("sse") << endl cout << __builtin_cpu_supports("sse2") << endl;
//     cout << __builtin_cpu_supports("avx") << endl;
//     cout << __builtin_cpu_supports("avx2") << endl;
//     cout << __builtin_cpu_supports("avx512f") << endl;
// }

void basic()
{
    double a[100], b[100], c[100];

    for (int i = 0; i < 100; i++)
    {
        a[i] = i;
        b[i] = i;
    }
    // iterate in blocks of 2,
    // because that's how many doubles can fit into a 128-bit register
    for (int i = 0; i < 100; i += 2)
    {
        // load two 128-bit segments into registers
        __m128d x = _mm_loadu_pd(&a[i]);
        __m128d y = _mm_loadu_pd(&b[i]);

        // add 4+4 64-bit numbers together
        __m128d z = _mm_add_pd(x, y);

        // write the 256-bit result into memory, starting with c[i]
        _mm_storeu_pd(&c[i], z);
    }
    for (int i = 0; i < 10; i++)
        cout << c[i] << ", ";
    cout << endl;
}

constexpr int T = 4;
typedef int v4si __attribute__((vector_size(T * sizeof(int))));

auto print = [](v4si x)
{
    for (int i = 0; i < 4; i++)
        cout << x[i] << ", ";
    cout << endl;
};

void gcc_vector_extension()
{
    v4si a = {1, 2, 3, 4};
    v4si b = {1, 2, 3, 4};
    v4si c = a + b;

    print(c);
    c *= 2;
    print(c);

    v4si x = {}; // all zeros
    print(x);
    v4si y = 42 + v4si{}; // broadcast
    print(y);
}

int sum_simd(int *a, int n)
{
    v4si *as = (v4si *)a;
    v4si s = {0};
    for (int i = 0; i < n / 4; i++)
        s += as[i];

    int res = 0;
    // sum the 4 accumulators into one
    for (int i = 0; i < 4; i++)
        res += s[i];
    // add the remainder of a
    for (int i = (n / 8) * 8; i < n; i++)
        res += a[i];
    return res;
}

// in prev impl we are still waiting 1 cycle b/w the loop iterations
// for vector addition (data hazard). but its throughput is 2, so we can
// optimize by having these two interleave, and since they are independent we get faster sum
int sum_simd_faster(int *a, int n)
{
    constexpr int B = 2;
    v4si *as = (v4si *)a;
    v4si s[B] = {0};
    for (int i = 0; i + (B - 1) < n / T; i += B)
        for (int j = 0; j < B; j++)
            s[j] += as[i + j];

    // sum all accumualators into one
    for (int i = 1; i < B; i++)
        s[0] += s[i];

    int res = 0;
    for (int i = 0; i < T; i++)
        res += s[0][i];

    for (int i = n / T * T; i < n; i++)
        res += a[i];
    return res;
}

// For 256, idh that because m1 sucks
// int hsum(__m256i x) {
//     __m128i l = _mm256_extracti128_si256(x, 0);
//     __m128i h = _mm256_extracti128_si256(x, 1);
//     l = _mm_add_epi32(l, h);
//     l = _mm_hadd_epi32(l, l);
//     return _mm_extract_epi32(l, 0) + _mm_extract_epi32(l, 1);
// }

int hsum(__m128 x)
{
    x = _mm_hadd_epi32(x, x);
    return _mm_extract_epi32(x, 0) + _mm_extract_epi32(x, 1);
}

// Sum using horizontal sum
int sum_simd_hsum(int *a, int n)
{
    constexpr int B = 2;
    v4si *as = (v4si *)a;
    v4si s[B] = {0};
    for (int i = 0; i + (B - 1) < n / T; i += B)
        for (int j = 0; j < B; j++)
            s[j] += as[i + j];

    // sum all accumualators into one
    for (int i = 1; i < B; i++)
        s[0] += s[i];

    int res = hsum(s[0]);
    for (int i = n / T * T; i < n; i++)
        res += a[i];
    return res;
}

void sum_simd_test()
{
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int n = sizeof(a) / sizeof(int);

    int expected = 0;
    for (int i = 0; i < n; i++)
        expected += a[i];
    int got = sum_simd(a, n);
    assert(expected == got);
    cout << "Got correct sum: " << got << endl;
    got = sum_simd_faster(a, n);
    assert(expected == got);
    cout << "Got correct sum(faster): " << got << endl;
    got = sum_simd_hsum(a, n);
    assert(expected == got);
    cout << "Got correct sum(hsum): " << got << endl;
}

int main()
{
    // cpuSupport();
    // basic();
    // gcc_vector_extension();
    sum_simd_test();
}