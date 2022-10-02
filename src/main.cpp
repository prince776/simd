#pragma GCC target("avx2")
#pragma GCC optimize("O3")

#include <iostream>
#include <random>
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

void print(v4si x)
{
    for (int i = 0; i < 4; i++)
        cout << x[i] << ", ";
    cout << endl;
}

void print(__m128i x)
{
    auto t = (int *)&x;
    for (int i = 0; i < T; i++)
        std::cout << t[i] << " ";
    std::cout << std::endl;
}
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

int hsum(__m128i x)
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

// sum all values less than k
int predicated_sum(int *a, int n, int k)
{
    v4si *as = (v4si *)a;
    v4si sum = {0};

    for (int i = 0; i < n / T; i++)
        sum += as[i] < k ? as[i] : 0;

    int res = hsum(sum);
    for (int i = n / T * T; i < n; i++)
        res += a[i] < k ? a[i] : 0;

    return res;
}

int predicated_sum_intrinsics_blend(int *a, int n, int k)
{
    const __m128i condn = _mm_set1_epi32(k - 1);
    const __m128i zero = _mm_setzero_si128();
    __m128i sum = zero;

    for (int i = 0; i + T - 1 < n; i += T)
    {
        __m128i curr = _mm_loadu_si128((__m128i *)&a[i]);
        __m128i mask = _mm_cmpgt_epi32(curr, condn);
        curr = _mm_blendv_epi8(curr, zero, mask);
        sum = _mm_add_epi32(sum, curr);
    }

    int res = hsum(sum);
    for (int i = n / T * T; i < n; i++)
        res += a[i] < k ? a[i] : 0;
    return res;
}

int predicated_sum_intrinsics_and(int *a, int n, int k)
{
    const __m128i condn = _mm_set1_epi32(k);
    const __m128i zero = _mm_setzero_si128();
    __m128i sum = zero;

    for (int i = 0; i + T - 1 < n; i += T)
    {
        __m128i curr = _mm_loadu_si128((__m128i *)&a[i]);
        __m128i mask = _mm_cmpgt_epi32(condn, curr);
        curr = _mm_and_si128(curr, mask);
        sum = _mm_add_epi32(sum, curr);
    }

    int res = hsum(sum);
    for (int i = n / T * T; i < n; i++)
        res += a[i] < k ? a[i] : 0;
    return res;
}

void predicated_sum_test()
{
    int k = 5;
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int n = sizeof(a) / sizeof(int);
    shuffle(a, a + n, default_random_engine(0));

    int expected = 0;
    for (int i = 0; i < n; i++)
        expected += a[i] < k ? a[i] : 0;

    int got = predicated_sum(a, n, k);
    assert(expected == got);
    cout << "Got correct predicated sum: " << got << endl;
    got = predicated_sum_intrinsics_blend(a, n, k);
    assert(expected == got);
    cout << "Got correct predicated sum(intrinsic blend): " << got << endl;
    got = predicated_sum_intrinsics_and(a, n, k);
    assert(expected == got);
    cout << "Got correct predicated sum(intrinsic and): " << got << endl;
}

int find_simd(int *a, int n, int k)
{
    __m128i needle = _mm_set1_epi32(k);
    for (int i = 0; i + T - 1 < n; i += T)
    {
        __m128i curr = _mm_loadu_si128((__m128i *)&a[i]);
        __m128i cmp = _mm_cmpeq_epi32(curr, needle);
        int mask = _mm_movemask_ps(cmp);
        if (mask != 0)
        {
            return i + __builtin_ctz(mask);
        }
    }
    for (int i = i / T * T; i < n; i++)
        if (a[i] == k)
            return i;
    return -1;
}

// we can use test to make it slightly faster
int find_simd_using_test(int *a, int n, int k)
{
    __m128i needle = _mm_set1_epi32(k);
    for (int i = 0; i + T - 1 < n; i += T)
    {
        __m128i curr = _mm_loadu_si128((__m128i *)&a[i]);
        __m128i cmp = _mm_cmpeq_epi32(curr, needle);
        if (!_mm_testz_si128(cmp, cmp))
        {
            int mask = _mm_movemask_ps(cmp);
            return i + __builtin_ctz(mask);
        }
    }
    for (int i = n / T * T; i < n; i++)
        if (a[i] == k)
            return i;
    return -1;
}

// test and movemask have throughput 1, so they bottleneck
// so what we can do is do two (load and cmpeq) manually to saturate better
int find_simd_2x(int *a, int n, int k)
{
    __m128i needle = _mm_set1_epi32(k);
    for (int i = 0; i + 2 * T - 1 < n; i += 2 * T)
    {
        __m128i curr1 = _mm_loadu_si128((__m128i *)&a[i]);
        __m128i curr2 = _mm_loadu_si128((__m128i *)&a[i + T]);
        __m128i cmp1 = _mm_cmpeq_epi32(curr1, needle);
        __m128i cmp2 = _mm_cmpeq_epi32(curr2, needle);
        __m128i cmpAgg = _mm_or_si128(cmp1, cmp2);
        if (!_mm_testz_si128(cmpAgg, cmpAgg))
        {
            int mask = (_mm_movemask_ps(cmp2) << T) + _mm_movemask_ps(cmp1);
            return i + __builtin_ctz(mask);
        }
    }
    for (int i = (n / (2 * T)) * (2 * T); i < n; i++)
        if (a[i] == k)
            return i;
    return -1;
}

// we can try doing 4 at a time, to saturate if decode width is 4 (since #instructions
// in prev is 5, so we are using 4/5
// int simd_find_4x(...) {...}

void find_test()
{
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int n = sizeof(a) / sizeof(int);

    for (int _ = 0; _ < 1000; _++)
    {
        shuffle(a, a + n, default_random_engine(0));
        int k = rand() % 10;

        int expected = -1;
        for (int i = 0; i < n; i++)
            if (a[i] == k)
                expected = i;

        int got = find_simd(a, n, k);
        assert(expected == got);
        got = find_simd_using_test(a, n, k);
        assert(expected == got);
        got = find_simd_2x(a, n, k);
        if (expected != got)
            cout << k << ", " << expected << ": " << got << endl;
        assert(expected == got);
    }
    cout << "Found correct idx" << endl;
}

int count_simd(int *a, int n, int k)
{
    __m128i needle = _mm_set1_epi32(k);
    __m128i ones = _mm_set1_epi32(1);
    __m128i sum = _mm_setzero_si128();

    for (int i = 0; i + T - 1 < n; i += T)
    {
        __m128i curr = _mm_loadu_si128((__m128i *)&a[i]);
        __m128i cmp = _mm_cmpeq_epi32(curr, needle);
        cmp = _mm_and_si128(cmp, ones);
        sum = _mm_add_epi32(sum, cmp);
    }

    int res = hsum(sum);
    for (int i = n / T * T; i < n; i++)
        res += a[i] == k;
    return res;
}

// prev impl is what auto vectorization will get
// we can speed it up by saturating more
// hence by doing in chunks of 2 independent loads
// as they are pipelined without data hazard
// i.e instruction level parallelism
// we can also just use the flag we get from cmpeq by negating it in end (as it is -1)
int count_simd_2x(int *a, int n, int k)
{
    __m128i needle = _mm_set1_epi32(k);
    __m128i sum1 = _mm_setzero_si128();
    __m128i sum2 = _mm_setzero_si128();

    for (int i = 0; i + 2 * T - 1 < n; i += 2 * T)
    {
        __m128i curr1 = _mm_loadu_si128((__m128i *)&a[i]);
        __m128i curr2 = _mm_loadu_si128((__m128i *)&a[i + T]);
        __m128i cmp1 = _mm_cmpeq_epi32(curr1, needle);
        __m128i cmp2 = _mm_cmpeq_epi32(curr2, needle);
        sum1 = _mm_add_epi32(sum1, cmp1);
        sum2 = _mm_add_epi32(sum2, cmp2);
    }

    sum1 = _mm_add_epi32(sum1, sum2);

    int res = -hsum(sum1);
    for (int i = n / T * T; i < n; i++)
        res += a[i] == k;
    return res;
}

void count_test()
{
    int a[] = {1, 2, 7, 4, 5, 7, 7, 8, 9};
    int n = sizeof(a) / sizeof(int);
    int k = 7;
    shuffle(a, a + n, default_random_engine(0));

    int expected = 0;
    for (int i = 0; i < n; i++)
        if (a[i] == k)
            expected++;

    int got = count_simd(a, n, k);
    assert(expected == got);
    got = count_simd_2x(a, n, k);
    assert(expected == got);
    cout << "Found correct count: " << got << endl;
}

// c[i] = a[i] * b[i]
void mul_simd_32_to_64_bit(int *a, int *b, long long *c, int n)
{
    for (int i = 0; i + T - 1 < n; i += T)
    {
        __m128i x = _mm_loadu_si128((__m128i *)&a[i]);
        __m128i y = _mm_loadu_si128((__m128i *)&b[i]);

        // do _mm_mullo_epi32 if result is 32 bit, but result can be 64 bit here
        // 0b|00|01|00|00 first element stores a[0] and third stores a[1]
        // 0b|00|11|00|10 first element stores a[2] and third stores a[3]
        __m128i x1 = _mm_shuffle_epi32(x, 0b00010000);
        __m128i x2 = _mm_shuffle_epi32(x, 0b00110010);
        __m128i y1 = _mm_shuffle_epi32(x, 0b00010000);
        __m128i y2 = _mm_shuffle_epi32(x, 0b00110010);

        x = _mm_mul_epi32(x1, y1);
        y = _mm_mul_epi32(x2, y2);

        _mm_storeu_si128((__m128i *)&c[i], x);
        _mm_storeu_si128((__m128i *)&c[i + T / 2], y);
    }

    for (int i = n / T * T; i < n; i++)
        c[i] = (long long)a[i] * b[i];
}

void mul_simd_test()
{
    int a[] = {1, 2, 7, 4, 5, 7, 7, 8, 9};
    int b[] = {1, 2, 7, 4, 5, 7, 7, 8, 9};
    constexpr int n = sizeof(a) / sizeof(int);
    long long expected[n] = {0};
    for (int i = 0; i < n; i++)
        expected[i] = (long long)a[i] * b[i];

    long long got[n];
    mul_simd_32_to_64_bit(a, b, got, n);

    for (int i = 0; i < n; i++)
        assert(expected[i] == got[i]);

    cout << "Found correct mulitplied array" << endl;
}

int main()
{
    // cpuSupport();
    // basic();
    // gcc_vector_extension();
    // sum_simd_test();
    // predicated_sum_test();
    // find_test();
    // count_test();
    mul_simd_test();
}
