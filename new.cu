#include <cstdio>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <fstream>
#include <string>

#include <numeric>
#include <sys/time.h>

#include <cstdarg>
#include <algorithm>
#include <cassert>

#include "utils.h"


#define VLEN 5 // The decimal buffer length
#define CAP 1000000000 // The capacity of one decimal buffer word
#define CAP_POW 9 // The capacity power of one decimal buffer word
// #define CAP 100
// #define CAP_POW 2


/*
 * decimal: representing a decimal number using VLEN int32_t;
 *   sign: 0: positive, 1: negative
 *   prec: the precision of the decimal;
 *   frac: the fraction part of the decimal;
 *   v[VLEN]: the values; v[0] is the least significant part;
 */
struct decimal
{
    uint32_t sign:1;
    int32_t prec:8;
    int32_t frac:7;
    int32_t v[VLEN];

    decimal(int32_t v, int f);
    decimal(std::string, int);
    decimal();

    friend std::ostream& operator<<(std::ostream&, const decimal &);
};

/*
 * initialize a decimal using int32_t
 *   value: the initialization number, without decimal point
 *       f: the decimal point position
 */
decimal::decimal(int32_t value, int f = 0)
{

    sign = value < 0;
    frac = f;
    if(value < 0) value = -value;
    for(int i = 0; i < VLEN; i++){
        v[i] = value % CAP;
        value /= CAP;
    }
}

decimal::decimal()
{
    decimal(0);
}

/*
 * initialize a decimal using std::string
 *   str: the initialzation number
 *   pow: change the representation according to the power, e.g., ("1", -2) would be v[0]=100, frac = 2
 */
decimal::decimal(std::string str, int pow = 0)
{
    size_t pos;

    memset(v, 0, VLEN*sizeof(int32_t));
    pos = str.find('.');

    if(str[0] == '-'){
        sign = 1;
        str.erase(str.begin());
    }else
        sign = 0;

    if(pos == std::string::npos){
        frac = 0;
    }else{
        frac = -(str.size() - pos - 1);
        str.erase(str.begin() + pos);
    }

    if(pow < 0){
        for(int i = 0; i < -pow; i++)
            str += "0";
        frac += pow;
    }

    int i = 0;
    while(str.size() >= CAP_POW){
        v[i++] = stoi(str.substr(str.size()-CAP_POW, CAP_POW));
        str = str.substr(0, str.size() - CAP_POW);
    }
    v[i] = stoi(str);
}

std::ostream &operator<<(std::ostream& os, const decimal &d)
{
    os << "frac: " << d.frac << ", ";
    os << (d.sign?"-":"+");
    for(int i = VLEN-1; i >= 0; i--)
        os << d.v[i] << (i != 0 ? " " : "");
    return os;
}


/*
 * Add two absolute decimals hold in two consecutive VLEN int32_t buffer
 *     a: the first decimal
 *     b: the second decimal
 *   res: the result decimal. This could be a or b
 *   overflow: indicate if overflow
 */
__host__ __device__ void abs_add(int32_t *a, int32_t *b, int32_t *res, int32_t &overflow)
{
    overflow = 0;
    for(int i = 0; i < VLEN; i++){
        res[i] = a[i] + b[i] + overflow;
        overflow = res[i] / CAP;
        res[i] = res[i] % CAP;
    }
}

/*
 * Compare two absolute decimals hold in two consecutive VLEN int32_t buffer
 *    a: the first decimal
 *    b: the second decimal
 * Return values: >0 (a > b), 0 (a == b), <0 (a < b)
 *
 */
__host__ __device__ int32_t abs_cmp(int32_t *a, int32_t *b)
{
    int32_t res = 0;
#pragma unroll
    for (int i = VLEN - 1; i >= 0 && res == 0; i--) {
        res = a[i] - b[i];
    }
    return res;
}

/*
 * Substract an absolute decimal (b) from an absolute decimal (a)
 */
__host__ __device__ void abs_sub(int32_t *a, int32_t *b, int32_t *res)
{
    int32_t *sub1, *sub2;
    int32_t r = abs_cmp(a, b);
    if(r >= 0){
        sub1 = a;
        sub2 = b;
    }else{
        sub1 = b;
        sub2 = a;
    }

    int32_t carry = 0;
    for(int i = 0; i < VLEN; i++){
        res[i] = sub1[i] + CAP - sub2[i] - carry;
        carry = !(res[i] / CAP);
        res[i] = res[i] % CAP;
    }
}

/*
 * Add two decimals, the fraction length should be the same before adding
 */
__host__ __device__ void var_add(struct decimal &v1, struct decimal &v2, struct decimal &res)
{
    int32_t overflow = 0;
    res.prec = v1.prec;
    res.frac = v1.frac;
    res.sign = v1.sign;
    if(v1.sign ^ v2.sign == 0){
        abs_add(v1.v, v2.v, res.v, overflow);
    }else{
        abs_sub(v1.v, v2.v, res.v);
        res.sign = (abs_cmp(v1.v, v2.v) > 0 && v1.sign) || (abs_cmp(v1.v, v2.v) < 0 && !v1.sign);
    }
}

__host__ __device__
void abs_lshift(int32_t *a, int len, int n, int32_t *res)
{
}

/*
 * right shift an absolute decimal
 *     a: the input decimal
 *   len: the length of the input buffer
 *     n: how many digits it shifts
 *   res: the output decimal, could be same as the input buffer
 */
__host__ __device__
void abs_rshift(int32_t *a, int len, int n, int32_t *res)
{
    int32_t rword = n / CAP_POW;
    int32_t rbit = n % CAP_POW;
    int32_t rd = 1;
    int32_t rl = 1;

    for(int i = 0; i < rbit; i++) rd *= 10;
    for(int i = 0; i < CAP_POW - rbit; i++) rl *= 10;
    for(int i = 0; i < len - rword - 1; i++){
        res[i] = a[rword + i] / rd + a[rword + i + 1] % rd * rl;
    }
    res[len - rword - 1] = a[len - 1] / rd;
    for(int i = len - rword; i < len; i++)
        res[i] = 0;
}

/*
 * multiply two absolute decimals
 *     a: the first decimal
 *     b: the second decimal
 *   res: the result decimal. The buffer size should be VLEN*2
 */
__host__ __device__
void abs_mul(int32_t *a, int32_t *b, int32_t *res)
{
    int64_t temp;
    int32_t carry;
    for(int i = 0; i < VLEN * 2; i++)
        res[i] = 0;
    for(int i = 0; i < VLEN; i++){
        carry = 0;
        for(int j = 0; j < VLEN; j++){
            temp = (int64_t)a[i] * b[j] + res[i+j] + carry;

            carry = temp / CAP;
            res[i+j] = temp % CAP;
        }
        res[i+VLEN] = carry;
    }
}

/*
 * multiply two decimals
 */
__host__ __device__
void var_mul(struct decimal &v1, struct decimal &v2, struct decimal &res)
{
    int32_t overflow = 0;
    res.prec = v1.prec;
    res.frac = v1.frac + v2.frac;
    res.sign = v1.sign ^ v2.sign;

    int32_t inner_res[VLEN*2];
    abs_mul(v1.v, v2.v, inner_res);

    //abs_rshift(inner_res, VLEN*2, res.prec - (v2.frac + v1.frac), inner_res); // or abs_lfshit
    for(int i = 0; i < VLEN; i++)
        res.v[i] = inner_res[i];

    for(int i = VLEN; i < 2*VLEN; i++)
        overflow = (overflow || inner_res[i]);
}

/*
 * multiply two decimals, and set the fraction of the result to frac
 */
__host__ __device__
void var_mul(struct decimal &v1, struct decimal &v2, struct decimal &res, int frac)
{
    int32_t overflow = 0;
    res.prec = v1.prec;
    res.frac = v1.frac + v2.frac;
    res.sign = v1.sign ^ v2.sign;

    int32_t inner_res[VLEN*2];
    abs_mul(v1.v, v2.v, inner_res);

    if(res.frac < frac){
        abs_rshift(inner_res, VLEN*2, frac - res.frac, inner_res);
        res.frac = frac;
    }else if(res.frac > frac){
        abs_lshift(inner_res, VLEN*2, frac - res.frac, inner_res);
        res.frac = frac;
    }

    for(int i = 0; i < VLEN; i++)
        res.v[i] = inner_res[i];

    for(int i = VLEN; i < 2*VLEN; i++)
        overflow = (overflow || inner_res[i]);
}


/*
 * accumulate decimals in the threadblock
 */
__global__ void accumulate(decimal *a, int n, decimal *res)
{
    extern __shared__ decimal sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        memcpy(sdata+tid, a+i, sizeof(decimal));
    else
        memset(sdata+tid, 0, sizeof(decimal));
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s){
            var_add(sdata[tid], sdata[tid+s], sdata[tid]);
        }
        __syncthreads();
    }

    if(tid == 0) memcpy(res+blockIdx.x, sdata, sizeof(decimal));
}

/*
 * calculate l_extendedprice*(1-l_discount)
 */
__global__ void mul_discount(decimal *e, decimal *d, int n, decimal *one)
{
    extern __shared__ decimal sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    memset(sdata+tid, 0, sizeof(decimal));
    if(tid == 0)
        memcpy(&sdata[blockDim.x], one, sizeof(decimal));
    __syncthreads();

    d[i].sign = 1;
    var_add(sdata[blockDim.x], d[i], sdata[tid]);
    var_mul(e[i], sdata[tid], sdata[tid]);

    memcpy(e+i, sdata+tid, sizeof(decimal));
}

/*
 * calculate l_extendedprice*(1-l_discount)*(1+l_tax)
 */
__global__ void mul_discount_tax(decimal *e, decimal *d, decimal *t, int n, decimal *one)
{
    extern __shared__ decimal sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    memset(sdata+tid*2, 0, sizeof(decimal)*2);
    if(tid == 0)
        memcpy(&sdata[blockDim.x*2], one, sizeof(decimal));
    __syncthreads();

    decimal &tmpRes = sdata[tid*2];
    decimal &tmpRes2 = sdata[tid*2+1];

    d[i].sign = 1;
    var_add(sdata[blockDim.x*2], d[i], tmpRes);
    var_add(sdata[blockDim.x*2], t[i], tmpRes2);
    var_mul(e[i], tmpRes, tmpRes);
    var_mul(tmpRes, tmpRes2, tmpRes2);

    memcpy(e+i, sdata+tid*2+1, sizeof(decimal));
}




int main(int argc, char *argv[])
{
    CPUTimer cpuTimer;
    double cpuPerf;

    GPUTimer gpuTimer;
    float gpuPerf;

    double cpuPerfTotal = 0.0;
    float gpuPerfTotal = 0.0;

    const char *datafile = "/data/tpch/data/scale_1/csv/org/lineitem.tbl";
    //const char *datafile = "/data/tpch/tpch100/lineitem.tbl";


    /*
     * l_quantity(5): decimal 36,0
     * l_extendedprice(6): decimal 36,2
     * l_discount(7): decimal 36,2
     * l_tax(8): decimal 36,2
     */
    std::vector<std::string> q_str;
    std::vector<std::string> e_str;
    std::vector<std::string> d_str;
    std::vector<std::string> t_str;

    // load quantity, extendedprice, discount, and tax from lineitem
    cpuPerf = cpuTimer.timing( [&](){
            readLines(datafile, (uint64_t)-1, [&](std::string l) {
                    extractFields(l, {4, 5, 6, 7}, 0, q_str, e_str, d_str, t_str);
                });
        });
    printf("Read file complete! %lf ms\n", cpuPerf);
    printf("  l_quantity.size() is %lu, l_extendedprice.size() is %lu, l_discount.size() is %lu, l_tax.size() is %lu\n", q_str.size(), e_str.size(), d_str.size(), t_str.size());


    decimal *q_cpu, *q_gpu;
    decimal *e_cpu, *e_gpu;
    decimal *d_cpu, *d_gpu;
    decimal *t_cpu, *t_gpu;


    // allocate memory on both GPU and CPU for holding decimals transformed from the string arrays
    auto allocate = [](std::vector<std::string> &strs, decimal **cpu, decimal **gpu) {
        size_t free, total;
        gpuErrchk( cudaMemGetInfo(&free, &total) );
        //printf("Device Memory: %lu/%lu MB\n", free / (1024 * 1024), total / (1024 * 1024));

        size_t size = sizeof(decimal) * strs.size();
        printf("    allocate %lf/%lf MB on CPU and GPU...\n", size / (1024 * 1024.0), free / (1024 * 1024.0));
        if(size > free){
            printf("Failed to allocate memory %lu (%lf MB), free: %lu\n", size, size / (1024 * 1024.0), free);
            exit(-1);
        }

        *cpu = (decimal *)malloc(sizeof(decimal) * strs.size());
        gpuErrchk( cudaMalloc((void **)gpu, sizeof(decimal) * strs.size()) );

        for(int i = 0; i < strs.size(); i++)
            (*cpu)[i] = decimal(strs[i]);
        gpuErrchk( cudaMemcpy(*gpu, *cpu, sizeof(decimal) * strs.size(), cudaMemcpyHostToDevice) );
    };

    // cpuPerf = cpuTimer.timing( [&](){
    //         allocate(q_str, &q_cpu, &q_gpu);
    //         allocate(e_str, &e_cpu, &e_gpu);
    //         allocate(d_str, &d_cpu, &d_gpu);
    //         allocate(t_str, &t_cpu, &t_gpu);
    //     });
    // printf("Load data complete! %lf ms\n", cpuPerf);



    decimal zero(0);
    decimal sum_cpu(0);
    size_t tupleNr = q_str.size();

    auto setZeroCpu = [&](decimal &d) {
        memcpy(&d, &zero, sizeof(decimal));
    };

    assert(q_str.size() == e_str.size());
    assert(e_str.size() == d_str.size());
    assert(d_str.size() == t_str.size());

    // thread number in a threadblock
    int threadNr = 256;
    size_t resNr = (tupleNr - 1) / threadNr + 1;

    decimal *sum_gpu;
    auto setZeroGpu = [&](decimal *d, size_t n) {
        for(int i = 0; i < n; i++)
            gpuErrchk( cudaMemcpy(d + i, &zero, sizeof(decimal), cudaMemcpyHostToDevice) );
    };
    decimal sum_res;


    // sum(l_quanlity)
    printf("sum(l_quanlity) tupleNr=%lu\n", tupleNr);

    cpuPerf = cpuTimer.timing( [&](){
            allocate(q_str, &q_cpu, &q_gpu);
        });
    printf("  Load data complete! %lf ms\n", cpuPerf);


    printf("  accumulation in decimal (CPU):");
    setZeroCpu(sum_cpu);
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++)
                var_add(q_cpu[i], sum_cpu, sum_cpu);
        });

    std::cout << sum_cpu;
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    gpuErrchk( cudaMalloc((void **)&sum_gpu, sizeof(decimal) * resNr) );
    setZeroGpu(sum_gpu, resNr);

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            decimal *_q_gpu = q_gpu;
            decimal *_sum_gpu = sum_gpu;
            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(decimal)*threadNr>>>(_q_gpu, _tupleNr, _sum_gpu);
                decimal *tmp = _q_gpu;
                _q_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _q_gpu, sizeof(decimal), cudaMemcpyDeviceToHost) );
        });
    cudaDeviceSynchronize();


    std::cout << sum_res;
    printf(" %f ms\n", gpuPerf);
    gpuPerfTotal += gpuPerf;

    free( q_cpu );
    gpuErrchk( cudaFree(q_gpu) );



    // sum(l_extendedprice)
    printf("sum(l_extendedprice) tupleNr=%lu\n", tupleNr);
    cpuPerf = cpuTimer.timing( [&](){
            allocate(e_str, &e_cpu, &e_gpu);
        });
    printf("  Load data complete! %lf ms\n", cpuPerf);

    printf("  accumulation in decimal (CPU):");
    setZeroCpu(sum_cpu);

    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++)
                var_add(e_cpu[i], sum_cpu, sum_cpu);
        });

    std::cout << sum_cpu;
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    setZeroGpu(sum_gpu, resNr);
    setZeroCpu(sum_res);

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            decimal *_e_gpu = e_gpu;
            decimal *_sum_gpu = sum_gpu;
            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(decimal)*threadNr>>>(_e_gpu, _tupleNr, _sum_gpu);
                decimal *tmp = _e_gpu;
                _e_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _e_gpu, sizeof(decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << sum_res;
    printf(" %f ms\n", gpuPerf);
    gpuPerfTotal += gpuPerf;

    free( e_cpu );
    gpuErrchk( cudaFree(e_gpu) );


#if 1
    // sum(l_extendedprice*(1-l_discount))
    printf("sum(l_extendedprice * (1 - l_discount)) tupleNr=%lu\n", tupleNr);
    cpuPerf = cpuTimer.timing( [&](){
            allocate(e_str, &e_cpu, &e_gpu);
            allocate(d_str, &d_cpu, &d_gpu);
        });
    printf("  Load data complete! %lf ms\n", cpuPerf);

    decimal one_cpu = decimal("1", -2);
    decimal tmpRes = decimal("0");

    printf("  accumulation in decimal (CPU):");
    setZeroCpu(sum_cpu);
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++) {
                d_cpu[i].sign = 1;
                var_add(one_cpu, d_cpu[i], tmpRes);
                var_mul(e_cpu[i], tmpRes, tmpRes);
                var_add(tmpRes, sum_cpu, sum_cpu);
            }
        });

    std::cout << sum_cpu;
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    setZeroGpu(sum_gpu, resNr);
    setZeroCpu(sum_res);
    decimal *one_gpu;
    gpuErrchk( cudaMalloc((void **)&one_gpu, sizeof(decimal)) );
    gpuErrchk( cudaMemcpy(one_gpu, &one_cpu, sizeof(decimal), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(e_gpu, e_cpu, sizeof(decimal) * tupleNr, cudaMemcpyHostToDevice) );

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            decimal *_e_gpu = e_gpu;
            decimal *_d_gpu = d_gpu;
            decimal *_sum_gpu = sum_gpu;

            mul_discount<<<_resNr, threadNr, sizeof(decimal)*(threadNr + 1)>>>(_e_gpu, _d_gpu, _tupleNr, one_gpu);


            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(decimal)*threadNr>>>(_e_gpu, _tupleNr, _sum_gpu);
                decimal *tmp = _e_gpu;
                _e_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _e_gpu, sizeof(decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << sum_res;
    printf(" %f ms\n", gpuPerf);
    gpuPerfTotal += gpuPerf;
    free( e_cpu );
    gpuErrchk( cudaFree(e_gpu) );
    free( d_cpu );
    gpuErrchk( cudaFree(d_gpu) );


    // sum(l_extendedprice*(1-l_discount)*(1+l_tax))
    printf("sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) tupleNr=%lu\n", tupleNr);
    cpuPerf = cpuTimer.timing( [&](){
            allocate(e_str, &e_cpu, &e_gpu);
            allocate(d_str, &d_cpu, &d_gpu);
            allocate(t_str, &t_cpu, &t_gpu);
        });
    printf("  Load data complete! %lf ms\n", cpuPerf);

    tmpRes = decimal("0");
    decimal tmpRes2 = decimal("0");

    printf("  accumulation in decimal (CPU):");
    setZeroCpu(sum_cpu);
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++) {
                d_cpu[i].sign = 1;
                var_add(one_cpu, d_cpu[i], tmpRes);
                var_add(one_cpu, t_cpu[i], tmpRes2);
                var_mul(e_cpu[i], tmpRes, tmpRes);
                var_mul(tmpRes, tmpRes2, tmpRes2);
                var_add(tmpRes2, sum_cpu, sum_cpu);
            }
        });
    std::cout << sum_cpu;
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    setZeroGpu(sum_gpu, resNr);
    setZeroCpu(sum_res);

    gpuErrchk( cudaMemcpy(e_gpu, e_cpu, sizeof(decimal) * tupleNr, cudaMemcpyHostToDevice) );
    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            decimal *_e_gpu = e_gpu;
            decimal *_d_gpu = d_gpu;
            decimal *_t_gpu = t_gpu;
            decimal *_sum_gpu = sum_gpu;

            mul_discount_tax<<<_resNr, threadNr, sizeof(decimal)*(2*threadNr + 1)>>>(_e_gpu, _d_gpu, _t_gpu, _tupleNr, one_gpu);

            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(decimal)*threadNr>>>(_e_gpu, _tupleNr, _sum_gpu);
                decimal *tmp = _e_gpu;
                _e_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _e_gpu, sizeof(decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << sum_res;
    printf(" %f ms\n", gpuPerf);
    gpuPerfTotal += gpuPerf;

    free( e_cpu );
    gpuErrchk( cudaFree(e_gpu) );
    free( d_cpu );
    gpuErrchk( cudaFree(d_gpu) );
    free( t_cpu );
    gpuErrchk( cudaFree(t_gpu) );
#endif


    // avg(l_discount)
    printf("avg(l_discount) tupleNr=%lu\n", tupleNr);
    cpuPerf = cpuTimer.timing( [&](){
            allocate(d_str, &d_cpu, &d_gpu);
        });
    printf("  Load data complete! %lf ms\n", cpuPerf);


    printf("  accumulation in decimal (CPU):");
    setZeroCpu(sum_cpu);
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++)
                var_add(d_cpu[i], sum_cpu, sum_cpu);
        });

    std::cout << sum_cpu;
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    setZeroGpu(sum_gpu, resNr);
    setZeroCpu(sum_res);

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            decimal *_d_gpu = d_gpu;
            decimal *_sum_gpu = sum_gpu;
            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(decimal)*threadNr>>>(_d_gpu, _tupleNr, _sum_gpu);
                decimal *tmp = _d_gpu;
                _d_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _d_gpu, sizeof(decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << sum_res;
    printf(" %f ms\n", gpuPerf);
    gpuPerfTotal += gpuPerf;

    free( d_cpu );
    gpuErrchk( cudaFree(d_gpu) );


    printf("Time on CPU: %lf ms\n", cpuPerfTotal);
    printf("Time on GPU: %f ms\n", gpuPerfTotal);


    return 0;
}
