#include <stdio.h>
#include <string.h>
#include <iostream>

#include <sys/time.h>
#include <functional>
#include <float.h>
#include <assert.h>
#include <sys/time.h>

#include "decimal.h"

#include <cuda_runtime.h>
#include <numeric>
#include <sys/time.h>

#include <functional>
#include <cstdarg>
#include <algorithm>
#include <cassert>

__global__ void vector_add(int *a, int *b, size_t n, int *out)
{
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = offset; i < n; i += stride){
        printf("%d (i = %d): %d, %d\n", offset, i, a[i], b[i]);
        out[i] = a[i] + b[i];
    }//

}

#define N 10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class CPUTimer {
public:
    struct timeval st, ed;

    template <typename Func>
    double timing(Func func){
        gettimeofday(&st, 0);
        func();
        gettimeofday(&ed, 0);
        return (1000000.0*(ed.tv_sec-st.tv_sec) + ed.tv_usec-st.tv_usec)/1000.0;
    }
};

class GPUTimer {
public:
    cudaEvent_t start, stop;

    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    virtual ~GPUTimer() { }

    template <typename Func>
    float timing(Func func) {
        float perf;

        cudaEventRecord(start);

        func();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&perf, start, stop);

        return perf;
    }
};





#include <functional>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

std::string origin(std::string str)
{
    return str;
}

int mul100Int(std::string str)
{
    double res = std::stod(str);
    return res*100;
}

template <typename T>
void retrieveData(const char *datafile, int n, std::function<T (std::string)> func, std::vector<T> &data)
{
    std::ifstream myfile;
    std::string line;
    size_t found, last_found;
    myfile.open(datafile);
    if(myfile.is_open()){

        while(std::getline(myfile, line)){
            found = last_found = 0;
            for(int i = 0; i < n; i++) {
                found = line.find('|', last_found);
                if(found == std::string::npos){
                    std::cout << "failed to find the " << i << "th '|'" << std::endl;
                    return;
                }
                if(i == n - 1)
                    data.push_back(func(line.substr(last_found, found - last_found)));
                last_found = found+1;
            }
        }
        myfile.close();
    }

}







int bytesForDecimal(int p, int d)
{
    int n = 0;
    while(p > 0) {
        if(p >= 9){
            p -= 9;
            n += 4;
        }else{
            if(p > 6)
                n += 4;
            else if(p > 4)
                n += 3;
            else if(p > 2)
                n += 2;
            else if(p > 0)
                n += 1;
            p = 0;
        }
    }
    return n;
}

double timeElapsed(struct timeval &st, struct timeval &ed)
{
    long sec = ed.tv_sec - st.tv_sec;
    long msec = ed.tv_usec - st.tv_usec;
    double elapsed = sec + msec * 1e-6;
    return elapsed;
}

/*
void genRandDecimal(char *head, int n, int nr)
{
    int leftover = n % 4;
    int fourBytes = n / 4;
    int r, mask;
    for(int i = 0; i < nr; i++) {
        for(int j = 0; j < fourBytes; j++) {
            r = rand();
            memcpy( head + i * n + leftover + j * 4, &r, sizeof(int) );
        }
        if(leftover) {
            if(leftover == 1)
                mask = 0x000000ff;
            else if(leftover == 2)
                mask = 0x0000ffff;
            else if(leftover == 3)
                mask = 0x00ffffff;
            r = rand() % mask;
            char buf[4];
            memcpy(buf, &r, 4);
            for(int j = 0; j < leftover; j++)
                head[i * n + j] = buf[4 - leftover + j];
    }
}
*/

template<typename T>
class ArithmeticTest {
public:
    ArithmeticTest(int32_t n);
    virtual ~ArithmeticTest();

    void runAdd();
    void runSub();
    void runMul();
    void runDiv();
private:
    T *src1;
    T *src2;
    T *dst;

    int32_t _n;
};


template<typename T>
ArithmeticTest<T>::ArithmeticTest(int32_t n) : _n(n)
{
    int res;
    if(res = posix_memalign((void **)&src1, 8, _n * sizeof(T)))
        printf("Error in allocating memory for src1: %s\n", strerror(res));
    if(res = posix_memalign((void **)&src2, 8, _n * sizeof(T)))
        printf("Error in allocating memory for src2: %s\n", strerror(res));
    if(res = posix_memalign((void **)&dst, 8, _n * sizeof(T)))
        printf("Error in allocating memory for dst: %s\n", strerror(res));
}

template<typename T>
ArithmeticTest<T>::~ArithmeticTest()
{
    if(src1 != nullptr){
        free(src1);
        src1 = nullptr;
    }
    if(src2 != nullptr){
        free(src2);
        src2 = nullptr;
    }
    if(dst != nullptr){
        free(dst);
        dst = nullptr;
    }
}

template<typename T>
void ArithmeticTest<T>::runAdd()
{
    for(int32_t i = 0; i < _n; i++)
        dst[i] = src1[i] + src2[i];
}

template<typename T>
void ArithmeticTest<T>::runSub()
{
    for(int32_t i = 0; i < _n; i++)
        dst[i] = src1[i] - src2[i];
}

template<typename T>
void ArithmeticTest<T>::runMul()
{
    for(int32_t i = 0; i < _n; i++)
        dst[i] = src1[i] * src2[i];
}

template<typename T>
void ArithmeticTest<T>::runDiv()
{
    for(int32_t i = 0; i < _n; i++)
        dst[i] = src1[i] / src2[i];
}


class ProfileFunc {
public:
    ProfileFunc() {}
    virtual ~ProfileFunc() { }


    static double timeElapsed(struct timeval &st, struct timeval &ed) {
            long sec = ed.tv_sec - st.tv_sec;
            long msec = ed.tv_usec - st.tv_usec;
            double elapsed = sec + msec * 1e-6;
            return elapsed;
    }

    template<typename T>
    static double profile( void (ArithmeticTest<T>::*func)(void), ArithmeticTest<T> &testClass ) {
            timeval st, ed;
            gettimeofday(&st, 0);
            (testClass.*func)();
            gettimeofday(&ed, 0);
            return ProfileFunc::timeElapsed(st, ed);
    }
};

int test() {
    throw 10;
    return 100;
}

struct Intg{
    int32_t v;

    friend Intg operator+( const Intg& left, const Intg& right );
    Intg& operator+=(const Intg& d);
    Intg& operator=( int32_t i );
};

Intg &Intg::operator=( int32_t i ) {
    v = i;
    return *this;
}

Intg &Intg::operator+=(const Intg &i) {
    //Intg added(i);
    v += i.v;
    return *this;
}

Intg operator+( const Intg& left, const Intg& right ){
    Intg tmp(left);
    return tmp += right;
}

int GetNeedBits(int base10Precision) {
    int len = base10Precision / DIG_PER_INT32 * 32;
    switch (base10Precision % DIG_PER_INT32) {
    case 0:
        len += 0;
        break;
    case 1:
        len += 4;
        break;
    case 2:
        len += 7;
        break;
    case 3:
        len += 10;
        break;
    case 4:
        len += 14;
        break;
    case 5:
        len += 17;
        break;
    case 6:
        len += 20;
        break;
    case 7:
        len += 24;
        break;
    case 8:
        len += 27;
        break;
    }
    return len;
}

#define PER_DEC_MAX_SCALE 1000000000  //每个int的值不能大于此值

int32_t GetPowers10(int i) {
    int32_t res = 1;
    switch (i) {
    case 0:
        res = 1;
        break;
    case 1:
        res = 10;
        break;
    case 2:
        res = 100;
        break;
    case 3:
        res = 1000;
        break;
    case 4:
        res = 10000;
        break;
    case 5:
        res = 100000;
        break;
    case 6:
        res = 1000000;
        break;
    case 7:
        res = 10000000;
        break;
    case 8:
        res = 100000000;
        break;
    case 9:
        res = PER_DEC_MAX_SCALE;
        break;
    default:
        break;
    }
    return res;
}

__host__ __device__ void test_int32_arith()
{
    int32_t a = INT32_MIN;
    printf("int32_t: a is %d, a/-1 = %d, 0-a = %d\n", a, a/-1, 0-a);
}

__global__ void test_int32_arith_device()
{
    test_int32_arith();
}

#define TEST_INT32_ARITH_HOST() test_int32_arith()
#define TEST_INT32_ARITH_DEVICE() test_int32_arith_device<<<1, 1>>>()

__host__ __device__ void test_int64_arith()
{
    int64_t a = INT64_MIN;
    printf("int64_t: a is %ld, a/-1 = %ld, 0-a = %ld\n", a, a/-1, 0-a);
}

__global__ void test_int64_arith_device()
{
    test_int64_arith();
}

#define TEST_INT32_ARITH_HOST() test_int32_arith()
#define TEST_INT32_ARITH_DEVICE() test_int32_arith_device<<<1, 1>>>()

#define TEST_INT64_ARITH_HOST() test_int64_arith()
#define TEST_INT64_ARITH_DEVICE() test_int64_arith_device<<<1, 1>>>()

using namespace aries_acc;

#define ONEONE     1.111111111111111
#define TWOTWO     2.222222222222222
#define THREETHREE 3.333333333333333

// __global__ void test()
// {
//     // int offset = blockDim.x * blockIdx.x + threadIdx.x;
//     // int stride = gridDim.x * blockDim.x;

//     // for(int i = offset; i < n; i += stride){

//     // }
//     printf("hello!\n");

// }

__global__ void cuda_hello (char *d1, char *d2)
{
    // Decimal _dec1("1.1111");
    // Decimal _dec2("2.2222");
    Decimal &_dec1 = *(Decimal *)d1;
    Decimal &_dec2 = *(Decimal *)d2;
    Decimal _dec3;
    _dec3 = _dec1 / _dec2;

    char result[256];
    _dec3.GetDecimal(result);
    //(Decimal *)d1.GetDecimal(result);
    printf("Hello World from GPU! %s\n", result);
}

__global__ void cuda_add (char *d1, char *d2)
{
    Decimal &_dec1 = *(Decimal *)d1;
    Decimal &_dec2 = *(Decimal *)d2;
    for(int i = 0; i < 1000; i++)
        _dec1 += _dec2;
}

__global__ void cuda_sub (char *d1, char *d2)
{
    Decimal &_dec1 = *(Decimal *)d1;
    Decimal &_dec2 = *(Decimal *)d2;
    _dec1 -= _dec2;
}

__global__ void cuda_mul (char *d1, char *d2)
{
    Decimal &_dec1 = *(Decimal *)d1;
    Decimal &_dec2 = *(Decimal *)d2;
    _dec1 *= _dec2;
}

__global__ void cuda_div (char *d1, char *d2)
{
//    clock_t st, ed;
//    st = clock();
    Decimal &_dec1 = *(Decimal *)d1;
    Decimal &_dec2 = *(Decimal *)d2;
    _dec1 /= _dec2;
//    ed = clock();
//    printf("diff is %d\n", ed - st);
    // double elapsed = ((double)(ed - st)) / CLOCKS_PER_SEC;
    // printf("diff is %d, elapsed = %f\n", ed -st, elapsed);
}

__global__ void cuda_acc (Decimal *d, size_t n, Decimal *res)
{
    Decimal &Res = *res;
    for(size_t i = 0; i < n; i++)
        Res += d[i];
}

struct dec
{
    // uint16_t sign:1;
    // uint16_t prec:8;
    // uint16_t frac:7;
    int32_t v[3];
};

__global__ void cuda_pr (Decimal *a, Decimal *b, Decimal *res)
{
    Decimal &ma = *a;
    Decimal &mb = *b;
    Decimal &mres = *res;
    for(int i = 0; i < 10000; i++)
        mres = ma * mb;

}


__global__ void accumulate(Decimal *a, int n, Decimal *res)
{
    extern __shared__ Decimal sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        memcpy(sdata+tid, a+i, sizeof(Decimal));
    else
        memset(sdata+tid, 0, sizeof(Decimal));
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if(tid == 0) memcpy(res+blockIdx.x, sdata, sizeof(Decimal));
}

__global__ void mul_discount(Decimal *e, Decimal *d, int n)
{
    extern __shared__ Decimal sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    Decimal &tmpRes = sdata[tid];
    tmpRes = 0;
    __syncthreads();

    tmpRes = 1 - d[i];
    tmpRes *= e[i];

    memcpy(e+i, &tmpRes, sizeof(Decimal));
}


__global__ void mul_discount_tax(Decimal *e, Decimal *d, Decimal *t, int n)
{
    extern __shared__ Decimal sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    Decimal &tmpRes = sdata[tid*2];
    Decimal &tmpRes2 = sdata[tid*2 + 1];
    tmpRes = 0;
    tmpRes2 = 0;
    __syncthreads();

    tmpRes = 1 - d[i];
    tmpRes2 = 1 + t[i];
    tmpRes *= tmpRes2;
    tmpRes *= e[i];

    memcpy(e+i, &tmpRes, sizeof(Decimal));
}




#include <string>
//#define PR_PROP
#define PR_PROP_DIV


template <typename LineProc>
uint64_t readLines(const char *datafile, uint64_t lnr, LineProc lineProc)
{
    std::ifstream myfile;
    std::string line;
    uint64_t n = 0;

    myfile.open(datafile);
    if(myfile.is_open()){
        while(std::getline(myfile, line) && n++ < lnr){
            lineProc(line);
        }
        myfile.close();
        return n;
    }else{
        std::cerr << "Open file " << datafile << " error!\n";
        return 0;
    }
}

void extractFields(std::string line, std::vector<int> idx, size_t start, std::vector<std::string> &container)
{
    size_t found = start, last_found = start;
    int i = 0;
    do{
        found = line.find('|', last_found);
        auto pos = std::find(idx.begin(), idx.end(), i);
        if(pos != idx.end()){
            container.push_back(line.substr(last_found, found - last_found));
            break;
        }

        i++;
        last_found = found+1;
    }while(found != std::string::npos);
}

template <typename FirstContainer, typename... Containers>
void extractFields(std::string line, std::vector<int> idx, size_t start, FirstContainer &firstContainer, Containers&... containers)
{
    size_t found = start, last_found = start;
    int i = 0;
    do{
        found = line.find('|', last_found);
        auto pos = std::find(idx.begin(), idx.end(), i);
        if(pos != idx.end()){
            firstContainer.push_back(line.substr(last_found, found - last_found));
            for(auto iter = idx.begin(); iter != idx.end(); iter++)
                if(iter != pos)
                    (*iter) = (*iter) - (*pos) - 1;

            idx.erase(pos);
            extractFields(line, idx, found+1, containers...);
            break;
        }

        i++;
        last_found = found+1;
    }while(found != std::string::npos);
}


int main(int argc, char *argv[])
{
    CPUTimer cpuTimer;
    double cpuPerf;

    GPUTimer gpuTimer;
    float gpuPerf;

    double cpuPerfTotal = 0.0;
    float gpuPerfTotal = 0.0;

    char prRes[256];

    const char *datafile = "/data/tpch/data/scale_1/csv/org/lineitem.tbl";
    //const char *datafile = "/data/tpch/tpch100/lineitem.tbl";

    std::vector<std::string> q_str;
    std::vector<std::string> e_str;
    std::vector<std::string> d_str;
    std::vector<std::string> t_str;

#define MPREC 27
#define MFRAC 2
    cpuPerf = cpuTimer.timing( [&](){
            readLines(datafile, (uint64_t)-1, [&](std::string l) {
                    extractFields(l, {4, 5, 6, 7}, 0, q_str, e_str, d_str, t_str);
                });
        });
    printf("Read file complete! %lf ms\n", cpuPerf);
    printf("  l_quantity.size() is %lu, l_extendedprice.size() is %lu, l_discount.size() is %lu, l_tax.size() is %lu\n", q_str.size(), e_str.size(), d_str.size(), t_str.size());

    Decimal *q_cpu, *q_gpu;
    Decimal *e_cpu, *e_gpu;
    Decimal *d_cpu, *d_gpu;
    Decimal *t_cpu, *t_gpu;


    auto allocate = [](std::vector<std::string> &strs, Decimal **cpu, Decimal **gpu) {
        size_t free, total;
        gpuErrchk( cudaMemGetInfo(&free, &total) );

        size_t size = sizeof(Decimal) * strs.size();
        printf("    allocate %lf/%lf MB on CPU and GPU...\n", size / (1024 * 1024.0), free / (1024 * 1024.0));
        if(size > free){
            printf("Failed to allocate memory %lu (%lf MB), free: %lu\n", size, size / (1024 * 1024.0), free);
            exit(-1);
        }

        *cpu = (Decimal *)malloc(sizeof(Decimal) * strs.size());
        gpuErrchk( cudaMalloc((void **)gpu, sizeof(Decimal) * strs.size()) );

        for(int i = 0; i < strs.size(); i++)
            (*cpu)[i] = Decimal(MPREC, MFRAC, strs[i].c_str());
        gpuErrchk( cudaMemcpy(*gpu, *cpu, sizeof(Decimal) * strs.size(), cudaMemcpyHostToDevice) );
    };


    Decimal zero(MPREC, MFRAC, "0");
    Decimal sum_cpu(MPREC, MFRAC, "0");

    size_t tupleNr = q_str.size();

    assert(q_str.size() == e_str.size());
    assert(e_str.size() == d_str.size());
    assert(d_str.size() == t_str.size());

    int threadNr = 256;
    size_t resNr = (tupleNr - 1) / threadNr + 1;

    Decimal *sum_gpu;
    auto setZeroGpu = [&](Decimal *d, size_t n) {
        for(int i = 0; i < n; i++)
            gpuErrchk( cudaMemcpy(d + i, &zero, sizeof(Decimal), cudaMemcpyHostToDevice) );
    };
    Decimal sum_res;

    // sum(l_quanlity)
    printf("sum(l_quanlity) tupleNr=%lu\n", tupleNr);

    cpuPerf = cpuTimer.timing( [&](){
            allocate(q_str, &q_cpu, &q_gpu);
        });
    printf("  Load data complete! %lf ms\n", cpuPerf);


    printf("  accumulation in decimal (CPU):");
    sum_cpu = 0;
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++)
                sum_cpu += q_cpu[i];
        });


    sum_cpu.GetDecimal(prRes);
    printf(" %s", prRes);
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    gpuErrchk( cudaMalloc((void **)&sum_gpu, sizeof(Decimal) * resNr) );
    setZeroGpu(sum_gpu, resNr);

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            Decimal *_q_gpu = q_gpu;
            Decimal *_sum_gpu = sum_gpu;
            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(Decimal)*threadNr>>>(_q_gpu, _tupleNr, _sum_gpu);
                Decimal *tmp = _q_gpu;
                _q_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _q_gpu, sizeof(Decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    sum_res.GetDecimal(prRes);
    printf(" %s", prRes);
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
    sum_cpu = 0;

    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++)
                sum_cpu += e_cpu[i];
        });

    sum_cpu.GetDecimal(prRes);
    printf(" %s", prRes);
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    setZeroGpu(sum_gpu, resNr);
    sum_res = 0;

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            Decimal *_e_gpu = e_gpu;
            Decimal *_sum_gpu = sum_gpu;
            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(Decimal)*threadNr>>>(_e_gpu, _tupleNr, _sum_gpu);
                Decimal *tmp = _e_gpu;
                _e_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _e_gpu, sizeof(Decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    sum_res.GetDecimal(prRes);
    printf(" %s", prRes);
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

    Decimal one_cpu = Decimal(MPREC, MFRAC, "1");
    Decimal tmpRes = Decimal(MPREC, MFRAC, "0");

    printf("  accumulation in decimal (CPU):");
    sum_cpu = 0;
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++) {
                sum_cpu += (e_cpu[i] * (1 - d_cpu[i]));
            }
        });

    sum_cpu.GetDecimal(prRes);
    printf(" %s", prRes);
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    setZeroGpu(sum_gpu, resNr);
    sum_res = 0;

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            Decimal *_e_gpu = e_gpu;
            Decimal *_d_gpu = d_gpu;
            Decimal *_sum_gpu = sum_gpu;

            mul_discount<<<_resNr, threadNr, sizeof(Decimal)*(threadNr + 1)>>>(_e_gpu, _d_gpu, _tupleNr);


            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(Decimal)*threadNr>>>(_e_gpu, _tupleNr, _sum_gpu);
                Decimal *tmp = _e_gpu;
                _e_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _e_gpu, sizeof(Decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    sum_res.GetDecimal(prRes);
    printf(" %s", prRes);
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

    printf("  accumulation in decimal (CPU):");
    sum_cpu = 0;
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++)
                sum_cpu += (e_cpu[i] * (1 - d_cpu[i]) * (1 + t_cpu[i]));
        });
    sum_cpu.GetDecimal(prRes);
    printf(" %s", prRes);
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    setZeroGpu(sum_gpu, resNr);
    sum_res = 0;

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            Decimal *_e_gpu = e_gpu;
            Decimal *_d_gpu = d_gpu;
            Decimal *_t_gpu = t_gpu;
            Decimal *_sum_gpu = sum_gpu;

            mul_discount_tax<<<_resNr, threadNr, sizeof(Decimal)*(2*threadNr + 1)>>>(_e_gpu, _d_gpu, _t_gpu, _tupleNr);

            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(Decimal)*threadNr>>>(_e_gpu, _tupleNr, _sum_gpu);
                Decimal *tmp = _e_gpu;
                _e_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _e_gpu, sizeof(Decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    sum_res.GetDecimal(prRes);
    printf(" %s", prRes);
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
    sum_cpu = 0;
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++)
                sum_cpu += d_cpu[i];
        });

    sum_cpu.GetDecimal(prRes);
    printf(" %s", prRes);
    printf(" %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
    setZeroGpu(sum_gpu, resNr);
    sum_res = 0;

    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            Decimal *_d_gpu = d_gpu;
            Decimal *_sum_gpu = sum_gpu;
            while(_tupleNr > 1){
                accumulate<<<_resNr, threadNr, sizeof(Decimal)*threadNr>>>(_d_gpu, _tupleNr, _sum_gpu);
                Decimal *tmp = _d_gpu;
                _d_gpu = _sum_gpu;
                _sum_gpu = tmp;
                _tupleNr = _resNr;
                _resNr = (_tupleNr - 1) / threadNr + 1;
            }
            gpuErrchk( cudaMemcpy(&sum_res, _d_gpu, sizeof(Decimal), cudaMemcpyDeviceToHost) );
        });
    gpuErrchk( cudaDeviceSynchronize() );

    sum_res.GetDecimal(prRes);
    printf(" %s", prRes);
    printf(" %f ms\n", gpuPerf);
    gpuPerfTotal += gpuPerf;

    free( d_cpu );
    gpuErrchk( cudaFree(d_gpu) );


    printf("Time on CPU: %lf ms\n", cpuPerfTotal);
    printf("Time on GPU: %f ms\n", gpuPerfTotal);





#if 0
#define CUDA_EV_TIMER
//#define CPU_TIMER
#ifdef CUDA_EV_TIMER
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float perf;
#endif

    struct timeval st, ed;
    double cpuPerf;


    const char *datafile = "/data/tpch/data/scale_1/csv/org/lineitem.tbl";
    std::vector<std::string> quantity;
    Decimal *quantity_cpu, *quantity_gpu;

    retrieveData<std::string>(datafile, 5, origin, quantity);
    quantity_cpu = (Decimal *)malloc(sizeof(Decimal) * quantity.size());
    cudaMalloc((void **)&quantity_gpu, sizeof(Decimal) * quantity.size());

    for(int i =0; i < quantity.size(); i++)
        quantity_cpu[i] = Decimal(MPREC, MFRAC, quantity[i].c_str());
    cudaMemcpy(quantity_gpu, quantity_cpu, sizeof(Decimal) * quantity.size(), cudaMemcpyHostToDevice);


    std::vector<std::string> extendedprice;
    Decimal *extendedprice_cpu, *extendedprice_gpu;
    retrieveData<std::string>(datafile, 6, origin, extendedprice);
    extendedprice_cpu = (Decimal *)malloc(sizeof(Decimal) * extendedprice.size());
    cudaMalloc((void **)&extendedprice_gpu, sizeof(Decimal) * extendedprice.size());

    for(int i = 0; i< extendedprice.size(); i++)
        extendedprice_cpu[i] = Decimal(MPREC, MFRAC, extendedprice[i].c_str());
    cudaMemcpy(extendedprice_gpu, extendedprice_cpu, sizeof(Decimal) * extendedprice.size(), cudaMemcpyHostToDevice);


    std::vector<std::string> discount;
    Decimal *discount_cpu, *discount_gpu;
    retrieveData<std::string>(datafile, 7, origin, discount);
    discount_cpu = (Decimal *)malloc(sizeof(Decimal) * discount.size());
    cudaMalloc((void **)&discount_gpu, sizeof(Decimal) * discount.size());

    for(int i = 0; i < discount.size(); i++)
        discount_cpu[i] = Decimal(MPREC, MFRAC, discount[i].c_str());
    cudaMemcpy(discount_gpu, discount_cpu, sizeof(Decimal) * discount.size(), cudaMemcpyHostToDevice);


    /*
     * sum(l_quantity)
     */
    Decimal quantity_res(MPREC, MFRAC, "0"), zero(MPREC, MFRAC, "0");
    char prRes[256];
    size_t addNr = quantity.size();

    printf("sum(l_quantity), addNr=%lu:\n", addNr);
    gettimeofday(&st, 0);

    for(int i = 0; i < addNr; i++)
        quantity_res += quantity_cpu[i];

    gettimeofday(&ed, 0);
    cpuPerf = (1000000.0*(ed.tv_sec-st.tv_sec) + ed.tv_usec-st.tv_usec)/1000.0;

    quantity_res.GetDecimal(prRes);
    printf("  Calculation in Decimal (CPU): result = %s, %lf ms\n", prRes, cpuPerf);




    size_t resNr = (addNr-1) / 256 + 1;
    Decimal *quantity_resgpu, *q_res;
    cudaMalloc((void **)&quantity_resgpu, sizeof(Decimal) * resNr);
    for(int i = 0; i < resNr; i++)
        cudaMemcpy(quantity_resgpu + i, &zero, sizeof(Decimal), cudaMemcpyHostToDevice);
    q_res = (Decimal *)malloc(sizeof(Decimal) * resNr);
    cudaEventRecord(start);

    while(addNr > 1){
        accumulate<<<resNr, 256, sizeof(Decimal)*256>>>(quantity_gpu, addNr, quantity_resgpu);
        Decimal *tmp = quantity_gpu;
        quantity_gpu = quantity_resgpu;
        quantity_resgpu = tmp;
        addNr = resNr;
        resNr = (addNr-1)/256 + 1;
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    cudaMemcpy(q_res, quantity_gpu, sizeof(Decimal) * resNr, cudaMemcpyDeviceToHost);
    printf("  Calculation in Decimal (GPU): ");
    for(int i = 0; i < resNr; i++){
        q_res[i].GetDecimal(prRes);
        printf("%s, ", prRes);
    }
    perf = 0;
    cudaEventElapsedTime(&perf, start, stop);
    printf(" %f ms\n", perf);


    /*
     * sum(l_extendedprice*(1-l_discount))
     */
    // 1
    printf("\n");
    addNr = extendedprice.size();
    printf("sum(l_extendedprice * (1 - l_discount)), addNr = %lu:\n", addNr);
    Decimal sum2(36, 2, "0");

    // accumulation on CPU
    printf("  Calculation in Decimal (CPU):");
    gettimeofday(&st, 0);

    for(int i = 0; i < addNr; i++)
        sum2 += extendedprice_cpu[i] * (1 - discount_cpu[i]);


    gettimeofday(&ed, 0);
    cpuPerf = (1000000.0*(ed.tv_sec-st.tv_sec) + ed.tv_usec-st.tv_usec)/1000.0;
    sum2.GetDecimal(prRes);
    printf(" %s, %lf ms\n", prRes, cpuPerf);



    // 2
    printf("  Calculation in Decimal (GPU):");
    Decimal *acc_res, *a_res;
    resNr = (addNr-1) / 256 + 1;

    cudaMalloc((void **)&acc_res, sizeof(Decimal) * resNr);
    for(int i = 0; i < resNr; i++)
        cudaMemcpy(acc_res + i, &zero, sizeof(Decimal), cudaMemcpyHostToDevice);
    a_res = (Decimal *)malloc(sizeof(Decimal) * resNr);

    cudaEventRecord(start);

    mul_discount<<<resNr, 256, sizeof(Decimal)*256>>>(extendedprice_gpu, discount_gpu, addNr);


    while(addNr > 1){
        accumulate<<<resNr, 256, sizeof(Decimal)*256>>>(extendedprice_gpu, addNr, acc_res);
        Decimal *tmp = extendedprice_gpu;
        extendedprice_gpu = acc_res;
        acc_res = tmp;
        addNr = resNr;
        resNr = (addNr-1)/256 + 1;
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    //cudaMemcpy(&sum_cpu, quantity_gpu, sizeof(decimal), cudaMemcpyDeviceToHost);
    //std::cout << sum_cpu;
    cudaMemcpy(a_res, extendedprice_gpu, sizeof(Decimal) * resNr, cudaMemcpyDeviceToHost);
    for(int i = 0; i < resNr; i++){
        a_res[i].GetDecimal(prRes);
        printf("%s, ", prRes);
    }
    perf = 0;
    cudaEventElapsedTime(&perf, start, stop);
    printf(" %f ms\n", perf);



    return 0;
#endif
#if 0
    Decimal mul_a(36, 2, "10000003.11");
    Decimal mul_b(36, 2, "10000003.11");
    Decimal mul_res(36, 2, "0.00");

    Decimal *ma_d, *mb_d, *mres_d;
    cudaMalloc((void **)&ma_d, sizeof(Decimal));
    cudaMalloc((void **)&mb_d, sizeof(Decimal));
    cudaMalloc((void **)&mres_d, sizeof(Decimal));

    cudaMemcpy(ma_d, &mul_a, sizeof(Decimal), cudaMemcpyHostToDevice);
    cudaMemcpy(mb_d, &mul_b, sizeof(Decimal), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    cuda_pr<<<1, 1>>>(ma_d, mb_d, mres_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    cudaMemcpy(&mul_res, mres_d, sizeof(Decimal), cudaMemcpyDeviceToHost);

    cudaFree(ma_d);
    cudaFree(mb_d);
    cudaFree(mres_d);

    char mresStr[256];
    mul_res.GetDecimal(mresStr);

    printf("Mul Result = %s\n", mresStr);

    perf = 0;
    cudaEventElapsedTime(&perf, start, stop);
    printf("event timer %f ms\n", perf);

    return 0;

    //const char *datafile = "/data/tpch/data/scale_1/csv/org/lineitem.tbl";
    std::vector<std::string> nums;

    retrieveData<std::string>(datafile, 6, origin, nums);
    printf("nums.size() = %lu\n", nums.size());
    for(int i = 0; i < 10; i++)
        std::cout << nums[i] << std::endl;

    int64_t acc_result = 0;
    for(auto n : nums)
        acc_result += (std::stod(n) * 100);

    std::cout << "accresult = " << acc_result << std::endl;

    Decimal Res(35, 2, "0");
    Decimal *ds = (Decimal *)malloc(sizeof(Decimal) * nums.size());
    Decimal *ds_d, *Res_d;


    for(int i = 0; i < nums.size(); i++)
        ds[i] = Decimal(35, 2, nums[i].c_str());

    cudaMalloc((void **)&ds_d, sizeof(Decimal) * nums.size());
    cudaMalloc((void **)&Res_d, sizeof(Decimal));

    cudaMemcpy(ds_d, ds, sizeof(Decimal) * nums.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(Res_d, &Res, sizeof(Decimal), cudaMemcpyHostToDevice);


    cudaEventRecord(start);
    //cuda_accumulate<<<1, 1>>>(ds_d, nums.size(), Res_d);
    cuda_acc<<<1, 1>>>(ds_d, nums.size(), Res_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);


    cudaMemcpy(&Res, Res_d, sizeof(Decimal), cudaMemcpyDeviceToHost);

    free(ds);
    cudaFree(ds_d);
    cudaFree(Res_d);

    char Result[256];
    Res.GetDecimal(Result);

    printf("Accumulation Result = %s\n", Result);

    perf = 0;
    cudaEventElapsedTime(&perf, start, stop);
    printf("event timer %f ms\n", perf);


    return 0;
#endif
#if 0
    dec dd1;
    printf("sizeof(dd1) is %lu\n", sizeof(dd1));

    printf("sizeof(Decimal) is %lu\n", sizeof(Decimal));

    /*
    for(int i = 9; i < 31; i++)
        for(int j = 1; j < i-4; j++){
            Decimal testDec(i, j, "1008.1");
            char testRes[256];
            testDec.GetDecimal(testRes);
            printf("test decimal (%d, %d) is %s\n", i, j, testRes);
        }
    */

    const char *op1 = "111111111111111555555555555555555";
    const char *op2 = "333333333333333666666666666666666";
    const char *op = "+";

    printf(">>>>>>>> %s %s %s <<<<<<<<\n", op1, op, op2);

    double f1, f2;
    f1 = atof(op1); f2 = atof(op2);
    double res;

//    struct timeval st, ed;
    double t;

    if(!strcmp("+", op)){
        gettimeofday(&st, 0);
        res = f1 + f2;
    }else if(!strcmp("-", op)){
        gettimeofday(&st, 0);
        res = f1 - f2;
    }else if(!strcmp("*", op)){
        gettimeofday(&st, 0);
        res = f1 * f2;
    }else if(!strcmp("/", op)){
        gettimeofday(&st, 0);
        res = f1 / f2;
    }else{
        printf("unsupported operator: %s\n", op);
        return -1;
    }
    gettimeofday(&ed, 0);
    t = (1000000.0*(ed.tv_sec-st.tv_sec) + ed.tv_usec-st.tv_usec)/1000.0;
    printf("double calculation: %lf, %lf ms\n", res, t);


    char prLine[1024]; size_t pos = 0;
    // int precs[] = {9, 18, 27, 36, 20, 21, 22, 23, 24, 25, 26, 27, 31};
    // int scals[] = {4,  9, 13, 18, 16, 17, 18, 19, 20, 21, 22, 23, 27};
    // int precs[] = {9, 9, 18, 18, 18, 27, 27, 27, 27, 36, 36, 36, 36, 36};
    // int scals[] = {0, 9,  0,  9, 18,  0,  9, 18, 27,  0,  9, 18, 27, 30};
    int precs[] = {36};
    int scals[] = {18};
    assert(sizeof(precs) == sizeof(scals));
    size_t pair_nr = sizeof(precs) / sizeof(int);

//#define CHK_CODE
    printf("\n*** CPU ***\n");
    char result[256];
    for(size_t i = 0; i < pair_nr; i++){
        Decimal d1(precs[i], scals[i], op1);
        Decimal d2(precs[i], scals[i], op2);
        Decimal d3;
        if(!strcmp("+", op)){
            gettimeofday(&st, 0);
            d3 = d1 + d2;
        }else if(!strcmp("-", op)){
            gettimeofday(&st, 0);
            d3 = d1 - d2;
        }else if(!strcmp("*", op)){
            gettimeofday(&st, 0);
            d3 = d1 * d2;
        }else if(!strcmp("/", op)){
            gettimeofday(&st, 0);
            d3 = d1 / d2;
        }else{
            printf("unsupported operator: %s\n", op);
            return -1;
        }
        gettimeofday(&ed, 0);
        t = (1000000.0*(ed.tv_sec-st.tv_sec) + ed.tv_usec-st.tv_usec)/1000.0;
        d3.GetDecimal(result);
        printf("precision: %d, scale: %d, result: %s, %lf ms\n", precs[i], scals[i], result, t);
        if(i < pair_nr - 1)
            pos += sprintf(prLine + pos, "%lf\t", t);
        else
            pos += sprintf(prLine + pos, "%lf", t);
    }
    printf("%s\n", prLine);
    pos = 0;


    printf("\n***GPU***\n");
    char *d1_d, *d2_d;
    gpuErrchk( cudaMalloc((void **)&d1_d, sizeof(Decimal)) );
    gpuErrchk( cudaMalloc((void **)&d2_d, sizeof(Decimal)) );
    for(size_t i = 0; i < pair_nr; i++){

        Decimal d1(precs[i], scals[i], op1);
        Decimal d2(precs[i], scals[i], op2);
        gpuErrchk( cudaMemcpy(d1_d, &d1, sizeof(Decimal), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d2_d, &d2, sizeof(Decimal), cudaMemcpyHostToDevice) );

        Decimal d3;
        if(!strcmp("+", op)){
#ifdef CUDA_EV_TIMER
            cudaEventRecord(start);
#endif
#ifdef CPU_TIMER
            gettimeofday(&st, 0);
#endif
            cuda_add<<<1, 1>>>(d1_d, d2_d);
#ifdef CUDA_EV_TIMER
            cudaEventRecord(stop);
#endif
        }else if(!strcmp("-", op)){
#ifdef CUDA_EV_TIMER
            cudaEventRecord(start);
#endif
#ifdef CPU_TIMER
            gettimeofday(&st, 0);
#endif
            cuda_sub<<<1, 1>>>(d1_d, d2_d);
#ifdef CUDA_EV_TIMER
            cudaEventRecord(stop);
#endif
        }else if(!strcmp("*", op)){
#ifdef CUDA_EV_TIMER
            cudaEventRecord(start);
#endif
#ifdef CPU_TIMER
            gettimeofday(&st, 0);
#endif
            cuda_mul<<<1, 1>>>(d1_d, d2_d);
#ifdef CUDA_EV_TIMER
            cudaEventRecord(stop);
#endif
        }else if(!strcmp("/", op)){
#ifdef CUDA_EV_TIMER
            cudaEventRecord(start);
#endif
#ifdef CPU_TIMER
            gettimeofday(&st, 0);
#endif
            cuda_div<<<1, 1>>>(d1_d, d2_d);
#ifdef CUDA_EV_TIMER
            cudaEventRecord(stop);
#endif
        }else{
            printf("unsupported operator: %s\n", op);
            return -1;
        }
        gpuErrchk( cudaDeviceSynchronize() );
#ifdef CUDA_EV_TIMER
        gpuErrchk( cudaEventSynchronize(stop) );
#endif
#ifdef CPU_TIMER
            gettimeofday(&ed, 0);
#endif

        gpuErrchk( cudaMemcpy(&d3, d1_d, sizeof(Decimal), cudaMemcpyDeviceToHost) );

#ifdef CUDA_EV_TIMER
        perf = 0;
        cudaEventElapsedTime(&perf, start, stop);
#endif
#ifdef CPU_TIMER
        t = (1000000.0*(ed.tv_sec-st.tv_sec) + ed.tv_usec-st.tv_usec)/1000.0;
#endif

        d3.GetDecimal(result);
        printf("precision: %d, scale: %d, result: %s, ", precs[i], scals[i], result);
#ifdef CUDA_EV_TIMER
        printf("event timer %f ms", perf);
        if(i < pair_nr - 1)
            pos += sprintf(prLine + pos, "%f\t", perf);
        else
            pos += sprintf(prLine + pos, "%f", perf);
#endif
#ifdef CPU_TIMER
        printf("CPU timer %f ms", t);
#endif
        printf("\n");

    }
    printf("%s\n", prLine);
    pos = 0;

    gpuErrchk( cudaFree(d1_d) );
    gpuErrchk( cudaFree(d2_d) );
    // Decimal d1(20, 5, "1.111");
    // Decimal d2(20, 5, "3.333");

    // char *d1_d, *d2_d;
    // gpuErrchk( cudaMalloc((void **)&d1_d, sizeof(Decimal)) );
    // gpuErrchk( cudaMalloc((void **)&d2_d, sizeof(Decimal)) );
    // gpuErrchk( cudaMemcpy(d1_d, &d1, sizeof(Decimal), cudaMemcpyHostToDevice) );
    // gpuErrchk( cudaMemcpy(d2_d, &d2, sizeof(Decimal), cudaMemcpyHostToDevice) );

    // cuda_hello<<<1, 1>>>(d1_d, d2_d);
    // gpuErrchk( cudaDeviceSynchronize() );

    // Decimal d3;
    // d3 = d1 / d2;
    // char result[256];
    // d3.GetDecimal(result);
    // printf("CPU: the result is %s\n", result);

    cudaDeviceReset();
    return 0;
#endif
}

/**/



/*
 * decimal.cxx
 *
 *  Created on: 2019年6月26日
 *      Author: david
 */
#include "decimal.h"
#include <cassert>
#include <cmath>

namespace aries_acc{
/*******************add by zmh*********************************************************************************/

#define aries_max(a,b) ( ((a)>(b)) ? (a):(b) )
#define aries_min(a,b) ( ((a)>(b)) ? (b):(a) )
#define aries_abs(a) ( ((a)<(0)) ? (-a):(a) )
#define aries_is_digit(c) ((c) >= '0' && (c) <= '9')


    ARIES_HOST_DEVICE_NO_INLINE int aries_is_space(int ch) {
        return (unsigned long) (ch - 9) < 5u || ' ' == ch;
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_atoi( const char *str, const char *end )
    {
        int sign;
        int n = 0;
        const char *p = str;

        while( p != end && aries_is_space( *p ) )
            p++;
        if( p != end )
        {
            sign = ( '-' == *p ) ? -1 : 1;
            if( '+' == *p || '-' == *p )
                p++;

            for( n = 0; p != end && aries_is_digit( *p ); p++ )
                n = 10 * n + ( *p - '0' );

            if( sign == -1 )
                n = -n;
        }
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_atoi( const char *str )
    {
        int sign;
        int n = 0;
        const char *p = str;

        while( aries_is_space( *p ) )
            p++;

        sign = ( '-' == *p ) ? -1 : 1;
        if( '+' == *p || '-' == *p )
            p++;

        for( n = 0; aries_is_digit( *p ); p++ )
            n = 10 * n + ( *p - '0' );

        if( sign == -1 )
            n = -n;
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t aries_atol( const char *str, const char *end )
    {
        int sign;
        int64_t n = 0;
        const char *p = str;

        while( p != end && aries_is_space( *p ) )
            p++;
        if( p != end )
        {
            sign = ( '-' == *p ) ? -1 : 1;
            if( '+' == *p || '-' == *p )
                p++;

            for( n = 0; p != end && aries_is_digit( *p ); p++ )
                n = 10 * n + ( *p - '0' );

            if( sign == -1 )
                n = -n;
        }
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t aries_atol( const char *str )
    {
        int sign;
        int64_t n = 0;
        const char *p = str;

        while( aries_is_space( *p ) )
            p++;

        sign = ( '-' == *p ) ? -1 : 1;
        if( '+' == *p || '-' == *p )
            p++;

        for( n = 0; aries_is_digit( *p ); p++ )
            n = 10 * n + ( *p - '0' );

        if( sign == -1 )
            n = -n;
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_strlen(const char *str) {
        const char *p = str;
        while (*p++);

        return (int) (p - str - 1);
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strcpy(char *strDest, const char *strSrc) {
        if (strDest == strSrc) {
            return strDest;
        }
        assert((strDest != NULL) && (strSrc != NULL));
        char *address = strDest;
        while ((*strDest++ = *strSrc++));
        return address;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strncpy(char *strDest, const char *strSrc, unsigned int count) {
        if (strDest == strSrc) {
            return strDest;
        }
        assert((strDest != NULL) && (strSrc != NULL));
        char *address = strDest;
        while (count-- && *strSrc)
            *strDest++ = *strSrc++;
        *strDest = 0;
        return address;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strcat(char *strDes, const char *strSrc) {
        assert((strDes != NULL) && (strSrc != NULL));
        char *address = strDes;
        while (*strDes)
            ++strDes;
        while ((*strDes++ = *strSrc++));
        return address;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strncat(char *strDes, const char *strSrc, unsigned int count) {
        assert((strDes != NULL) && (strSrc != NULL));
        char *address = strDes;
        while (*strDes)
            ++strDes;
        while (count-- && *strSrc)
            *strDes++ = *strSrc++;
        *strDes = 0;
        return address;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strchr(const char *str, int ch) {
        while (*str && *str != (char) ch)
            str++;

        if (*str == (char) ch)
            return ((char *) str);

        return 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_sprintf(char *dst, const char *fmt, int v) {
        int startPos = 0;
        int len = aries_strlen(fmt);
        //only support format : %d, %010d
        if (fmt[startPos++] != '%' || fmt[len - 1] != 'd') {
            assert(0);
            return dst;
        }

        int outLen = -1;
        bool fillwithz = false;
        if (fmt[startPos] == '0') {
            fillwithz = true;
            ++startPos;
        }
        char tmp[128];
        if (startPos + 1 < len) {
            aries_strncpy(tmp, fmt + startPos, len - startPos - 1);
            outLen = aries_atoi(tmp);
        }
        //no out
        if (outLen == 0) {
            dst[0] = '0';
            dst[1] = 0;
            return dst;
        }
        int negsign = 0;
        int val = v;
        startPos = 0;
        if (val < 0) {
            negsign = 1;
            val = -val;
        }
        do {
            tmp[startPos++] = char('0' + val % 10);
            val /= 10;
        } while (val > 0);

        len = startPos;
        startPos = 0;
        if (negsign) {
            dst[startPos++] = '-';
        }
        if (outLen == -1) {
            if (len == 0) {
                dst[startPos++] = '0';
            } else {
                for (int i = len - 1; i >= 0; i--) {
                    dst[startPos++] = tmp[i];
                }
            }
            dst[startPos] = 0;
        } else {
            int realLen = len + negsign;
            if (fillwithz) {
                int rep0 = outLen - realLen;
                if (rep0 > 0) {
                    for (int i = 0; i < rep0; i++) {
                        dst[startPos++] = '0';
                    }
                }
            }
            int cpylen = outLen - startPos;
            cpylen = cpylen > len ? len : cpylen;
            for (int i = cpylen - 1; i >= 0; i--) {
                dst[startPos++] = tmp[i];
            }
            dst[startPos] = 0;
        }
        return dst;
    }

    ARIES_HOST_DEVICE_NO_INLINE void *aries_memset(void *dst, int val, unsigned long ulcount) {
        if (!dst)
            return 0;
        char *pchdst = (char *) dst;
        while (ulcount--)
            *pchdst++ = (char) val;

        return dst;
    }

    ARIES_HOST_DEVICE_NO_INLINE void *aries_memcpy(void *dst, const void *src, unsigned long ulcount) {
        if (!(dst && src))
            return 0;
        if (!ulcount)
            return dst;
        char *pchdst = (char *) dst;
        char *pchsrc = (char *) src;
        while (ulcount--)
            *pchdst++ = *pchsrc++;

        return dst;
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_strcmp(const char *source, const char *dest) {
        int ret = 0;
        if (!source || !dest)
            return -2;
        while (!(ret = *(unsigned char *) source - *(unsigned char *) dest) && *dest) {
            source++;
            dest++;
        }

        if (ret < 0)
            ret = -1;
        else if (ret > 0)
            ret = 1;

        return (ret);
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strstr(const char *strSrc, const char *str) {
        assert(strSrc != NULL && str != NULL);
        const char *s = strSrc;
        const char *t = str;
        for (; *strSrc; ++strSrc) {
            for (s = strSrc, t = str; *t && *s == *t; ++s, ++t);
            if (!*t)
                return (char *) strSrc;
        }
        return 0;
    }


    /*************add by zmh end******************************************************************************************/
#define FIX_INTG_FRAC_ERROR(len, intg1, frac1, error)       \
    do                                                      \
    {                                                       \
        if (intg1+frac1 > (len))                            \
        {                                                   \
            if (intg1 > (len))                              \
            {                                               \
                intg1=(len);                                \
                frac1=0;                                    \
                error=ERR_OVER_FLOW;                        \
            }                                               \
            else                                            \
            {                                               \
                frac1=(len)-intg1;                          \
                error=ERR_TRUNCATED;                        \
            }                                               \
        }                                                   \
        else                                                \
        {                                                   \
            error=ERR_OK;                                   \
        }                                                   \
    } while(0)

#define FIX_TAGET_INTG_FRAC_ERROR(len, intg1, frac1, error) \
    do                                                      \
    {                                                       \
        if (intg1+frac1 > (len))                            \
        {                                                   \
            if (frac1 > (len))                              \
            {                                               \
                intg1=(len);                                \
                frac1=0;                                    \
                error=ERR_OVER_FLOW;                        \
            }                                               \
            else                                            \
            {                                               \
                intg1=(len)-frac1;                          \
                error=ERR_TRUNCATED;                        \
            }                                               \
        }                                                   \
        else                                                \
        {                                                   \
            error=ERR_OK;                                   \
        }                                                   \
    } while(0)

#define SET_PREC_SCALE_VALUE(t, d0, d1, d2) (t = (d1 != d2 ? d1 * DIG_PER_INT32 : d0))

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal() : Decimal(DEFAULT_PRECISION, DEFAULT_SCALE) {}

//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal( const Decimal& d )
//    {
//        intg = d.intg;
//        frac = d.frac;
//        mode = d.mode;
//        error = d.error;
//        for( int i = 0; i < NUM_TOTAL_DIG; i++ )
//        {
//            values[i] = d.values[i];
//        }
//    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale) : Decimal(precision, scale, (uint32_t) ARIES_MODE_EMPTY) {
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, uint32_t m) {
        initialize(precision - scale, scale, m);
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, const char s[]) : Decimal( precision, scale, ARIES_MODE_EMPTY, s) {
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, uint32_t m, const char s[] ) {
        initialize(precision - scale, scale, m);
        Decimal d(s);
        cast(d);
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const CompactDecimal *compact, uint32_t precision, uint32_t scale, uint32_t m) {
        initialize(precision - scale, scale, m);
        SignPos signPos;
        int fracBits = GetNeedBits(frac);
        int intgBits = GetNeedBits(intg);
        int realFracBytes = NEEDBYTES(fracBits);
        int realIntgBytes = NEEDBYTES(intgBits);
        if (HAS_FREE_BIT(intgBits)) {
            signPos = INTG_PART;
        } else if (HAS_FREE_BIT(fracBits)) {
            signPos = FRAC_PART;
        } else {
            signPos = ADDITIONAL_PART;
        }
        int sign = 0;
        //handle frag part
        int fracInts = NEEDELEMENTS(frac);
        if (realFracBytes) {
            aries_memcpy((char *)(values + (NUM_TOTAL_DIG - fracInts) ), compact->data + realIntgBytes, realFracBytes);
            if (signPos == FRAC_PART) {
                char *temp = ((char *)(values + INDEX_LAST_DIG));
                if (GET_COMPACT_BYTES(realFracBytes) == realFracBytes) {
                    // <= 3 bytes only
                    temp += GET_COMPACT_BYTES(realFracBytes) - 1;
                } else {
                    // >=4 bytes, have one sort
                    temp += 3;
                }
                sign = GET_SIGN_FROM_BIT(*temp);
                *temp = *temp & 0x7f;
            }
            if (GET_COMPACT_BYTES(realFracBytes)) {
                values[INDEX_LAST_DIG] = values[INDEX_LAST_DIG] * GetPowers10( DIG_PER_INT32 - frac % DIG_PER_INT32);
            }
        }
        //handle intg part
        if (realIntgBytes) {
            int wholeInts = GET_WHOLE_INTS(realIntgBytes);
            int compactPart = GET_COMPACT_BYTES(realIntgBytes);
            int pos = NUM_TOTAL_DIG - (fracInts + NEEDELEMENTS(intg));
            if (compactPart) {
                if (wholeInts) {
                    aries_memcpy((char *)(values + (pos + 1)), compact->data + compactPart, realIntgBytes - compactPart);
                }
                aries_memcpy((char *)(values + pos), compact->data, compactPart);
            } else if (wholeInts) {
                aries_memcpy((char *)(values + pos), compact->data, realIntgBytes);
            }
            if (signPos == INTG_PART) {
                char *temp = ((char *)(values + (INDEX_LAST_DIG - fracInts)));
                if (compactPart == realIntgBytes) {
                    // <= 3 bytes only
                    temp += compactPart - 1;
                } else {
                    // >=4 bytes, have one sort
                    temp += 3;
                }
                sign = GET_SIGN_FROM_BIT(*temp);
                *temp = *temp & 0x7f;
            }
        }
        if (signPos == ADDITIONAL_PART) {
            sign = compact->data[realFracBytes + realIntgBytes];
        }
        if (sign) {
            Negate();
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const char s[]) {
        initialize(0, 0, 0);
        bool success = StringToDecimal((char *) s);
        if (!success) {
            SET_ERR(error, ERR_STR_2_DEC);
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int8_t v) {
        initialize(TINYINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        values[INDEX_LAST_DIG] = v;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int16_t v) {
        initialize(SMALLINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        values[INDEX_LAST_DIG] = v;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int32_t v) {
        initialize(INT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        values[INDEX_LAST_DIG - 1] = v / PER_DEC_MAX_SCALE;
        values[INDEX_LAST_DIG] = v % PER_DEC_MAX_SCALE;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int64_t v) {
        initialize(BIGINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        int64_t t = v / PER_DEC_MAX_SCALE;
        values[INDEX_LAST_DIG - 2] = t / PER_DEC_MAX_SCALE;
        values[INDEX_LAST_DIG - 1] = t % PER_DEC_MAX_SCALE;
        values[INDEX_LAST_DIG] = v % PER_DEC_MAX_SCALE;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint8_t v) {
        initialize(TINYINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        values[INDEX_LAST_DIG] = v;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint16_t v) {
        initialize(SMALLINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        values[INDEX_LAST_DIG] = v;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t v) {
        initialize(INT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        values[INDEX_LAST_DIG - 1] = v / PER_DEC_MAX_SCALE;
        values[INDEX_LAST_DIG] = v % PER_DEC_MAX_SCALE;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint64_t v) {
        initialize(BIGINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
        int64_t t = v / PER_DEC_MAX_SCALE;
        values[INDEX_LAST_DIG - 2] = t / PER_DEC_MAX_SCALE;
        values[INDEX_LAST_DIG - 1] = t % PER_DEC_MAX_SCALE;
        values[INDEX_LAST_DIG] = v % PER_DEC_MAX_SCALE;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::ToCompactDecimal(char * buf, int len) {
        SignPos signPos;
        int fracBits = GetNeedBits(frac);
        int intgBits = GetNeedBits(intg);
        int compactFracBytes = NEEDBYTES(fracBits);
        int compactIntgBytes = NEEDBYTES(intgBits);
        if (HAS_FREE_BIT(intgBits)) {
            signPos = INTG_PART;
        } else if (HAS_FREE_BIT(fracBits)) {
            signPos = FRAC_PART;
        } else {
            signPos = ADDITIONAL_PART;
        }
        if (len != compactFracBytes + compactIntgBytes + (signPos == ADDITIONAL_PART)) {
            return false;
        }
        int sign = 0;
        if (isLessZero()) {
            sign = 1;
            Negate();
        }
        //handle Frac part
        int usedInts = NEEDELEMENTS(frac);
        if (compactFracBytes) {
            int compactPart = GET_COMPACT_BYTES(compactFracBytes);
            if (compactFracBytes != compactPart) {
                aries_memcpy(buf + compactIntgBytes, (char *)(values + (NUM_TOTAL_DIG - usedInts)), compactFracBytes - compactPart);
            }
            if (compactPart) {
                int v = values[INDEX_LAST_DIG] / GetPowers10(DIG_PER_INT32 - frac % DIG_PER_INT32);
                aries_memcpy(buf + (compactIntgBytes + compactFracBytes - compactPart), (char *)&v, compactPart);
            }
            if (signPos == FRAC_PART) {
                int signBytePos = compactIntgBytes + compactFracBytes - 1;
                //has at last one Integer, use last byte of last one Integer
                if (compactFracBytes != compactPart) {
                    signBytePos -= compactPart;
                }
                assert((buf[signBytePos] & 0x80) == 0x0);
                SET_SIGN_BIT(buf[signBytePos], sign);
            }
        }
        //handle Intg part
        if (compactIntgBytes) {
            usedInts += NEEDELEMENTS(intg); //used to indicating total used Ints
            int wholeInts = GET_WHOLE_INTS(compactIntgBytes);
            int compactPart = GET_COMPACT_BYTES(compactIntgBytes);
            if (compactPart) {
                if (wholeInts) {
                    aries_memcpy(buf + compactPart, (char *)(values + (NUM_TOTAL_DIG - usedInts + 1)), compactIntgBytes - compactPart);
                }
                aries_memcpy(buf, (char *)(values + (NUM_TOTAL_DIG - usedInts)), compactPart);
            } else if (wholeInts) {
                aries_memcpy(buf, (char *)(values + (NUM_TOTAL_DIG - usedInts)), compactIntgBytes);
            }
            if (signPos == INTG_PART) {
                //sign bit is in last byte of intg part
                assert((buf[compactIntgBytes - 1] & 0x80) == 0x0);
                SET_SIGN_BIT(buf[compactIntgBytes - 1], sign);
            }
        }
        if (signPos == ADDITIONAL_PART) {
            buf[compactFracBytes + compactIntgBytes] = (char)sign;
        }

        if (sign) {
            Negate();
        }
        return true;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetInnerPrecisionScale(char result[]) {
        char temp[8];
        aries_sprintf(temp, "%d", intg + frac);
        aries_strcpy(result, temp);
        aries_strcat(result, ",");
        aries_sprintf((char *) temp, "%d", frac);
        aries_strcat(result, temp);
        return result;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetTargetPrecisionScale(char result[]) {
        return GetInnerPrecisionScale(result);
    }

    ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetPrecisionScale(char result[]) {
        if (GET_CALC_INTG(mode) + GET_CALC_FRAC(error) == 0) {
            return GetInnerPrecisionScale(result);
        }
        char temp[8];
        aries_sprintf(temp, "%d", GET_CALC_INTG(mode) + GET_CALC_FRAC(error));
        aries_strcpy(result, temp);
        aries_strcat(result, ",");
        aries_sprintf((char *) temp, "%d", GET_CALC_FRAC(error));
        aries_strcat(result, temp);
        return result;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint16_t Decimal::GetSqlMode() {
        return GET_MODE(mode);
    }

    ARIES_HOST_DEVICE_NO_INLINE uint16_t Decimal::GetError() {
        return GET_ERR(error);
    }

    ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetInnerDecimal(char result[]) const {
        char temp[16];
        int frac0 = NEEDELEMENTS(frac);
        //check sign
        bool postive = true;
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            if (values[i] < 0) {
                postive = false;
                break;
            }
        }
        //handle integer part
        int start = -1;
        int end = NUM_TOTAL_DIG - frac0;
        for (int i = 0; i < end; i++) {
            if (values[i] == 0)
                continue;
            start = i;
            break;
        }
        if (start == -1) {
            aries_strcpy(result, postive ? "0" : "-0");
        } else {
            aries_sprintf(result, "%d", values[start++]);
            for (int i = start; i < NUM_TOTAL_DIG - frac0; i++) {
                aries_sprintf(temp, values[i] < 0 ? "%010d" : "%09d", values[i]);
                aries_strcat(result, values[i] < 0 ? temp + 1 : temp);
            }
        }
        //handle frac part
        if (frac0) {
            aries_strcat(result, ".");
            int start = NUM_TOTAL_DIG - frac0;
            for ( int i = start; i < start + frac / DIG_PER_INT32; i++) {
                aries_sprintf(temp, values[i] < 0 ? "%010d" : "%09d", values[i]);
                aries_strcat(result, values[i] < 0 ? temp + 1 : temp);
            }
            //handle last one
            int remainLen = frac % DIG_PER_INT32;
            if (remainLen) {
                aries_sprintf(temp, values[INDEX_LAST_DIG] < 0 ? "%010d" : "%09d", values[INDEX_LAST_DIG]);
                aries_strncat(result, values[INDEX_LAST_DIG] < 0 ? temp + 1 : temp, remainLen);
            }
        }
        return result;
    }

    ARIES_HOST_DEVICE_NO_INLINE char * Decimal::GetDecimal(char result[]) const{
        int frac0 = GET_CALC_FRAC(error), intg0 = GET_CALC_INTG(mode);
        if (frac0 == 0 && intg0 == 0) {
            return GetInnerDecimal(result);
        }
        if (frac0 != frac || intg0 != intg) {
            //need cast
            Decimal tmp(GET_CALC_INTG(mode) + GET_CALC_FRAC(error), GET_CALC_FRAC(error), GET_MODE(mode));
            SET_ERR(tmp.error, GET_ERR(error));
            tmp.cast(*this);
            return tmp.GetInnerDecimal(result);
        }
        return GetInnerDecimal(result);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckOverFlow() {
        int intg0 = intg == 0 ? 0 : NEEDELEMENTS(intg);
        int frac0 = frac == 0 ? 0 : NEEDELEMENTS(frac);
        int hiScale = intg0 * DIG_PER_INT32 - intg;
        bool neg = *this < 0;
        if (neg) {
            Negate();
        }
        //cross over values
        if (hiScale == 0) {
            intg0 += 1;
        } else {
            hiScale = DIG_PER_INT32 - hiScale;
        }
        int32_t hiMax = GetPowers10(hiScale) - 1;
        int st = NUM_TOTAL_DIG - frac0 - intg0;
        //check highest value
        int over = values[st] > hiMax ? 1 : 0;
        if (!over) {
            for (int i = 0; i < st; ++i) {
                if (values[i]) {
                    over = 1;
                    break;
                }
            }
        }
        if (over) {
            if (GET_MODE(mode) == ARIES_MODE_STRICT_ALL_TABLES) {
                SET_ERR(error, ERR_OVER_FLOW);
            }
            GenMaxDecByPrecision();
        }
        if (neg) {
            Negate();
        }
    }

    /*
     * integer/frac part by pos index
     *   0: value of 0 int
     *   1: value of 1 int
     *   2: value of 2 int
     *   3: value of 3 int
     * */
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::setIntPart(int value, int pos) {
        int frac0 = NEEDELEMENTS(frac);
        int set = frac0 + pos;
        if (set < NUM_TOTAL_DIG) {
            values[INDEX_LAST_DIG - set] = value;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::setFracPart(int value, int pos) {
        int frac0 = NEEDELEMENTS(frac);
        if (pos < frac0) {
            values[INDEX_LAST_DIG - pos] = value;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE int Decimal::getIntPart(int pos) const {
        int frac0 = NEEDELEMENTS(frac);
        int get = frac0 + pos;
        if (get >= NUM_TOTAL_DIG) {
            return 0;
        }
        return values[INDEX_LAST_DIG - get];
    }

    ARIES_HOST_DEVICE_NO_INLINE int Decimal::getFracPart(int pos) const {
        int frac0 = NEEDELEMENTS(frac);
        if (pos >= frac0) {
            return 0;
        }
        return values[INDEX_LAST_DIG - pos];
    }

    //global method
    ARIES_HOST_DEVICE_NO_INLINE Decimal abs(Decimal decimal) {
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            if (decimal.values[i] < 0) {
                decimal.values[i] = -decimal.values[i];
            }
        }
        return decimal;
    }

    ARIES_HOST_DEVICE_NO_INLINE int GetRealBytes(uint16_t precision, uint16_t scale) {
        int fracBits = GetNeedBits(scale);
        int intgBits = GetNeedBits(precision - scale);
        if (HAS_FREE_BIT(fracBits) || HAS_FREE_BIT(intgBits)) {
            return NEEDBYTES(fracBits) +  NEEDBYTES(intgBits);
        } else {
            return NEEDBYTES(fracBits) +  NEEDBYTES(intgBits) + 1;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE int GetNeedBits(int base10Precision) {
        int len = base10Precision / DIG_PER_INT32 * 32;
        switch (base10Precision % DIG_PER_INT32) {
            case 0:
                len += 0;
                break;
            case 1:
                len += 4;
                break;
            case 2:
                len += 7;
                break;
            case 3:
                len += 10;
                break;
            case 4:
                len += 14;
                break;
            case 5:
                len += 17;
                break;
            case 6:
                len += 20;
                break;
            case 7:
                len += 24;
                break;
            case 8:
                len += 27;
                break;
        }
        return len;
    }

    ARIES_HOST_DEVICE_NO_INLINE int GetValidElementsCount( uint16_t precision, uint16_t scale )
    {
        return NEEDELEMENTS( precision - scale ) + NEEDELEMENTS( scale );
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::cast(const Decimal &v) {
        if (frac >= v.frac) {
            SET_ERR(error, GET_ERR(v.error));
            int shift = NEEDELEMENTS(frac) - NEEDELEMENTS(v.frac);
            for (int i = 0; i < shift; ++i) {
                values[i] = 0;
            }
            for (int i = shift; i < NUM_TOTAL_DIG; ++i) {
                values[i - shift] = v.values[i];
            }
            if (intg < v.intg) {
                CheckOverFlow();
            }
        } else {
            if (!v.isFracZero()) {
                int shift = NEEDELEMENTS(v.frac) - NEEDELEMENTS(frac);
                for (int i = 0; i < shift; ++i) {
                    values[i] = 0;
                }
                for (int i = shift; i < NUM_TOTAL_DIG; ++i) {
                    values[i] = v.values[i - shift];
                }
                bool neg = *this < 0;
                if (neg) {
                    Negate();
                }
                //scale down
                int scale = frac;
                if ( scale >= DIG_PER_INT32) {
                    scale %= DIG_PER_INT32;
                }
                if (scale) {
                    // scale 5: 123456789 -> 123460000
                    values[INDEX_LAST_DIG] = values[INDEX_LAST_DIG] / GetPowers10( DIG_PER_INT32 - scale) * GetPowers10( DIG_PER_INT32 - scale);
                }

                //check the carry if cast
                //scale 9, check 1 of next value
                if (++scale == 1) {
                    //use shift as index of values later, change check frac value index
                    --shift;
                }
                scale = DIG_PER_INT32 - scale;
                if (aries_abs(v.values[INDEX_LAST_DIG - shift] / GetPowers10(scale)) % 10 >= 5) {
                    int max = GetPowers10( DIG_PER_INT32);
                    int carry = scale + 1 == DIG_PER_INT32 ? 1 : GetPowers10( scale + 1);
                    for (int i = INDEX_LAST_DIG; i >= 0; --i) {
                        values[i] += carry;
                        if (values[i] < max) {
                            carry = 0;
                            break;
                        }
                        carry = 1;
                        values[i] = 0;
                    }
                    // check highest one
                    if (carry == 1) {
                        values[0] = max;
                    }
                }
                if (neg) {
                    Negate();
                }
            }
            CheckOverFlow();
        }
        assert(intg + frac <= SUPPORTED_MAX_PRECISION && frac <= SUPPORTED_MAX_SCALE);
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::truncate( int p ) {
        uint16_t frac0 = frac, intg0 = intg;
        CalcInnerTruncatePrecision(p);
        CalcTruncatePrecision(p);
        if (p > 0) {
            p = frac;
        } else {
            if (-p >= intg0) {
                //result should be zero
                p = -(NEEDELEMENTS(intg0) + NEEDELEMENTS(frac0)) * DIG_PER_INT32;
            }
        }
        int shift = p >= 0 ? NEEDELEMENTS(frac0) - NEEDELEMENTS(p) : NEEDELEMENTS(frac0);
        if (shift > 0) {
            for ( int i = INDEX_LAST_DIG - shift; i >= 0; --i ) {
                values[i + shift] = values[i];
            }
            for ( int i = 0; i < shift; ++i )
            {
                values[i] = 0;
            }
        } else if (shift < 0) {
            for ( int i = -shift; i < NUM_TOTAL_DIG; ++i ) {
                values[i + shift] = values[i];
            }
            for ( int i = NUM_TOTAL_DIG + shift; i < NUM_TOTAL_DIG; ++i )
            {
                values[i] = 0;
            }
        }
        if (frac > p) {
            int cutPowersN = p > 0 ? (DIG_PER_INT32 - p) % DIG_PER_INT32 : -p;
            int cutInt = cutPowersN / DIG_PER_INT32;
            int cutPowers10 = cutPowersN % DIG_PER_INT32;
            if (cutInt) {
                int cutStartIndex = INDEX_LAST_DIG - (cutPowers10 ? 1 : 0);
                for (int i = cutStartIndex; i > cutStartIndex - cutInt; --i) {
                    values[i] = 0;
                }
            }
            if (cutPowers10) {
                values[INDEX_LAST_DIG] -= values[INDEX_LAST_DIG] % GetPowers10(cutPowers10);
            }
        }
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcTruncTargetPrecision( int p ) {
        frac = p >= 0 ? aries_min(p, SUPPORTED_MAX_SCALE) : 0;
        if (-p >= intg) {
            intg = 1;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcTruncatePrecision( int p ) {
        if (GET_CALC_INTG(mode) == 0 && GET_CALC_FRAC(error) == 0) {
            SET_CALC_INTG(mode, intg);
            SET_CALC_FRAC(error, frac);
        }
        uint16_t frac0 = p >= 0 ? aries_min(p, SUPPORTED_MAX_SCALE) : 0;
        uint16_t intg0 = GET_CALC_INTG(mode);
        if (-p >= intg0) {
            intg0 = 1;
        }
        uint8_t e = 0;
        FIX_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
        SET_CALC_INTG(mode, intg0);
        SET_CALC_FRAC(error,frac0);
        SET_ERR(error, e);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerTruncatePrecision( int p ) {
        uint16_t frac0 = p >= 0 ? aries_min(p, SUPPORTED_MAX_SCALE) : 0;
        uint16_t intg0 = intg;
        if (-p >= intg) {
            intg0 = 1;
        }
        uint16_t frac1, frac2;
        frac1 = frac2 = NEEDELEMENTS(frac0);
        uint16_t intg1, intg2;
        intg1 = intg2 = NEEDELEMENTS(intg0);
        uint8_t e = 0;
        FIX_INTG_FRAC_ERROR(INNER_MAX_PRECISION_INT32_NUM, intg1, frac1, e);
        SET_PREC_SCALE_VALUE(frac, frac0, frac1, frac2);
        SET_PREC_SCALE_VALUE(intg, intg0, intg1, intg2);
        SET_ERR(error, e);
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal::operator bool() const {
        return !isZero();
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::operator-() {
        Decimal decimal(*this);
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            decimal.values[i] = -decimal.values[i];
        }
        return decimal;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int8_t v) {
        Decimal tmp(v);
        SET_MODE(tmp.mode, GET_MODE(mode));
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int16_t v) {
        Decimal tmp(v);
        SET_MODE(tmp.mode, GET_MODE(mode));
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int32_t v) {
        Decimal tmp(v);
        SET_MODE(tmp.mode, GET_MODE(mode));
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int64_t v) {
        Decimal tmp(v);
        SET_MODE(tmp.mode, GET_MODE(mode));
        *this = tmp;
        return *this;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint8_t v) {
        Decimal tmp(v);
        SET_MODE(tmp.mode, GET_MODE(mode));
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint16_t v) {
        Decimal tmp(v);
        SET_MODE(tmp.mode, GET_MODE(mode));
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint32_t v) {
        Decimal tmp(v);
        SET_MODE(tmp.mode, GET_MODE(mode));
        *this = tmp;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint64_t v) {
        Decimal tmp(v);
        SET_MODE(tmp.mode, GET_MODE(mode));
        *this = tmp;
        return *this;
    }

    //for decimal
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, const Decimal &right) {
        int temp;
        if (ALIGNED(left.frac, right.frac)) {
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                if ((temp = (left.values[i] - right.values[i]))) {
                    return temp > 0;
                }
            }
        } else {
            Decimal l(left);
            Decimal r(right);
            l.AlignAddSubData(r);
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                if ((temp = (l.values[i] - r.values[i]))) {
                    return temp > 0;
                }
            }
        }
        return false;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, const Decimal &right) {
        return !(left < right);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, const Decimal &right) {
        int temp;
        if (ALIGNED(left.frac, right.frac)) {
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                if ((temp = (left.values[i] - right.values[i]))) {
                    return temp < 0;
                }
            }
        } else {
            Decimal l(left);
            Decimal r(right);
            l.AlignAddSubData(r);
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                if ((temp = (l.values[i] - r.values[i]))) {
                    return temp < 0;
                }
            }
        }
        return false;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, const Decimal &right) {
        return !(left > right);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, const Decimal &right) {
        if (ALIGNED(left.frac, right.frac)) {
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                if (left.values[i] - right.values[i]) {
                    return false;
                }
            }
        } else {
            Decimal l(left);
            Decimal r(right);
            l.AlignAddSubData(r);
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                if (l.values[i] - r.values[i]) {
                    return false;
                }
            }
        }
        return true;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, const Decimal &right) {
        return !(left == right);
    }

    // for int8_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int8_t left, const Decimal &right) {
        return (int32_t) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int8_t left, const Decimal &right) {
        return (int32_t) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int8_t left, const Decimal &right) {
        return (int32_t) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int8_t left, const Decimal &right) {
        return (int32_t) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int8_t left, const Decimal &right) {
        return (int32_t) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int8_t left, const Decimal &right) {
        return !(left == right);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int8_t right) {
        return left > (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int8_t right) {
        return left >= (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int8_t right) {
        return left < (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int8_t right) {
        return left <= (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int8_t right) {
        return left == (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int8_t right) {
        return left != (int32_t) right;
    }

    // for uint8_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint8_t left, const Decimal &right) {
        return (uint32_t) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint8_t left, const Decimal &right) {
        return (uint32_t) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint8_t left, const Decimal &right) {
        return (uint32_t) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint8_t left, const Decimal &right) {
        return (uint32_t) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint8_t left, const Decimal &right) {
        return (uint32_t) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint8_t left, const Decimal &right) {
        return !(left == right);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint8_t right) {
        return left > (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint8_t right) {
        return left >= (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint8_t right) {
        return left < (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint8_t right) {
        return left <= (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint8_t right) {
        return left == (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint8_t right) {
        return left != (uint32_t) right;
    }

    //for int16_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int16_t left, const Decimal &right) {
        return (int32_t) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int16_t left, const Decimal &right) {
        return (int32_t) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int16_t left, const Decimal &right) {
        return (int32_t) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int16_t left, const Decimal &right) {
        return (int32_t) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int16_t left, const Decimal &right) {
        return (int32_t) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int16_t left, const Decimal &right) {
        return (int32_t) left != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int16_t right) {
        return left > (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int16_t right) {
        return left >= (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int16_t right) {
        return left < (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int16_t right) {
        return left <= (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int16_t right) {
        return left == (int32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int16_t right) {
        return left != (int32_t) right;
    }

    //for uint16_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint16_t left, const Decimal &right) {
        return (uint32_t) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint16_t left, const Decimal &right) {
        return (uint32_t) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint16_t left, const Decimal &right) {
        return (uint32_t) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint16_t left, const Decimal &right) {
        return (uint32_t) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint16_t left, const Decimal &right) {
        return (uint32_t) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint16_t left, const Decimal &right) {
        return (uint32_t) left != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint16_t right) {
        return left > (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint16_t right) {
        return left >= (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint16_t right) {
        return left < (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint16_t right) {
        return left <= (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint16_t right) {
        return left == (uint32_t) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint16_t right) {
        return left != (uint32_t) right;
    }

    //for int32_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int32_t left, const Decimal &right) {
        Decimal d(left);
        return d != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left > d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left >= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left < d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left <= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left == d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int32_t right) {
        Decimal d(right);
        return left != d;
    }

    //for uint32_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint32_t left, const Decimal &right) {
        Decimal d(left);
        return d != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left > d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left >= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left < d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left <= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left == d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint32_t right) {
        Decimal d(right);
        return left != d;
    }

    //for int64_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int64_t left, const Decimal &right) {
        Decimal d(left);
        return d != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left > d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left >= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left < d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left <= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left == d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int64_t right) {
        Decimal d(right);
        return left != d;
    }

    //for uint64_t
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint64_t left, const Decimal &right) {
        Decimal d(left);
        return d != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left > d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left >= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left < d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left <= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left == d;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint64_t right) {
        Decimal d(right);
        return left != d;
    }

    //for float
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(float left, const Decimal &right) {
        return (double) left > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(float left, const Decimal &right) {
        return (double) left >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(float left, const Decimal &right) {
        return (double) left < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(float left, const Decimal &right) {
        return (double) left <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(float left, const Decimal &right) {
        return (double) left == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(float left, const Decimal &right) {
        return (double) left != right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, float right) {
        return left > (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, float right) {
        return left >= (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, float right) {
        return left < (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, float right) {
        return left <= (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, float right) {
        return left == (double) right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, float right) {
        return left != (double) right;
    }

    //for double
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(double left, const Decimal &right) {
        return left > right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(double left, const Decimal &right) {
        return left >= right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(double left, const Decimal &right) {
        return left < right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(double left, const Decimal &right) {
        return left <= right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(double left, const Decimal &right) {
        return left == right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(double left, const Decimal &right) {
        return left != right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, double right) {
        return left.GetDouble() > right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, double right) {
        return left.GetDouble() >= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, double right) {
        return left.GetDouble() < right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, double right) {
        return left.GetDouble() <= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, double right) {
        return left.GetDouble() == right;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, double right) {
        return left.GetDouble() != right;
    }

    // for add
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerAddPrecision(const Decimal& d) {
        uint16_t frac0 = aries_min(aries_max(frac, d.frac), SUPPORTED_MAX_SCALE);
        uint16_t intg0 = aries_max(intg, d.intg);
        int highestV1, highestV2, i1 = GetRealIntgSize(highestV1), i2 = d.GetRealIntgSize(highestV2);
        if (aries_max(i1, i2) >= NEEDELEMENTS(intg0)) {
            int value = i1 > i2 ? highestV1 : i1 < i2 ? highestV2 : highestV1 + highestV2;
            int maxIntg = intg0 % DIG_PER_INT32;
            if (maxIntg == 0) {
                maxIntg = DIG_PER_INT32;
            }
            if (value && (aries_abs(value) >= GetPowers10(maxIntg) - 1)) {
                intg0 += 1;
            }
        }
        uint16_t frac1, frac2;
        frac1 = frac2 = NEEDELEMENTS(frac0);
        uint16_t intg1, intg2;
        intg1 = intg2 = NEEDELEMENTS(intg0);
        uint8_t e = 0;
        FIX_INTG_FRAC_ERROR(INNER_MAX_PRECISION_INT32_NUM, intg1, frac1, e);
        SET_PREC_SCALE_VALUE(frac, frac0, frac1, frac2);
        SET_PREC_SCALE_VALUE(intg, intg0, intg1, intg2);
        SET_ERR(error, e);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcAddPrecision(const Decimal &d) {
        uint16_t frac0 = aries_min(aries_max(GET_CALC_FRAC(error), GET_CALC_FRAC(d.error)), SUPPORTED_MAX_SCALE);
        uint16_t intg0 = aries_max(GET_CALC_INTG(mode), GET_CALC_INTG(d.mode));
        int highestV1, highestV2, i1 = GetRealIntgSize(highestV1), i2 = d.GetRealIntgSize(highestV2);
        if (aries_max(i1, i2) >= NEEDELEMENTS(intg0)) {
            int value = i1 > i2 ? highestV1 : i1 < i2 ? highestV2 : highestV1 + highestV2;
            int maxIntg = intg0 % DIG_PER_INT32;
            if (maxIntg == 0) {
                maxIntg = DIG_PER_INT32;
            }
            if (value && (aries_abs(value) >= GetPowers10(maxIntg) - 1)) {
                intg0 += 1;
            }
        }
        uint8_t e = 0;
        FIX_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
        SET_CALC_INTG(mode, intg0);
        SET_CALC_FRAC(error,frac0);
        SET_ERR(error, e);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcAddTargetPrecision( const Decimal& d ) {
        uint16_t frac0 = aries_min(aries_max(frac, d.frac), SUPPORTED_MAX_SCALE);
        uint16_t intg0 = aries_max(intg, d.intg) + 1;
        uint8_t e = 0;
        FIX_TAGET_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
        intg = intg0;
        frac = frac0;
        error = e;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::AddBothPositiveNums(Decimal &d) {
        AlignAddSubData(d);
        //add
        int32_t carry = 0;
        for (int i = INDEX_LAST_DIG; i >= 0; i--) {
            values[i] += d.values[i];
            values[i] += carry;
            if (values[i] >= PER_DEC_MAX_SCALE) {
                carry = 1;
                values[i] -= PER_DEC_MAX_SCALE;
            } else {
                carry = 0;
            }
        }
        //        CheckOverFlow();
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(const Decimal &d) {
#ifdef CHK_CODE
        char res[256];
        d.GetDecimal(res);
        printf("Decimal::operator+=! operand=%s\n", res);
#endif
#ifdef PR_PROP
        clock_t st, ed;
        st = clock();
#endif
        CheckAndSetCalcPrecision();
        Decimal added(d);
        added.CheckAndSetCalcPrecision();
        //calculate precision after plus
        uint8_t intg0, frac0, mode0, error0;
        if (1 > 0) {
            Decimal calcPrecision(*this);
            calcPrecision.CalcAddPrecision(added);
            calcPrecision.CalcInnerAddPrecision(added);
            intg0 = calcPrecision.intg;
            frac0 = calcPrecision.frac;
            mode0 = calcPrecision.mode;
            error0 = calcPrecision.error;
        }
#ifdef PR_PROP
        ed = clock();
        printf("setup precision: %ld\n", ed - st);
        st = clock();
#endif
        bool addedNeg = added.isLessZero();
        if (isLessZero())  //-
        {
            Negate();
            if (addedNeg)  // --
            {
                //-a + -b = - (a + b)
                added.Negate();
                AddBothPositiveNums(added);
            } else //-+
            {
                //-a + b = - (a - b)
                SubBothPositiveNums(added);
            }
            Negate();
        } else {
            if (addedNeg) //+ -
            {
                // a + -b = a - (-b)
                added.Negate();
                SubBothPositiveNums(added);
            } else {
                AddBothPositiveNums(added);
            }
        }
#ifdef PR_PROP
        ed = clock();
        printf("calculation: %ld\n", ed - st);
#endif
        //set precision
        intg = intg0;
        frac = frac0;
        mode = mode0;
        error = error0;
        return *this;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(int8_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(int16_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(int32_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(int64_t i) {
        Decimal d(i);
        return *this += d;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(uint8_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(uint16_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(uint32_t i) {
        Decimal d(i);
        return *this += d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(uint64_t i) {
        Decimal d(i);
        return *this += d;
    }

    //double / float
    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator+=(const float &f) {
        return *this += (double) f;
    }

    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator+=(const double &l) {
        return GetDouble() + l;
    }

    //self operator
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator++() {
        Decimal d((int8_t) 1);
        *this += d;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::operator++(int32_t) {
        Decimal d((int8_t) 1);
        *this += d;
        return *this;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, int8_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, int16_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, int32_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, int64_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, uint8_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, uint16_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, uint32_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, uint64_t right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp += right;
    }

    //double / float
    ARIES_HOST_DEVICE_NO_INLINE double operator+(const Decimal &left, float right) {
        return left.GetDouble() + right;
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator+(const Decimal &left, double right) {
        return left.GetDouble() + right;
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator+(float left, const Decimal &right) {
        return left + right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator+(double left, const Decimal &right) {
        return left + right.GetDouble();
    }

    // for sub
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcSubPrecision(const Decimal &d) {
        CalcAddPrecision(d);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcSubTargetPrecision(const Decimal &d) {
        CalcAddTargetPrecision(d);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerSubPrecision( const Decimal &d ) {
        CalcInnerAddPrecision(d);
    }

    // op1 and op2 are positive
    ARIES_HOST_DEVICE_NO_INLINE int32_t Decimal::CompareInt(int32_t *op1, int32_t *op2) {
        int32_t res = 0;
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG && res == 0; i++) {
            res = op1[i] - op2[i];
        }
        return res;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::SubBothPositiveNums(Decimal &d) {
        int sign = 1;
        int32_t *p1 = (int32_t *) values, *p2 = (int32_t *) d.values;
        AlignAddSubData(d);
        int32_t r = CompareInt(p1, p2);
        if (r == 0) {
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                values[i] = 0;
            }
            return *this;
        } else if (r < 0) {
            int32_t *tmp;
            tmp = p1;
            p1 = p2;
            p2 = tmp;
            sign = -1;
        }
        //sub
        int32_t carry = 0; //借位
        for (int i = INDEX_LAST_DIG; i >= 0; i--) {
            p1[i] -= p2[i];
            p1[i] -= carry;
            if (p1[i] < 0) {
                p1[i] += PER_DEC_MAX_SCALE;
                carry = 1;
            } else {
                carry = 0;
            }
        }
        if (p1 != values) {
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                values[i] = p1[i];
            }
        }
        if (sign == -1) {
            Negate();
        }
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(const Decimal &d) {
#ifdef CHK_CODE
        char res[256];
        d.GetDecimal(res);
        printf("Decimal::operator-=! operand=%s\n", res);
#endif
#ifdef PR_PROP
        clock_t st, ed;
        st = clock();
#endif
        CheckAndSetCalcPrecision();
        Decimal subd(d);
        subd.CheckAndSetCalcPrecision();
        //calculate precision after plus
        uint8_t intg0, frac0, mode0, error0;
        if (1 > 0) {
            Decimal calcPrecision(*this);
            calcPrecision.CalcAddPrecision(subd);
            calcPrecision.CalcInnerAddPrecision(subd);
            intg0 = calcPrecision.intg;
            frac0 = calcPrecision.frac;
            mode0 = calcPrecision.mode;
            error0 = calcPrecision.error;
        }
#ifdef PR_PROP
        ed = clock();
        printf("setup precsion: %ld\n", ed - st);
        st = clock();
#endif
        bool subdNeg = subd.isLessZero();
        //
        if (isLessZero())   //被减数为负数
        {
            Negate();
            if (subdNeg) //减数为负数
            {
                // -a - -b = b - a = - (a - b)
                subd.Negate();
                SubBothPositiveNums(subd);
            } else //减数为正数
            {
                //-a - b = - (a + b)
                AddBothPositiveNums(subd);
            }
            Negate();
        } else   //被减数为正数
        {
            if (subdNeg) //减数为负数
            {
                //a - -b = a + b
                subd.Negate();
                AddBothPositiveNums(subd);
            } else {
                SubBothPositiveNums(subd);
            }
        }
#ifdef PR_PROP
        ed = clock();
        printf("calculation: %ld\n", ed - st);
#endif
        //set precision
        intg = intg0;
        frac = frac0;
        mode = mode0;
        error = error0;
        return *this;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(int8_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(int16_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(int32_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(int64_t i) {
        Decimal d(i);
        return *this -= d;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(uint8_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(uint16_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(uint32_t i) {
        Decimal d(i);
        return *this -= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(uint64_t i) {
        Decimal d(i);
        return *this -= d;
    }

    //double / float
    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator-=(const float &f) {
        return GetDouble() - f;
    }

    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator-=(const double &l) {
        return GetDouble() - l;
    }

    //self operator
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator--() {
        Decimal d((int8_t) 1);
        return *this -= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::operator--(int32_t) {
        Decimal tmp(*this);
        Decimal d((int8_t) 1);
        return tmp -= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, int8_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, int16_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, int32_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, int64_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, uint8_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, uint16_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, uint32_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, uint64_t right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp -= right;
    }

    //double / float
    ARIES_HOST_DEVICE_NO_INLINE double operator-(const Decimal &left, const float right) {
        return left.GetDouble() - right;
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator-(const Decimal &left, const double right) {
        return left.GetDouble() - right;
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator-(const float left, const Decimal &right) {
        return left - right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator-(const double left, const Decimal &right) {
        return left - right.GetDouble();
    }

    // for multiple
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerMulPrecision(const Decimal& d) {
        uint16_t frac0 = aries_min(frac + d.frac, SUPPORTED_MAX_SCALE);
        uint16_t frac1, frac2;
        frac1 = frac2 = NEEDELEMENTS(frac0);
        uint16_t intg0 = intg + d.intg;
        uint16_t intg1, intg2;
        intg1 = intg2 = NEEDELEMENTS(intg0);
        uint8_t e = 0;
        FIX_INTG_FRAC_ERROR(INNER_MAX_PRECISION_INT32_NUM, intg1, frac1, e);
        SET_PREC_SCALE_VALUE(frac, frac0, frac1, frac2);
        SET_PREC_SCALE_VALUE(intg, intg0, intg1, intg2);
        SET_ERR(error, e);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcMulPrecision(const Decimal &d) {
        uint16_t frac0 = aries_min(GET_CALC_FRAC(error) + GET_CALC_FRAC(d.error), SUPPORTED_MAX_SCALE);
        uint16_t intg0 = GET_CALC_INTG(mode) + GET_CALC_INTG(d.mode);
        uint8_t e = 0;
        FIX_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
        SET_CALC_INTG(mode, intg0);
        SET_CALC_FRAC(error,frac0);
        SET_ERR(error, e);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcMulTargetPrecision(const Decimal &d) {
        uint16_t frac0 = aries_min(frac + d.frac, SUPPORTED_MAX_SCALE);
        uint16_t intg0 = intg + intg;
        uint8_t e = 0;
        FIX_TAGET_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
        intg = intg0;
        frac = frac0;
        error = e;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(const Decimal &d) {
#ifdef CHK_CODE
        char mres[256];
        d.GetDecimal(mres);
        printf("Decimal::operator*=! operand=%s\n", mres);
#endif
#ifdef PR_PROP
        clock_t st, ed;
        st = clock();
#endif
        int sign = 1;
        CheckAndSetCalcPrecision();
        Decimal other(d);
        other.CheckAndSetCalcPrecision();
        if (isLessZero()) {
            sign = -sign;
            Negate();
        }
        if (other.isLessZero()) {
            sign = -sign;
            other.Negate();
        }
        int8_t cutFrac = NEEDELEMENTS(frac) + NEEDELEMENTS(d.frac);
        //calculate precision after multiple
        CalcMulPrecision(other);
        CalcInnerMulPrecision(other);
#ifdef PR_PROP
        ed = clock();
        printf("setup precision: %ld\n", ed - st);
        st = clock();
#endif
        cutFrac -= NEEDELEMENTS(frac);
        //swap values
        for ( int k = 0; k <= INDEX_LAST_DIG / 2; ++k ) {
            int32_t v = values[k];
            values[k] = values[INDEX_LAST_DIG - k];
            values[INDEX_LAST_DIG - k] = v;
            v = other.values[k];
            other.values[k] = other.values[INDEX_LAST_DIG - k];
            other.values[INDEX_LAST_DIG - k] = v;
        }
        int32_t res[NUM_TOTAL_DIG * 2] = {0};
        int32_t *op1 = values, *op2 = other.values;
#ifdef PR_PROP
        ed = clock();
        printf("swap values: %ld\n", ed - st);
        st = clock();
#endif
        //multiple
        int32_t carry = 0;
        int64_t temp = 0;
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            if (op2[i] == 0) {
                continue;
            }
            carry = 0;
            int32_t resIndex = 0;
            #pragma unroll
            for (int j = 0; j < NUM_TOTAL_DIG; j++) {
                resIndex = i + j;
                if (op1[j] || carry) {
                    if (op1[j]) {
                        temp = (int64_t) op1[j] * op2[i];
                    }
                    temp += res[resIndex] + carry;
                    if (temp >= PER_DEC_MAX_SCALE) {
                        carry = temp / PER_DEC_MAX_SCALE;
                        res[resIndex] = temp % PER_DEC_MAX_SCALE;
                    } else {
                        res[resIndex] = temp;
                        carry = 0;
                    }
                    temp = 0;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            values[INDEX_LAST_DIG - i] = res[i + cutFrac];
        }
        if (sign == -1) {
            Negate();
        }
#ifdef PR_PROP
        ed = clock();
        printf("calculation: %ld\n", ed - st);
#endif
        return *this;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(int8_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(int16_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(int32_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(int64_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(uint8_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(uint16_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(uint32_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(uint64_t i) {
        Decimal tmp(i);
        return *this *= tmp;
    }

    //double / float
    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator*=(const float &f) {
        return GetDouble() * f;
    }

    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator*=(const double &d) {
        return GetDouble() * d;
    }

    //two operators
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, int8_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, int16_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, int32_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, int64_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, uint8_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, uint16_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, uint32_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, uint64_t right) {
        Decimal tmp(right);
        return tmp *= left;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp *= right;
    }

    //double / float
    ARIES_HOST_DEVICE_NO_INLINE double operator*(const Decimal &left, const float right) {
        return left.GetDouble() * right;
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator*(const Decimal &left, const double right) {
        return left.GetDouble() * right;
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator*(const float left, const Decimal &right) {
        return left * right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator*(const double left, const Decimal &right) {
        return left * right.GetDouble();
    }

    // for division
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerDivPrecision(const Decimal& d) {
        uint16_t frac0 = aries_min(frac + DIV_FIX_INNER_FRAC, SUPPORTED_MAX_SCALE);
        int highestV1, highestV2, prec1 = GetRealPrecision(highestV1), prec2 = d.GetRealPrecision(highestV2);
        int16_t intg0 = prec1 - frac - (prec2 - d.frac) + (highestV1 >= highestV2);
        if (intg0 < 0) {
            intg0 = 0;
        }
        uint16_t frac1, frac2;
        frac1 = frac2 = NEEDELEMENTS(frac0);
        uint16_t intg1, intg2;
        intg1 = intg2 = NEEDELEMENTS(intg0);
        uint8_t e = 0;
        FIX_INTG_FRAC_ERROR(INNER_MAX_PRECISION_INT32_NUM, intg1, frac1, e);
        SET_PREC_SCALE_VALUE(frac, frac0, frac1, frac2);
        SET_PREC_SCALE_VALUE(intg, intg0, intg1, intg2);
        SET_ERR(error, e);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcDivPrecision( const Decimal &d ) {
        uint16_t frac0 = aries_min(GET_CALC_FRAC(error) + DIV_FIX_EX_FRAC, SUPPORTED_MAX_SCALE);
        int highestV1, highestV2, prec1 = GetRealPrecision(highestV1), prec2 = d.GetRealPrecision(highestV2);
        int16_t intg0 = prec1 - GET_CALC_FRAC(error) - (prec2 - GET_CALC_FRAC(d.error)) + (highestV1 >= highestV2);
        if (intg0 < 0) {
            intg0 = 0;
        }
        uint8_t e = 0;
        FIX_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
        SET_CALC_INTG(mode, intg0);
        SET_CALC_FRAC(error,frac0);
        SET_ERR(error, e);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcDivTargetPrecision( const Decimal &d ) {
        uint16_t frac0 = aries_min(frac + DIV_FIX_EX_FRAC, SUPPORTED_MAX_SCALE);
        uint16_t intg0 = aries_min(intg + d.frac, SUPPORTED_MAX_PRECISION);
        uint8_t e = 0;
        FIX_TAGET_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
        intg = intg0;
        frac = frac0;
        error = e;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator>>(int n) {
        int shiftDigits = n % DIG_PER_INT32;
        int shiftInt = n / DIG_PER_INT32;
        if (shiftDigits) {
            int lower = GetPowers10(shiftDigits);
            int higher = GetPowers10( DIG_PER_INT32 - shiftDigits);
            int carry = 0, temp = 0;
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
                if (values[i] != 0) {
                    temp = values[i] % lower;
                    values[i] = values[i] / lower;
                } else {
                    temp = 0;
                }
                if (carry) {
                    values[i] += carry * higher;
                }
                carry = temp;
            }
        }
        if (shiftInt) {
            for (int i = INDEX_LAST_DIG; i >= shiftInt; i--) {
                values[i] = values[i - shiftInt];
            }
            for (int i = 0; i < shiftInt; i++) {
                values[i] = 0;
            }
        }
        //for check
        for (int i = 0; i < shiftInt; i++) {
            assert(values[i] == 0);
        }
        if (shiftDigits) {
            int lower = GetPowers10(shiftDigits);
            assert(values[shiftInt] / lower == 0);
        }
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator<<(int n) {
        int shiftDigits = n % DIG_PER_INT32;
        int shiftInt = n / DIG_PER_INT32;
        int lower = GetPowers10( DIG_PER_INT32 - shiftDigits);
        int higher = GetPowers10(shiftDigits);
        if (shiftDigits) {
            int carry = 0, temp = 0;
            for (int i = INDEX_LAST_DIG; i >= 0; i--) {
                if (values[i] != 0) {
                    temp = values[i] / lower;
                    values[i] = values[i] % lower * higher;
                } else {
                    temp = 0;
                }
                if (carry) {
                    values[i] += carry;
                }
                carry = temp;
            }
        }
        if (shiftInt) {
            for (int i = 0; i < NUM_TOTAL_DIG - shiftInt; i++) {
                values[i] = values[i + shiftInt];
            }
            for (int i = NUM_TOTAL_DIG - shiftInt; i < NUM_TOTAL_DIG; i++) {
                values[i] = 0;
            }
        }
        intg += n;
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::UpdateIntgDigits() {
        int validPos = 0;
        for ( validPos = 0; validPos < NUM_TOTAL_DIG; ++validPos )
        {
            if (values[validPos]) {
                break;
            }
        }
        int intg0 = NUM_TOTAL_DIG - validPos - NEEDELEMENTS(frac);
        if (intg0 <= 0) {
            intg = 0;
        } else {
            int v = aries_abs(values[validPos]);
            int digit = 1;
            while(v >= GetPowers10(digit) && ++digit < DIG_PER_INT32);
            intg = (intg0 - 1) * DIG_PER_INT32 + digit;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE int Decimal::GetRealPrecision(int &highestValue) const {
        int validPos = 0;
        for ( ; validPos < NUM_TOTAL_DIG; ++validPos )
        {
            if (values[validPos]) {
                break;
            }
        }
        int prec0 = NUM_TOTAL_DIG - validPos;
        if (prec0 <= 0) {
            highestValue = 0;
            return 0;
        }
        int v = aries_abs(values[validPos]);
        int digit = 1;
        while(v >= GetPowers10(digit) && ++digit < DIG_PER_INT32);
        highestValue = v / GetPowers10(digit - 1);
        if (frac == 0) {
            return digit + (prec0 - 1) * DIG_PER_INT32;
        } else {
            int lastFrac = frac % DIG_PER_INT32;
            return digit + (prec0 - 2) * DIG_PER_INT32 + (lastFrac == 0 ? DIG_PER_INT32 :  lastFrac);
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckAndSetCalcPrecision() {
        CheckAndSetRealPrecision();
        if (GET_CALC_FRAC(error) == 0 && GET_CALC_INTG(mode) == 0) {
            SET_CALC_FRAC(error, frac);
            SET_CALC_INTG(mode, intg);
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckAndSetRealPrecision() {
        int highest;
        int prec = GetRealPrecision(highest);
        intg = prec - frac;
        if ((intg & 0x80) > 0) {
            intg = 0;
        }
        if (intg == 0 && frac == 0) {
            intg = 1;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE int Decimal::GetRealIntgSize(int &highestValue) const {
        int validPos = 0;
        for ( ; validPos < NUM_TOTAL_DIG; ++validPos )
        {
            if (values[validPos]) {
                break;
            }
        }
        int intg0 = NUM_TOTAL_DIG - validPos - NEEDELEMENTS(frac);
        if (intg0 <= 0) {
            highestValue = 0;
            intg0 = 0;
        } else {
            highestValue = values[validPos];
        }
        return intg0;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::GenIntDecimal(int shift) {
        int n = shift;
        if (frac) {
            n -= DIG_PER_INT32 - frac % DIG_PER_INT32;
        }
        if (n > 0) {
            *this << n;
        } else if (n < 0) {
            *this >> (-n);
        }
        frac = 0;
        UpdateIntgDigits();
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::HalfIntDecimal(const Decimal d1, const Decimal d2) {
        Decimal tmp(d1);
        tmp += d2;
        int32_t rds = 0;
        int64_t t[NUM_TOTAL_DIG];
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            t[i] = tmp.values[i];
        }
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            if (rds) {
                t[i] += rds * PER_DEC_MAX_SCALE;
            }
            if (t[i]) {
                rds = t[i] % 2;
                t[i] /= 2;
            }
        }
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            tmp.values[i] = t[i];
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::DivInt(const Decimal ds, const Decimal dt, Decimal &residuel) {
        if (ds.isZero()) {
            residuel = 0;
            return ds;
        }
        int q = ds.intg - dt.intg;
        Decimal qmax(q + 1, 0), qmin(q, 0), qmid, rsdmax, rsdmin, rsdtmp;
        qmax.GenMaxDecByPrecision();
        qmin.GenMinDecByPrecision();
        Decimal t = qmax * dt;
        rsdmax = ds - t;
        if (rsdmax >= 0) {
            residuel = rsdmax;
            return qmax;
        }
        rsdmin = ds - qmin * dt;
        if (rsdmin == 0) {
            residuel = 0;
            return qmin;
        }
        assert(rsdmin > 0);

        clock_t st, ed, acc = 0;
        int iter = 0;
        st = clock();
        while (qmin < qmax) {
            iter++;
            qmid = HalfIntDecimal(qmax, qmin);
            // st = clock();
            if (qmid == qmin) {
                break;
            }
            // ed = clock();
            // acc += (ed - st);
            rsdtmp = ds - qmid * dt;
            // st = clock();
            if (rsdtmp == 0) {
                rsdmin = 0;
                qmin = qmid;
                break;
            } else if (rsdtmp > 0) {
                rsdmin = rsdtmp;
                qmin = qmid;
            } else {
                rsdmax = rsdtmp;
                qmax = qmid;
            }
            // ed = clock();
            // acc += (ed - st);
        }
        ed = clock();
        printf("    divInt loop %ld, iter %d times\n", ed - st, iter);
        residuel = rsdmin;
        return qmin;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::DivByInt(const Decimal &d, int shift, bool isMod) {
        int dvt = d.values[INDEX_LAST_DIG];
        int remainder = 0;
        *this << shift;
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
            if (values[i] || remainder) {
                int64_t tmp = (int64_t) values[i] + (int64_t) remainder * PER_DEC_MAX_SCALE;
                values[i] = tmp / dvt;
                remainder = tmp % dvt;
            }
        }
        if (isMod) {
            *this = remainder;
        } else if (remainder << 1 > dvt) {
            *this += 1;
        }
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::DivByInt64(const Decimal &divisor, int shift, bool isMod) {
        int64_t dvs = ToInt64();
        while (shift > DIG_PER_INT32) {
            dvs *= GetPowers10(DIG_PER_INT32);
            shift -= DIG_PER_INT32;
        }
        dvs *= GetPowers10(shift);
        int64_t dvt = divisor.ToInt64();
        int64_t res = isMod ? (dvs % dvt) : (dvs / dvt + (((dvs % dvt) << 1) >= dvt ? 1 : 0));
        return *this = res;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::Negate() {
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
            values[i] = -values[i];
        }
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::IntToFrac(int fracDigits) {
        int frac0 = NEEDELEMENTS(fracDigits);

        Decimal intgPart(*this);
        intgPart >> (fracDigits);
        Decimal fracPart(*this);
        fracPart << ( DIG_PER_INT32 * NUM_TOTAL_DIG - fracDigits);
        for (int i = 0; i < NUM_TOTAL_DIG - frac0; i++) {
            values[i] = intgPart.values[i + frac0];
        }
        int fracBase = NUM_TOTAL_DIG - frac0;
        for (int i = fracBase; i < NUM_TOTAL_DIG; i++) {
            values[i] = fracPart.values[i - fracBase];
        }
        frac = fracDigits;
        UpdateIntgDigits();
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CopyValue(Decimal &d) {
        #pragma unroll
        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
            values[i] = d.values[i];
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::DivOrMod( const Decimal &d, bool isMod ) {
#ifdef COMPUTE_BY_STRING
        char divitend[128] =
        {   0};
        char divisor[128] =
        {   0};
        char result[128] =
        {   0};
        GetDivDecimalStr( divitend );
        Decimal tmpDt( d );
        tmpDt.GetDivDecimalStr( divisor );
        //multiple
        int len = aries_strlen( divitend );
        int end = len + d.frac + DIV_FIX_INNER_FRAC;
        for( int i = len; i < end; i++ )
        {
            divitend[i] = '0';
        }
        divitend[end] = 0;
        CalcDivPrecision( d );
        DivInt( divitend, divisor, 1, result );
        len = aries_strlen( result );
        assert( frac + intg >= len );
        if (len < frac)
        {
            for (int i = 0; i < frac -len; i++)
            {
                result[len + i] = '0';
            }
            result[frac] = 0;
            len = aries_strlen(result);
        }
        int p = len;
        InsertCh( result, len - frac, '.' );
        if (result[0] == '-')
        {
            p--;
        }
//        Decimal tmp( intg + frac, frac, result );
        Decimal tmp( p, frac, result );
        int err = error;
        *this = tmp;
        error = err;
#else
        /*
        printf("\n*** DivOrMod ***\n");
        clock_t st, ed;
        st = clock();
        printf("intg:%d, frac:%d; d.intg:%d, d.frac:%d\n", intg, frac, d.intg, d.frac);
        */
#ifdef PR_PROP_DIV
        clock_t st, ed;
        st = clock();
#endif
        CheckAndSetCalcPrecision();
        Decimal divitend(*this);
        Decimal divisor(d);
        divisor.CheckAndSetCalcPrecision();
        /*
        ed = clock();
        printf("CheckAndSetCalcPrecision(): %ld\n", ed - st);
        printf("1. divitend.intg=%d, divitend.frac=%d, divisor.intg=%d, divisor.frac=%d\n", divitend.intg, divitend.frac, divisor.intg, divisor.frac);
        st = clock();
        */
        if (isMod)
        {
            CalcModPrecision(divisor);
            CalcInnerModPrecision(divisor);
        } else {
            CalcDivPrecision(divisor);
            CalcInnerDivPrecision(divisor);
        }
        /*
        ed = clock();
        printf("Calc(Inner)DivPrecision(divisor): %ld\n", ed - st);
        printf("2. divitend.intg=%d, divitend.frac=%d, divisor.intg=%d, divisor.frac=%d\n", divitend.intg, divitend.frac, divisor.intg, divisor.frac);
        printf("   intg:%d, frac:%d; d.intg:%d, d.frac:%d\n", intg, frac, d.intg, d.frac);
        */
        if (isZero()) {
            return *this;
        } else if (d.isZero()) {
            SET_ERR(error, ERR_DIV_BY_ZERO);
            return *this;
        }
#ifdef PR_PROP_DIV
        ed = clock();
        printf("setup precision: %ld\n", ed - st);
        st = clock();
#endif

        //st = clock();
        uint8_t divitendFrac = divitend.frac;
        // printf("isMod=%d\n", isMod);
        // printf("  1. d.frac=%d\n", d.frac);
        divitend.GenIntDecimal(isMod ? (divitendFrac < d.frac ? d.frac - divitendFrac : 0) : 0);
        int sign = 1;
        if (divitend.isLessZero()) {
            divitend.Negate();
            sign = -sign;
        }

        divisor.GenIntDecimal(isMod ? (d.frac < divitendFrac ? divitendFrac - d.frac : 0) : 0);
        if (divisor.isLessZero()) {
            sign = -sign;
            divisor.Negate();
        }
#ifdef PR_PROP_DIV
        ed = clock();
        printf("shift fraction: %ld\n", ed - st);
        st = clock();
#endif
        /*
        ed = clock();
        printf("  2. d.frac=%d\n", d.frac);
        printf("GenIntDecimal for divisor and divitend: %ld\n", ed - st);
        printf("3. divitend.intg=%d, divitend.frac=%d, divisor.intg=%d, divisor.frac=%d\n", divitend.intg, divitend.frac, divisor.intg, divisor.frac);
        printf("   intg:%d, frac:%d; d.intg:%d, d.frac:%d\n", intg, frac, d.intg, d.frac);
        */
        int shift = d.frac + DIV_FIX_INNER_FRAC;
        //printf("  1. shift=%d\n", shift);
        if (!isMod) {
            // result is 0
            if (divitend.intg + shift < divisor.intg) {
                aries_memset(values, 0x00, sizeof(values));
                return *this;
            }
        } else {
            shift = 0;
        }
        //printf("  2. shift=%d\n", d.frac);

        /*
        char testRes[256];
        divitend.GetDecimal(testRes);
        printf("divitend: %s\n", testRes);
        divisor.GetDecimal(testRes);
        printf("divisor: %s\n", testRes);
        */
        Decimal res;
        /*
        clock_t acc = 0;
        int iter = 0;
        */
        //check if use integer div operator directly
        if (divitend.intg + shift <= DIG_PER_INT64 && divisor.intg <= DIG_PER_INT64) { 
            res = divitend.DivByInt64(divisor, shift, isMod);
        } else if (divisor.intg <= DIG_PER_INT32) {
            res = divitend.DivByInt(divisor, shift, isMod);
        } else {
            int tmpEx = shift;
            int nDigits = 0;
            //one step DIG_PER_INT32 digit left
            Decimal tmpRes;
            for (; tmpEx > 0;) {
                //iter++;
                divitend.UpdateIntgDigits();
                nDigits = INNER_MAX_PRECISION - divitend.intg - 1;
                if (nDigits > tmpEx) {
                    nDigits = tmpEx;
                }
                tmpEx -= nDigits;
                divitend << (nDigits);
                //st = clock();
                tmpRes = DivInt(divitend, divisor, divitend);
                //ed = clock();
                //acc += (ed - st);
                if (res != 0) {
                    res *= GetPowers10(nDigits);
                }
                res += tmpRes;
            }
            //check if need round up
            if (isMod) {
                res = divitend;
            } else {
                if (divitend + divitend >= divisor) {
                    res += 1;
                }
            }
        }
        CopyValue(res.IntToFrac(frac));
        if (sign == -1) {
            Negate();
        }
#ifdef PR_PROP_DIV
        ed = clock();
        printf("calculation: %ld\n", ed - st);
#endif
        //printf("the last step: %ld, iter %d times\n", acc, iter);
#endif
        return *this;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(const Decimal &d) {
#ifdef CHK_CODE
        char res[256];
        d.GetDecimal(res);
        printf("Decimal::operator/=! operand=%s\n", res);
#endif
        return DivOrMod(d);
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(int8_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(int16_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(int32_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(int64_t i) {
        Decimal d(i);
        return *this /= d;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(uint8_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(uint16_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(uint32_t i) {
        Decimal d(i);
        return *this /= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(uint64_t i) {
        Decimal d(i);
        return *this /= d;
    }

    //double / float
    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator/=(const float &f) {
        return GetDouble() / f;
    }

    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator/=(const double &d) {
        return GetDouble() / d;
    }

    //two operators
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, int8_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, int16_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, int32_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, int64_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, uint8_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, uint16_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, uint32_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, uint64_t right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp /= right;
    }

    //double / float
    ARIES_HOST_DEVICE_NO_INLINE double operator/(const Decimal &left, const float right) {
        return left.GetDouble() / right;
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator/(const Decimal &left, const double right) {
        return left.GetDouble() / right;
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator/(const float left, const Decimal &right) {
        return left / right.GetDouble();
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator/(const double left, const Decimal &right) {
        return left / right.GetDouble();
    }

    // for mod
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcModPrecision( const Decimal &d ) {
        int i = 0;
        uint8_t frac0 = GET_CALC_FRAC(error), frac1 = GET_CALC_FRAC(d.error), intg0;
        if (frac0 < frac1) {
            frac0 = frac1;
        } else {
            i = frac0 - frac1;
        }
        intg0 = GET_CALC_INTG(d.mode) + i;
        SET_CALC_INTG(mode, intg0);
        SET_CALC_FRAC(error, frac0);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcModTargetPrecision( const Decimal &d ) {
        int i = 0;
        uint8_t frac0 = frac, frac1 = d.frac, intg0;
        if (frac0 < frac1) {
            frac0 = frac1;
        } else {
            i = frac0 - frac1;
        }
        intg0 = d.intg + i;
        uint8_t e;
        FIX_TAGET_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
        intg = intg0;
        frac = frac0;
        error = e;
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerModPrecision( const Decimal &d ) {
        int i = 0;
        if (frac < d.frac) {
            frac = d.frac;
        } else {
            i = frac - d.frac;
        }
        intg = d.intg + i;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(const Decimal& d) {
        return DivOrMod(d, true);
    }
    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(int8_t i) {
        Decimal d(i);
        return *this %= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(int16_t i) {
        Decimal d(i);
        return *this %= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(int32_t i) {
        Decimal d(i);
        return *this %= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(int64_t i) {
        Decimal d(i);
        return *this %= d;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(uint8_t i) {
        Decimal d(i);
        return *this %= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(uint16_t i) {
        Decimal d(i);
        return *this %= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(uint32_t i) {
        Decimal d(i);
        return *this %= d;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(uint64_t i) {
        Decimal d(i);
        return *this %= d;
    }

    //double % float
    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator%=(const float &f) {
        return fmod(GetDouble(), f);
    }

    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator%=(const double &d) {
        return fmod(GetDouble(), d);
    }

    //two operators
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    //signed
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, int8_t right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, int16_t right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, int32_t right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, int64_t right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(int8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(int16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(int32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(int64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    //unsigned
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, uint8_t right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, uint16_t right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, uint32_t right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, uint64_t right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(uint8_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(uint16_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(uint32_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(uint64_t left, const Decimal &right) {
        Decimal tmp(left);
        return tmp %= right;
    }

    //double % float
    ARIES_HOST_DEVICE_NO_INLINE double operator%(const Decimal &left, const float right) {
        return fmod(left.GetDouble(), right);
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator%(const Decimal &left, const double right) {
        return fmod(left.GetDouble(), right);
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator%(const float left, const Decimal &right) {
        return fmod((double)left, right.GetDouble());
    }

    ARIES_HOST_DEVICE_NO_INLINE double operator%(const double left, const Decimal &right) {
        return fmod((double)left, right.GetDouble());
    }

    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isFracZero() const {
        for (int i = INDEX_LAST_DIG - NEEDELEMENTS(frac); i <= INDEX_LAST_DIG; ++i) {
            if (values[i]) {
                return false;
            }
        }
        return true;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isZero() const {
        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
            if (values[i] != 0) {
                return false;
            }
        }
        return true;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isLessZero() const {
        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
            if (values[i] < 0) {
                return true;
            }
        }
        return false;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isLessEqualZero() const {
        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
            if (values[i] > 0) {
                return false;
            }
        }
        return true;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isGreaterZero() const {
        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
            if (values[i] > 0) {
                return true;
            }
        }
        return false;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isGreaterEqualZero() const {
        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
            if (values[i] < 0) {
                return false;
            }
        }
        return true;
    }

    ARIES_HOST_DEVICE_NO_INLINE int32_t Decimal::GetPowers10(int i) const {
        int32_t res = 1;
        switch (i) {
            case 0:
                res = 1;
                break;
            case 1:
                res = 10;
                break;
            case 2:
                res = 100;
                break;
            case 3:
                res = 1000;
                break;
            case 4:
                res = 10000;
                break;
            case 5:
                res = 100000;
                break;
            case 6:
                res = 1000000;
                break;
            case 7:
                res = 10000000;
                break;
            case 8:
                res = 100000000;
                break;
            case 9:
                res = PER_DEC_MAX_SCALE;
                break;
            default:
                break;
        }
        return res;
    }

    ARIES_HOST_DEVICE_NO_INLINE int32_t Decimal::GetFracMaxTable(int i) const {
        int32_t res = 0;
        switch (i) {
            case 0:
                res = 900000000;
                break;
            case 1:
                res = 990000000;
                break;
            case 2:
                res = 999000000;
                break;
            case 3:
                res = 999900000;
                break;
            case 4:
                res = 999990000;
                break;
            case 5:
                res = 999999000;
                break;
            case 6:
                res = 999999900;
                break;
            case 7:
                res = 999999990;
                break;
            case 8:
                res = 999999999;
                break;
            default:
                break;
        }
        return res;
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::GenMaxDecByPrecision() {
        int index = NUM_TOTAL_DIG - NEEDELEMENTS(intg) - NEEDELEMENTS(frac);
        // clear no use values
        for (int i = 0; i < index; i++) {
            values[i] = 0;
        }
        int firstDigits = intg % DIG_PER_INT32;
        if (firstDigits) {
            values[index++] = GetPowers10(firstDigits) - 1;
        }
        int32_t overPerDec = PER_DEC_MAX_SCALE - 1;
        for (int i = index; i < NUM_TOTAL_DIG; i++) {
            values[i] = overPerDec;
        }
        //replace last frac if necessary
        if (frac) {
            int lastDigits = frac % DIG_PER_INT32;
            if (lastDigits) {
                values[INDEX_LAST_DIG] = GetFracMaxTable(lastDigits - 1);
            }
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::GenMinDecByPrecision() {
        int index = NUM_TOTAL_DIG - NEEDELEMENTS(intg) - NEEDELEMENTS(frac);
        // clear no use values
        for (int i = 0; i < index; i++) {
            values[i] = 0;
        }
        if (intg) {
            int firstDigits = intg % DIG_PER_INT32;
            if (firstDigits) {
                values[index++] = GetPowers10(firstDigits - 1);
            } else {
                values[index++] = GetPowers10( DIG_PER_INT32 - 1);
            }
        } else if (frac) {
            values[index++] = GetPowers10( DIG_PER_INT32 - 1);
        }
        for (int i = index; i < NUM_TOTAL_DIG; i++) {
            values[i] = 0;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::TransferData(const Decimal *v) {
        if (intg >= v->intg && frac >= v->frac) {
            SET_MODE(mode, GET_MODE(v->mode));
            SET_ERR(error, GET_ERR(v->error));
            int shift = NEEDELEMENTS(frac) - NEEDELEMENTS(v->frac);
            for (int i = shift; i < NUM_TOTAL_DIG; i++) {
                values[i - shift] = v->values[i];
            }
        } else {
            assert(0);
            SET_MODE(mode, GET_MODE(v->mode));
            SET_ERR(error, ERR_OVER_FLOW);
        }
        assert(intg + frac <= SUPPORTED_MAX_PRECISION && frac <= SUPPORTED_MAX_SCALE);
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::AlignAddSubData(Decimal &d) {
        if (frac == d.frac) {
            //do nothing
            return;
        }
        int fracc = NEEDELEMENTS(frac);
        int fracd = NEEDELEMENTS(d.frac);
        //align integer and frac part
        if (fracc == fracd) {
            //do nothing
            return;
        }
        if (fracc > fracd) {
            //shift forward d only, and discard high number
            int shift = fracc - fracd;
            for (int i = 0; i < NUM_TOTAL_DIG - shift; i++) {
                d.values[i] = d.values[i + shift];
            }
            for (int i = NUM_TOTAL_DIG - shift; i < NUM_TOTAL_DIG; i++) {
                d.values[i] = 0;
            }
        } else {
            //shift forward current only, and discard high number
            int shift = fracd - fracc;
            for (int i = 0; i < NUM_TOTAL_DIG - shift; i++) {
                values[i] = values[i + shift];
            }
            for (int i = NUM_TOTAL_DIG - shift; i < NUM_TOTAL_DIG; i++) {
                values[i] = 0;
            }
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE void Decimal::initialize(uint32_t ig, uint32_t fc, uint32_t m) {
        if (fc > SUPPORTED_MAX_SCALE) {
            fc = SUPPORTED_MAX_SCALE;
        }
        if (ig + fc > SUPPORTED_MAX_PRECISION) {
            ig = SUPPORTED_MAX_PRECISION - fc;
        }
        intg = ig;
        frac = fc;
        mode = m;
        error = ERR_OK;
//        SET_CALC_INTG(mode, intg);
//        SET_CALC_FRAC(error, frac);
        aries_memset(values, 0x00, sizeof(values));
    }

    ARIES_HOST_DEVICE_NO_INLINE double Decimal::GetDouble() const {
        double z = 0;
        int frac0 = NEEDELEMENTS(frac);
        for (int i = 0; i < NUM_TOTAL_DIG - frac0; i++) {
            if (values[i]) {
                z += values[i];
            }
            if (z) {
                z *= PER_DEC_MAX_SCALE;
            }
        }
        //handle scale part
        double s = 0;
        for (int i = NUM_TOTAL_DIG - frac0; i < NUM_TOTAL_DIG; i++) {
            if (values[i]) {
                s += values[i];
            }
            if (s) {
                s *= PER_DEC_MAX_SCALE;
            }
        }
        for (int i = 0; i < frac0; i++) {
            s /= PER_DEC_MAX_SCALE;
        }
        z += s;
        return z / PER_DEC_MAX_SCALE;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t Decimal::ToInt64() const {
        //only 2 digits are valid and no frac part
        int64_t res = values[INDEX_LAST_DIG];
        if (values[INDEX_LAST_DIG - 1]) {
            res += (int64_t) values[INDEX_LAST_DIG - 1] * PER_DEC_MAX_SCALE;
        }
        return res;
    }
    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::CheckIfValidStr2Dec(char * str)
    {
        if (*str == '-') ++str;
        for ( int i = 0; i < aries_strlen(str); ++i )
        {
            if (aries_is_digit(str[i]))
            {
                continue;
            }
            if (str[i] == '.')
            {
                continue;
            }
            return false;
        }
        return true;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::StringToDecimal( char * str )
    {
        if (!CheckIfValidStr2Dec(str))
        {
            return false;
        }
        char sign = 1;
        if (*str == '-') {
            ++str;
            sign = -1;
        }
        char *intgend = aries_strchr(str, '.');
        int strLen = aries_strlen(str);
        int intgLen = intgend ? intgend - str : strLen;
        int fracLen = intgend ? strLen - intgLen - 1 : 0;
        assert(fracLen <= SUPPORTED_MAX_SCALE);
        assert(intgLen + fracLen <= SUPPORTED_MAX_PRECISION);
        intg = intgLen;
        frac = fracLen;
        SET_CALC_INTG(mode, intg);
        SET_CALC_FRAC(error, frac);
        int intg0 = NEEDELEMENTS(intgLen);
        int frac0 = NEEDELEMENTS(fracLen);
        int pos = NUM_TOTAL_DIG - frac0 - intg0;
        char temp[16];
        //handle intg part
        int firstLen = intgLen % DIG_PER_INT32;
        if (firstLen) {
            aries_strncpy(temp, str, firstLen);
            temp[firstLen] = 0;
            values[pos++] = aries_atoi(temp);
            str += firstLen;
        }
        for (int i = pos; i < NUM_TOTAL_DIG - frac0; i++) {
            aries_strncpy( temp, str, DIG_PER_INT32);
            temp[DIG_PER_INT32] = 0;
            values[i] = aries_atoi(temp);
            str += DIG_PER_INT32;
        }
        //handle frac part
        if (intgend) {
            str = intgend + 1;
            for (int i = NUM_TOTAL_DIG - frac0; i < NUM_TOTAL_DIG - 1; i++) {
                aries_strncpy( temp, str, DIG_PER_INT32);
                temp[DIG_PER_INT32] = 0;
                values[i] = aries_atoi(temp);
                str += DIG_PER_INT32;
            }
            //handle last one
            aries_strcpy(temp, str);
            values[INDEX_LAST_DIG] = aries_atoi(temp);
            int frac1 = fracLen % DIG_PER_INT32;
            if (frac1) {
                values[INDEX_LAST_DIG] *= GetPowers10( DIG_PER_INT32 - frac1);
            }
        }
        if (sign == -1) {
            #pragma unroll
            for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
                values[i] = -values[i];
            }
        }
        return true;
    }



} //end namespace aries_acc

/**/