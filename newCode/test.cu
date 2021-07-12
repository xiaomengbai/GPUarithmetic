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

// __global__ void vector_add(int *a, int *b, size_t n, int *out)
// {
    // int offset = blockDim.x * blockIdx.x + threadIdx.x;
    // int stride = gridDim.x * blockDim.x;

    // for(int i = offset; i < n; i += stride){
        // //printf("%d (i = %d): %d, %d\n", offset, i, a[i], b[i]);
        // out[i] = a[i] + b[i];
    // }//

// }

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

// std::string origin(std::string str)
// {
    // return str;
// }

// int mul100Int(std::string str)
// {
    // double res = std::stod(str);
    // return res*100;
// }

// template <typename T>
// void retrieveData(const char *datafile, int n, std::function<T (std::string)> func, std::vector<T> &data)
// {
    // std::ifstream myfile;
    // std::string line;
    // size_t found, last_found;
    // myfile.open(datafile);
    // if(myfile.is_open()){

        // while(std::getline(myfile, line)){
            // found = last_found = 0;
            // for(int i = 0; i < n; i++) {
                // found = line.find('|', last_found);
                // if(found == std::string::npos){
                    // std::cout << "failed to find the " << i << "th '|'" << std::endl;
                    // return;
                // }
                // if(i == n - 1)
                    // data.push_back(func(line.substr(last_found, found - last_found)));
                // last_found = found+1;
            // }
        // }
        // myfile.close();
    // }

// }







// int bytesForDecimal(int p, int d)
// {
    // int n = 0;
    // while(p > 0) {
        // if(p >= 9){
            // p -= 9;
            // n += 4;
        // }else{
            // if(p > 6)
                // n += 4;
            // else if(p > 4)
                // n += 3;
            // else if(p > 2)
                // n += 2;
            // else if(p > 0)
                // n += 1;
            // p = 0;
        // }
    // }
    // return n;
// }

// double timeElapsed(struct timeval &st, struct timeval &ed)
// {
    // long sec = ed.tv_sec - st.tv_sec;
    // long msec = ed.tv_usec - st.tv_usec;
    // double elapsed = sec + msec * 1e-6;
    // return elapsed;
// }

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

// template<typename T>
// class ArithmeticTest {
// public:
    // ArithmeticTest(int32_t n);
    // virtual ~ArithmeticTest();

    // void runAdd();
    // void runSub();
    // void runMul();
    // void runDiv();
// private:
    // T *src1;
    // T *src2;
    // T *dst;

    // int32_t _n;
// };


// template<typename T>
// ArithmeticTest<T>::ArithmeticTest(int32_t n) : _n(n)
// {
    // int res;
    // if(res = posix_memalign((void **)&src1, 8, _n * sizeof(T)))
        // printf("Error in allocating memory for src1: %s\n", strerror(res));
    // if(res = posix_memalign((void **)&src2, 8, _n * sizeof(T)))
        // printf("Error in allocating memory for src2: %s\n", strerror(res));
    // if(res = posix_memalign((void **)&dst, 8, _n * sizeof(T)))
        // printf("Error in allocating memory for dst: %s\n", strerror(res));
// }

// template<typename T>
// ArithmeticTest<T>::~ArithmeticTest()
// {
    // if(src1 != nullptr){
        // free(src1);
        // src1 = nullptr;
    // }
    // if(src2 != nullptr){
        // free(src2);
        // src2 = nullptr;
    // }
    // if(dst != nullptr){
        // free(dst);
        // dst = nullptr;
    // }
// }

// template<typename T>
// void ArithmeticTest<T>::runAdd()
// {
    // for(int32_t i = 0; i < _n; i++)
        // dst[i] = src1[i] + src2[i];
// }

// template<typename T>
// void ArithmeticTest<T>::runSub()
// {
    // for(int32_t i = 0; i < _n; i++)
        // dst[i] = src1[i] - src2[i];
// }

// template<typename T>
// void ArithmeticTest<T>::runMul()
// {
    // for(int32_t i = 0; i < _n; i++)
        // dst[i] = src1[i] * src2[i];
// }

// template<typename T>
// void ArithmeticTest<T>::runDiv()
// {
    // for(int32_t i = 0; i < _n; i++)
        // dst[i] = src1[i] / src2[i];
// }


// class ProfileFunc {
// public:
    // ProfileFunc() {}
    // virtual ~ProfileFunc() { }


    // static double timeElapsed(struct timeval &st, struct timeval &ed) {
            // long sec = ed.tv_sec - st.tv_sec;
            // long msec = ed.tv_usec - st.tv_usec;
            // double elapsed = sec + msec * 1e-6;
            // return elapsed;
    // }

    // template<typename T>
    // static double profile( void (ArithmeticTest<T>::*func)(void), ArithmeticTest<T> &testClass ) {
            // timeval st, ed;
            // gettimeofday(&st, 0);
            // (testClass.*func)();
            // gettimeofday(&ed, 0);
            // return ProfileFunc::timeElapsed(st, ed);
    // }
// };

// int test() {
    // throw 10;
    // return 100;
// }

// struct Intg{
    // int32_t v;

    // friend Intg operator+( const Intg& left, const Intg& right );
    // Intg& operator+=(const Intg& d);
    // Intg& operator=( int32_t i );
// };

// Intg &Intg::operator=( int32_t i ) {
    // v = i;
    // return *this;
// }

// Intg &Intg::operator+=(const Intg &i) {
    // //Intg added(i);
    // v += i.v;
    // return *this;
// }

// Intg operator+( const Intg& left, const Intg& right ){
    // Intg tmp(left);
    // return tmp += right;
// }

// int GetNeedBits(int base10Precision) {
    // int len = base10Precision / DIG_PER_INT32 * 32;
    // switch (base10Precision % DIG_PER_INT32) {
    // case 0:
        // len += 0;
        // break;
    // case 1:
        // len += 4;
        // break;
    // case 2:
        // len += 7;
        // break;
    // case 3:
        // len += 10;
        // break;
    // case 4:
        // len += 14;
        // break;
    // case 5:
        // len += 17;
        // break;
    // case 6:
        // len += 20;
        // break;
    // case 7:
        // len += 24;
        // break;
    // case 8:
        // len += 27;
        // break;
    // }
    // return len;
// }

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

// __host__ __device__ void test_int32_arith()
// {
    // int32_t a = INT32_MIN;
    // printf("int32_t: a is %d, a/-1 = %d, 0-a = %d\n", a, a/-1, 0-a);
// }

// __global__ void test_int32_arith_device()
// {
    // test_int32_arith();
// }

#define TEST_INT32_ARITH_HOST() test_int32_arith()
#define TEST_INT32_ARITH_DEVICE() test_int32_arith_device<<<1, 1>>>()

// __host__ __device__ void test_int64_arith()
// {
    // int64_t a = INT64_MIN;
    // printf("int64_t: a is %ld, a/-1 = %ld, 0-a = %ld\n", a, a/-1, 0-a);
// }

// __global__ void test_int64_arith_device()
// {
    // test_int64_arith();
// }

#define TEST_INT32_ARITH_HOST() test_int32_arith()
#define TEST_INT32_ARITH_DEVICE() test_int32_arith_device<<<1, 1>>>()

#define TEST_INT64_ARITH_HOST() test_int64_arith()
#define TEST_INT64_ARITH_DEVICE() test_int64_arith_device<<<1, 1>>>()

using namespace aries_acc;

#define ONEONE     1.111111111111111
#define TWOTWO     2.222222222222222
#define THREETHREE 3.333333333333333

// struct dec
// {
    // // uint16_t sign:1;
    // // uint16_t prec:8;
    // // uint16_t frac:7;
    // int32_t v[3];
// };



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

__global__ void mul_discount(Decimal *e, Decimal *d, int n,Decimal *one)
{
	extern __shared__ Decimal sdata[];
	
    int tid = threadIdx.x;	//线程的一维下标，此处即为线程在块内的索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;	//获取进程的全局编号,块大小 * 块编号 + 线程在块内的索引
	memset(sdata+tid, 0, sizeof(Decimal));
	if(tid == 0)
        memcpy(&sdata[blockDim.x], one, sizeof(Decimal));
    __syncthreads();

    d[i].sign = 1;
    sdata[tid] = sdata[blockDim.x] + d[i]; 
	sdata[tid] = e[i] * sdata[tid];

    memcpy(e+i, sdata+tid, sizeof(Decimal));
}



#include <string>
#define PR_PROP
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

    const char *datafile = "/data/tpch/data/scale_1/csv/org/lineitem.tbl";

	std::vector<std::string> q_str;
    std::vector<std::string> e_str;
    std::vector<std::string> d_str;
    std::vector<std::string> t_str;
	
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


    // allocate memory on both GPU and CPU for holding decimals transformed from the string arrays
    auto allocate = [](std::vector<std::string> &strs, Decimal **cpu, Decimal **gpu) {
        size_t free, total;
        gpuErrchk( cudaMemGetInfo(&free, &total) );
        //printf("Device Memory: %lu/%lu MB\n", free / (1024 * 1024), total / (1024 * 1024));

        size_t size = sizeof(Decimal) * strs.size();
        printf("    allocate %lf/%lf MB on CPU and GPU...\n", size / (1024 * 1024.0), free / (1024 * 1024.0));
        if(size > free){
            printf("Failed to allocate memory %lu (%lf MB), free: %lu\n", size, size / (1024 * 1024.0), free);
            exit(-1);
        }

        *cpu = (Decimal *)malloc(sizeof(Decimal) * strs.size());
        gpuErrchk( cudaMalloc((void **)gpu, sizeof(Decimal) * strs.size()) );

        for(int i = 0; i < strs.size(); i++){
			(*cpu)[i] = Decimal(&strs[i][0]);
			// printf("数据[%d]:: sign = %d,frac = %d,v = %d %d %d %d %d\n",i,(*cpu)[i].sign,(*cpu)[i].frac,(*cpu)[i].v[4],(*cpu)[i].v[3],(*cpu)[i].v[2],(*cpu)[i].v[1],(*cpu)[i].v[0]);  
		}
            
        gpuErrchk( cudaMemcpy(*gpu, *cpu, sizeof(Decimal) * strs.size(), cudaMemcpyHostToDevice) );
    };
	
	Decimal zero("0");
	Decimal sum_cpu("0");
    size_t tupleNr = q_str.size();

    auto setZeroCpu = [&](Decimal &d) {
        memcpy(&d, &zero, sizeof(Decimal));
    };

    assert(q_str.size() == e_str.size());
    assert(e_str.size() == d_str.size());
    assert(d_str.size() == t_str.size());

    // // thread number in a threadblock
    int threadNr = 256;
    size_t resNr = (tupleNr - 1) / threadNr + 1;

    Decimal *sum_gpu;
    auto setZeroGpu = [&](Decimal *d, size_t n) {
        for(int i = 0; i < n; i++)
            gpuErrchk( cudaMemcpy(d + i, &zero, sizeof(Decimal), cudaMemcpyHostToDevice) );
    };
    Decimal sum_res("0");

	
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
                sum_cpu = q_cpu[i] + sum_cpu;
        });

    printf("  cpu计算的结果::sum_cpu :: sign = %d,frac = %d,v = %d %d %d %d %d\n",sum_cpu.sign,sum_cpu.frac,sum_cpu.v[4],sum_cpu.v[3],sum_cpu.v[2],sum_cpu.v[1],sum_cpu.v[0]);  
	
    printf("  cpu计算的时长:: %lf ms\n", cpuPerf);

    printf("  accumulation in Decimal (GPU):");
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
    cudaDeviceSynchronize();
	 printf("  gpu计算的结果::sum_res :: sign = %d,frac = %d,v = %d %d %d %d %d\n",sum_res.sign,sum_res.frac,sum_res.v[4],sum_res.v[3],sum_res.v[2],sum_res.v[1],sum_res.v[0]);  
	
     printf("  gpu计算的时长:: %f ms\n", gpuPerf);
    gpuPerfTotal += gpuPerf;

    free( q_cpu );
    gpuErrchk( cudaFree(q_gpu) );
	
	
	sum_cpu = Decimal("0");
	//sum(l_extendedprice*(1-l_discount))
    printf("sum(l_extendedprice * (1 - l_discount)) tupleNr=%lu\n", tupleNr);
    cpuPerf = cpuTimer.timing( [&](){
            allocate(e_str, &e_cpu, &e_gpu);
            allocate(d_str, &d_cpu, &d_gpu);
        });
    printf("  Load data complete! %lf ms\n", cpuPerf);

    Decimal one_cpu = Decimal("1.00");
    Decimal tmpRes = Decimal("0");

    printf("  accumulation in decimal (CPU):");
    
    cpuPerf = cpuTimer.timing( [&](){
            for(int i = 0; i < tupleNr; i++) {
                sum_cpu += (e_cpu[i] * (one_cpu - d_cpu[i]));
            }
        });

    printf("  cpu计算的结果::sum_cpu :: sign = %d,frac = %d,v = %d %d %d %d %d\n",sum_cpu.sign,sum_cpu.frac,sum_cpu.v[4],sum_cpu.v[3],sum_cpu.v[2],sum_cpu.v[1],sum_cpu.v[0]);  
	
    printf("  cpu计算的时长:: %lf ms\n", cpuPerf);
    cpuPerfTotal += cpuPerf;


    printf("  accumulation in decimal (GPU):");
	gpuErrchk( cudaMalloc((void **)&sum_gpu, sizeof(Decimal) * resNr) );
    setZeroGpu(sum_gpu, resNr);
	
	Decimal *one_gpu;
	gpuErrchk( cudaMalloc((void **)&one_gpu, sizeof(Decimal)) );
	gpuErrchk( cudaMemcpy(one_gpu, &one_cpu, sizeof(Decimal), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(e_gpu, e_cpu, sizeof(Decimal) * tupleNr, cudaMemcpyHostToDevice) );
	
    gpuPerf = gpuTimer.timing( [&](){
            size_t _tupleNr = tupleNr;
            size_t _resNr = resNr;
            Decimal *_e_gpu = e_gpu;
            Decimal *_d_gpu = d_gpu;
            Decimal *_sum_gpu = sum_gpu;

            mul_discount<<<_resNr, threadNr, sizeof(Decimal)*(threadNr + 1)>>>(_e_gpu, _d_gpu, _tupleNr,one_gpu);


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

    printf("  gpu计算的结果::sum_res :: sign = %d,frac = %d,v = %d %d %d %d %d\n",sum_res.sign,sum_res.frac,sum_res.v[4],sum_res.v[3],sum_res.v[2],sum_res.v[1],sum_res.v[0]);  
	
    printf("  gpu计算的时长:: %f ms\n", gpuPerf);
    gpuPerfTotal += gpuPerf;
    free( e_cpu );
    gpuErrchk( cudaFree(e_gpu) );
    free( d_cpu );
    gpuErrchk( cudaFree(d_gpu) );
    

	// sum_cpu = Decimal("0");

    // printf("  accumulation in decimal (CPU):\n");

	// printf("0.0::被除数::9876987698.7698769876987698769876\n");
	// printf("0.0::  除数::9876.987698769876\n");
	// printf("1000次除法\n");
   
	// Decimal xx("9876987698.7698769876987698769876");
	// Decimal yy("9876.987698769876");
	
	// cpuPerf = cpuTimer.timing( [&](){
            // for(int i = 0; i < 1000; i++)
                // sum_cpu = xx/yy;
        // });


     // printf("cpu计算的结果::sum_cpu :: sign = %d,frac = %d,v = %d %d %d %d %d\n",sum_cpu.sign,sum_cpu.frac,sum_cpu.v[4],sum_cpu.v[3],sum_cpu.v[2],sum_cpu.v[1],sum_cpu.v[0]);  
	
    // printf("cpu计算的时长:: %lf ms\n", cpuPerf);
	
	
	// printf("\n");
	// printf("1.0::被除数::123456123456.123456\n");
	// printf("1.0::  除数::1234.567\n");
   
	// Decimal cc("123456123456.123456");
	// Decimal dd("1234.567");
	
	// cpuPerf = cpuTimer.timing( [&](){
            // for(int i = 0; i < 1000; i++)
                // sum_cpu = cc/dd;
        // });
		
	// printf("cpu计算的结果::sum_cpu :: sign = %d,frac = %d,v = %d %d %d %d %d\n",sum_cpu.sign,sum_cpu.frac,sum_cpu.v[4],sum_cpu.v[3],sum_cpu.v[2],sum_cpu.v[1],sum_cpu.v[0]);  
	
    // printf("cpu计算的时长:: %lf ms\n", cpuPerf);
	
	
	// printf("\n");
	// printf("0.0::被除数::1234123.4\n");
	// printf("0.0::  除数::123.123\n\n");
   	// printf("1000次除法\n");

	// Decimal aa("1234123.4");
	// Decimal bb("123.123");
	
    // cpuPerf = cpuTimer.timing( [&](){
            // for(int i = 0; i < 1000; i++)
                // sum_cpu = aa/bb;
        // });


     // printf("cpu计算的结果::sum_cpu :: sign = %d,frac = %d,v = %d %d %d %d %d\n",sum_cpu.sign,sum_cpu.frac,sum_cpu.v[4],sum_cpu.v[3],sum_cpu.v[2],sum_cpu.v[1],sum_cpu.v[0]);  
	
    // printf("cpu计算的时长:: %lf ms\n", cpuPerf);


    cudaDeviceReset();
    return 0;

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

//lixin 更新
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

        if (*str == (char) ch){
			return ((char *) str);
		}
            

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

	//lixin  增加字符串减去指定位置字符函数
	ARIES_HOST_DEVICE_NO_INLINE char *aries_strerase(char *strSrc ,int n) {
		char *strDes = (char*)malloc((aries_strlen(strSrc)-1)*sizeof(char));
		char *address = strDes;
		for(int i = 0; i < n ; i++){
			*strDes++ = *strSrc++;
        }
		strSrc++;
		while ((*strDes++ = *strSrc++));
		
        return address;
    }
	
	//lixin  增加绝对比较函数，要求比较前已对齐
	ARIES_HOST_DEVICE_NO_INLINE int32_t abs_cmp(int32_t *a,const int32_t *b)
	{
		int32_t res = 0;
		#pragma unroll
		for (int i = NUM_TOTAL_DIG - 1; i >= 0 && res == 0; i--) {
			res = a[i] - b[i];
		}
		return res;
	}
	
	//lixin  增加绝对加法函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_add(int32_t *a,const int32_t *b, int32_t *res){
		
		//进位
		int overflow = 0;
		for(int i = 0; i < NUM_TOTAL_DIG; i++){
			res[i] = a[i] + b[i] + overflow; 
			//进位
			overflow = res[i] / PER_DEC_MAX_SCALE;
			//剩余
			res[i] = res[i] % PER_DEC_MAX_SCALE;
		}
		
	}
	
	//lixin  增加绝对减法函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_sub(int32_t *a,const int32_t *b, int32_t *res){
		//将 const a,b化为可变
		const int32_t *sub1, *sub2;
		int32_t r = abs_cmp(a, b);
		
		//将绝对值大的值 赋值到sub1 小的值赋值到sub2
		if(r >= 0){
			sub1 = a;
			sub2 = b;
		}else{
			sub1 = b;
			sub2 = a;
		}
		
		//借位
		int32_t carry = 0;
		//从高位开始减
		for(int i = 0; i < NUM_TOTAL_DIG; i++){
			res[i] = sub1[i] + PER_DEC_MAX_SCALE - sub2[i] - carry;
			carry = !(res[i] / PER_DEC_MAX_SCALE);
			res[i] = res[i] % PER_DEC_MAX_SCALE;
		}
		
		// for(int i = 0; i < NUM_TOTAL_DIG; i++){
		// 	printf("abs_sub::res::%d\n",res[i]);
		// }
	}
	
	//lixin  增加左移函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_lshift(int32_t *a, int len, int n, int32_t *res){
		
		//printf("************abs_lshift***********start\n");
		int32_t lword = n / DIG_PER_INT32;	//simple n=5	rword=0
		int32_t lbit = n % DIG_PER_INT32;		//simple n=5	rword=5
		int32_t ld = 1;
		int32_t ll = 1;
		
		//printf("abs_lshift::step1::lword =%d , lbit =%d ,ld =%d ,ll =%d\n",lword,lbit,ld,ll);
		
		for(int i = 0; i < lbit; i++) ld *= 10;		//simple n=5	ld=100000	10^5
		
		for(int i = 0; i < DIG_PER_INT32 - lbit; i++) ll *= 10;	//simple n=5	ll=10000	10^4
		
		//printf("abs_lshift::step2::lword =%d , lbit =%d ,ld =%d ,ll =%d\n",lword,lbit,ld,ll);
		
		int i=0;
		for(i=0;i<lword;i++){
			res[i]=0;
			//printf("abs_lshift::res[%d]=%d\n",i,res[i]);
		}
	
		res[i] = a[0] % ll * ld;
		//printf("abs_lshift::res[%d]=%d\n",i,res[i]);
		i++;
		
		
		for( i; i < len - lword; i++){	////simple n=5	5-1=4
			res[i] = a[i - lword] % ll * ld + a[i - lword - 1] / ll;
			//printf("abs_lshift::res[%d]=%d\n",i,res[i]);
		}

		if(i<=len - lword){
			i =  len - lword;
		}

		for(i; i < len; i++){
			res[i] = a[i - lword] % ll * ld + a[i - lword - 1] / ll;
		//	printf("abs_lshift::res[%d]=%d\n",i,res[i]);
		}
	
		//printf("************abs_lshift***********end\n");
	}
	
	//lixin  增加绝对乘法函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_mul(int32_t *a,const int32_t *b, int32_t *res){
		
		//printf("**********************abs_mul start****************\n");
		int64_t temp;
		int32_t carry;
		
		//x位 * y位 需要找一个 x+y位的数 来记录中间结果
		for(int i = 0; i < NUM_TOTAL_DIG * 2; i++)
			res[i] = 0;
		
		//a * b 从a的最低位开始 乘以b的每一位
		for(int i = 0; i < NUM_TOTAL_DIG; i++){
			//进位
			carry = 0;
			for(int j = 0; j < NUM_TOTAL_DIG; j++){
				//i*j 位的结果放在 i+j
				temp = (int64_t)a[i] * b[j] + res[i+j] + carry;
			//	printf("abs_mul::temp = %d * %d + %d + %d = %ld\n",a[i],b[j],res[i+j],carry,temp);
				carry = temp / PER_DEC_MAX_SCALE;
				//判断是否溢出
			//   printf("abs_mul::carry = %ld / %d = %d\n",temp,PER_DEC_MAX_SCALE,carry);
				res[i+j] = temp % PER_DEC_MAX_SCALE;
			//    printf("abs_mul::res[%d] = %ld mod %d = %d\n",i+j,temp,PER_DEC_MAX_SCALE,res[i+j]);
			//    printf("\n");
				
			}
			//a的i位已经 乘以了 b的每一位 ，将最后溢出的结果放在 i+NUM_TOTAL_DIG 即 i+5位上
			res[i+NUM_TOTAL_DIG] = carry;
			//printf("abs_mul::res[%d]  = %d\n",i+NUM_TOTAL_DIG,carry);
		}
	}
		
	//lixin  增加获取数字真实长度
	ARIES_HOST_DEVICE_NO_INLINE int getDecimalLen(const int32_t *a){
		int num=0;
		int i;
		int temp;
		for( i=NUM_TOTAL_DIG-1; i>=0; i--){
			if(a[i]!=0){
				temp = a[i];
				num += i*9;
				break;
			}
		}
		while(temp>0){
			temp /= 10;
			num++;
		}
		return num;
		//printf("**********************abs_mul end****************\n");
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

//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal() : Decimal(DEFAULT_PRECISION, DEFAULT_SCALE) {}
//
////    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal( const Decimal& d )
////    {
////        intg = d.intg;
////        frac = d.frac;
////        mode = d.mode;
////        error = d.error;
////        for( int i = 0; i < NUM_TOTAL_DIG; i++ )
////        {
////            values[i] = d.values[i];
////        }
////    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale) : Decimal(precision, scale, (uint32_t) ARIES_MODE_EMPTY) {
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, uint32_t m) {
//        initialize(precision - scale, scale, m);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, const char s[]) : Decimal( precision, scale, ARIES_MODE_EMPTY, s) {
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale, uint32_t m, const char s[] ) {
//        initialize(precision - scale, scale, m);
//        Decimal d(s);
//        cast(d);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const CompactDecimal *compact, uint32_t precision, uint32_t scale, uint32_t m) {
//        initialize(precision - scale, scale, m);
//        SignPos signPos;
//        int fracBits = GetNeedBits(frac);
//        int intgBits = GetNeedBits(intg);
//        int realFracBytes = NEEDBYTES(fracBits);
//        int realIntgBytes = NEEDBYTES(intgBits);
//        if (HAS_FREE_BIT(intgBits)) {
//            signPos = INTG_PART;
//        } else if (HAS_FREE_BIT(fracBits)) {
//            signPos = FRAC_PART;
//        } else {
//            signPos = ADDITIONAL_PART;
//        }
//        int sign = 0;
//        //handle frag part
//        int fracInts = NEEDELEMENTS(frac);
//        if (realFracBytes) {
//            aries_memcpy((char *)(values + (NUM_TOTAL_DIG - fracInts) ), compact->data + realIntgBytes, realFracBytes);
//            if (signPos == FRAC_PART) {
//                char *temp = ((char *)(values + INDEX_LAST_DIG));
//                if (GET_COMPACT_BYTES(realFracBytes) == realFracBytes) {
//                    // <= 3 bytes only
//                    temp += GET_COMPACT_BYTES(realFracBytes) - 1;
//                } else {
//                    // >=4 bytes, have one sort
//                    temp += 3;
//                }
//                sign = GET_SIGN_FROM_BIT(*temp);
//                *temp = *temp & 0x7f;
//            }
//            if (GET_COMPACT_BYTES(realFracBytes)) {
//                values[INDEX_LAST_DIG] = values[INDEX_LAST_DIG] * GetPowers10( DIG_PER_INT32 - frac % DIG_PER_INT32);
//            }
//        }
//        //handle intg part
//        if (realIntgBytes) {
//            int wholeInts = GET_WHOLE_INTS(realIntgBytes);
//            int compactPart = GET_COMPACT_BYTES(realIntgBytes);
//            int pos = NUM_TOTAL_DIG - (fracInts + NEEDELEMENTS(intg));
//            if (compactPart) {
//                if (wholeInts) {
//                    aries_memcpy((char *)(values + (pos + 1)), compact->data + compactPart, realIntgBytes - compactPart);
//                }
//                aries_memcpy((char *)(values + pos), compact->data, compactPart);
//            } else if (wholeInts) {
//                aries_memcpy((char *)(values + pos), compact->data, realIntgBytes);
//            }
//            if (signPos == INTG_PART) {
//                char *temp = ((char *)(values + (INDEX_LAST_DIG - fracInts)));
//                if (compactPart == realIntgBytes) {
//                    // <= 3 bytes only
//                    temp += compactPart - 1;
//                } else {
//                    // >=4 bytes, have one sort
//                    temp += 3;
//                }
//                sign = GET_SIGN_FROM_BIT(*temp);
//                *temp = *temp & 0x7f;
//            }
//        }
//        if (signPos == ADDITIONAL_PART) {
//            sign = compact->data[realFracBytes + realIntgBytes];
//        }
//        if (sign) {
//            Negate();
//        }
//    }
//
	//lixin  更改构造函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const char s[]) {
        initialize(0, 0, 0);
        bool success = StringToDecimal((char *) s);
    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int8_t v) {
//        initialize(TINYINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
//        values[INDEX_LAST_DIG] = v;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int16_t v) {
//        initialize(SMALLINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
//        values[INDEX_LAST_DIG] = v;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int32_t v) {
//        initialize(INT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
//        values[INDEX_LAST_DIG - 1] = v / PER_DEC_MAX_SCALE;
//        values[INDEX_LAST_DIG] = v % PER_DEC_MAX_SCALE;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int64_t v) {
//        initialize(BIGINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
//        int64_t t = v / PER_DEC_MAX_SCALE;
//        values[INDEX_LAST_DIG - 2] = t / PER_DEC_MAX_SCALE;
//        values[INDEX_LAST_DIG - 1] = t % PER_DEC_MAX_SCALE;
//        values[INDEX_LAST_DIG] = v % PER_DEC_MAX_SCALE;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint8_t v) {
//        initialize(TINYINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
//        values[INDEX_LAST_DIG] = v;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint16_t v) {
//        initialize(SMALLINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
//        values[INDEX_LAST_DIG] = v;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t v) {
//        initialize(INT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
//        values[INDEX_LAST_DIG - 1] = v / PER_DEC_MAX_SCALE;
//        values[INDEX_LAST_DIG] = v % PER_DEC_MAX_SCALE;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint64_t v) {
//        initialize(BIGINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
//        int64_t t = v / PER_DEC_MAX_SCALE;
//        values[INDEX_LAST_DIG - 2] = t / PER_DEC_MAX_SCALE;
//        values[INDEX_LAST_DIG - 1] = t % PER_DEC_MAX_SCALE;
//        values[INDEX_LAST_DIG] = v % PER_DEC_MAX_SCALE;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::ToCompactDecimal(char * buf, int len) {
//        SignPos signPos;
//        int fracBits = GetNeedBits(frac);
//        int intgBits = GetNeedBits(intg);
//        int compactFracBytes = NEEDBYTES(fracBits);
//        int compactIntgBytes = NEEDBYTES(intgBits);
//        if (HAS_FREE_BIT(intgBits)) {
//            signPos = INTG_PART;
//        } else if (HAS_FREE_BIT(fracBits)) {
//            signPos = FRAC_PART;
//        } else {
//            signPos = ADDITIONAL_PART;
//        }
//        if (len != compactFracBytes + compactIntgBytes + (signPos == ADDITIONAL_PART)) {
//            return false;
//        }
//        int sign = 0;
//        if (isLessZero()) {
//            sign = 1;
//            Negate();
//        }
//        //handle Frac part
//        int usedInts = NEEDELEMENTS(frac);
//        if (compactFracBytes) {
//            int compactPart = GET_COMPACT_BYTES(compactFracBytes);
//            if (compactFracBytes != compactPart) {
//                aries_memcpy(buf + compactIntgBytes, (char *)(values + (NUM_TOTAL_DIG - usedInts)), compactFracBytes - compactPart);
//            }
//            if (compactPart) {
//                int v = values[INDEX_LAST_DIG] / GetPowers10(DIG_PER_INT32 - frac % DIG_PER_INT32);
//                aries_memcpy(buf + (compactIntgBytes + compactFracBytes - compactPart), (char *)&v, compactPart);
//            }
//            if (signPos == FRAC_PART) {
//                int signBytePos = compactIntgBytes + compactFracBytes - 1;
//                //has at last one Integer, use last byte of last one Integer
//                if (compactFracBytes != compactPart) {
//                    signBytePos -= compactPart;
//                }
//                assert((buf[signBytePos] & 0x80) == 0x0);
//                SET_SIGN_BIT(buf[signBytePos], sign);
//            }
//        }
//        //handle Intg part
//        if (compactIntgBytes) {
//            usedInts += NEEDELEMENTS(intg); //used to indicating total used Ints
//            int wholeInts = GET_WHOLE_INTS(compactIntgBytes);
//            int compactPart = GET_COMPACT_BYTES(compactIntgBytes);
//            if (compactPart) {
//                if (wholeInts) {
//                    aries_memcpy(buf + compactPart, (char *)(values + (NUM_TOTAL_DIG - usedInts + 1)), compactIntgBytes - compactPart);
//                }
//                aries_memcpy(buf, (char *)(values + (NUM_TOTAL_DIG - usedInts)), compactPart);
//            } else if (wholeInts) {
//                aries_memcpy(buf, (char *)(values + (NUM_TOTAL_DIG - usedInts)), compactIntgBytes);
//            }
//            if (signPos == INTG_PART) {
//                //sign bit is in last byte of intg part
//                assert((buf[compactIntgBytes - 1] & 0x80) == 0x0);
//                SET_SIGN_BIT(buf[compactIntgBytes - 1], sign);
//            }
//        }
//        if (signPos == ADDITIONAL_PART) {
//            buf[compactFracBytes + compactIntgBytes] = (char)sign;
//        }
//
//        if (sign) {
//            Negate();
//        }
//        return true;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetInnerPrecisionScale(char result[]) {
//        char temp[8];
//        aries_sprintf(temp, "%d", intg + frac);
//        aries_strcpy(result, temp);
//        aries_strcat(result, ",");
//        aries_sprintf((char *) temp, "%d", frac);
//        aries_strcat(result, temp);
//        return result;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetTargetPrecisionScale(char result[]) {
//        return GetInnerPrecisionScale(result);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetPrecisionScale(char result[]) {
//        if (GET_CALC_INTG(mode) + GET_CALC_FRAC(error) == 0) {
//            return GetInnerPrecisionScale(result);
//        }
//        char temp[8];
//        aries_sprintf(temp, "%d", GET_CALC_INTG(mode) + GET_CALC_FRAC(error));
//        aries_strcpy(result, temp);
//        aries_strcat(result, ",");
//        aries_sprintf((char *) temp, "%d", GET_CALC_FRAC(error));
//        aries_strcat(result, temp);
//        return result;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE uint16_t Decimal::GetSqlMode() {
//        return GET_MODE(mode);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE uint16_t Decimal::GetError() {
//        return GET_ERR(error);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE char *Decimal::GetInnerDecimal(char result[]) const {
//        char temp[16];
//        int frac0 = NEEDELEMENTS(frac);
//        //check sign
//        bool postive = true;
//        #pragma unroll
//        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//            if (values[i] < 0) {
//                postive = false;
//                break;
//            }
//        }
//        //handle integer part
//        int start = -1;
//        int end = NUM_TOTAL_DIG - frac0;
//        for (int i = 0; i < end; i++) {
//            if (values[i] == 0)
//                continue;
//            start = i;
//            break;
//        }
//        if (start == -1) {
//            aries_strcpy(result, postive ? "0" : "-0");
//        } else {
//            aries_sprintf(result, "%d", values[start++]);
//            for (int i = start; i < NUM_TOTAL_DIG - frac0; i++) {
//                aries_sprintf(temp, values[i] < 0 ? "%010d" : "%09d", values[i]);
//                aries_strcat(result, values[i] < 0 ? temp + 1 : temp);
//            }
//        }
//        //handle frac part
//        if (frac0) {
//            aries_strcat(result, ".");
//            int start = NUM_TOTAL_DIG - frac0;
//            for ( int i = start; i < start + frac / DIG_PER_INT32; i++) {
//                aries_sprintf(temp, values[i] < 0 ? "%010d" : "%09d", values[i]);
//                aries_strcat(result, values[i] < 0 ? temp + 1 : temp);
//            }
//            //handle last one
//            int remainLen = frac % DIG_PER_INT32;
//            if (remainLen) {
//                aries_sprintf(temp, values[INDEX_LAST_DIG] < 0 ? "%010d" : "%09d", values[INDEX_LAST_DIG]);
//                aries_strncat(result, values[INDEX_LAST_DIG] < 0 ? temp + 1 : temp, remainLen);
//            }
//        }
//        return result;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE char * Decimal::GetDecimal(char result[]) const{
//        int frac0 = GET_CALC_FRAC(error), intg0 = GET_CALC_INTG(mode);
//        if (frac0 == 0 && intg0 == 0) {
//            return GetInnerDecimal(result);
//        }
//        if (frac0 != frac || intg0 != intg) {
//            //need cast
//            Decimal tmp(GET_CALC_INTG(mode) + GET_CALC_FRAC(error), GET_CALC_FRAC(error), GET_MODE(mode));
//            SET_ERR(tmp.error, GET_ERR(error));
//            tmp.cast(*this);
//            return tmp.GetInnerDecimal(result);
//        }
//        return GetInnerDecimal(result);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckOverFlow() {
//        int intg0 = intg == 0 ? 0 : NEEDELEMENTS(intg);
//        int frac0 = frac == 0 ? 0 : NEEDELEMENTS(frac);
//        int hiScale = intg0 * DIG_PER_INT32 - intg;
//        bool neg = *this < 0;
//        if (neg) {
//            Negate();
//        }
//        //cross over values
//        if (hiScale == 0) {
//            intg0 += 1;
//        } else {
//            hiScale = DIG_PER_INT32 - hiScale;
//        }
//        int32_t hiMax = GetPowers10(hiScale) - 1;
//        int st = NUM_TOTAL_DIG - frac0 - intg0;
//        //check highest value
//        int over = values[st] > hiMax ? 1 : 0;
//        if (!over) {
//            for (int i = 0; i < st; ++i) {
//                if (values[i]) {
//                    over = 1;
//                    break;
//                }
//            }
//        }
//        if (over) {
//            if (GET_MODE(mode) == ARIES_MODE_STRICT_ALL_TABLES) {
//                SET_ERR(error, ERR_OVER_FLOW);
//            }
//            GenMaxDecByPrecision();
//        }
//        if (neg) {
//            Negate();
//        }
//    }
//
//    /*
//     * integer/frac part by pos index
//     *   0: value of 0 int
//     *   1: value of 1 int
//     *   2: value of 2 int
//     *   3: value of 3 int
//     * */
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::setIntPart(int value, int pos) {
//        int frac0 = NEEDELEMENTS(frac);
//        int set = frac0 + pos;
//        if (set < NUM_TOTAL_DIG) {
//            values[INDEX_LAST_DIG - set] = value;
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::setFracPart(int value, int pos) {
//        int frac0 = NEEDELEMENTS(frac);
//        if (pos < frac0) {
//            values[INDEX_LAST_DIG - pos] = value;
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE int Decimal::getIntPart(int pos) const {
//        int frac0 = NEEDELEMENTS(frac);
//        int get = frac0 + pos;
//        if (get >= NUM_TOTAL_DIG) {
//            return 0;
//        }
//        return values[INDEX_LAST_DIG - get];
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE int Decimal::getFracPart(int pos) const {
//        int frac0 = NEEDELEMENTS(frac);
//        if (pos >= frac0) {
//            return 0;
//        }
//        return values[INDEX_LAST_DIG - pos];
//    }
//
//    //global method
//    ARIES_HOST_DEVICE_NO_INLINE Decimal abs(Decimal decimal) {
//        #pragma unroll
//        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//            if (decimal.values[i] < 0) {
//                decimal.values[i] = -decimal.values[i];
//            }
//        }
//        return decimal;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE int GetRealBytes(uint16_t precision, uint16_t scale) {
//        int fracBits = GetNeedBits(scale);
//        int intgBits = GetNeedBits(precision - scale);
//        if (HAS_FREE_BIT(fracBits) || HAS_FREE_BIT(intgBits)) {
//            return NEEDBYTES(fracBits) +  NEEDBYTES(intgBits);
//        } else {
//            return NEEDBYTES(fracBits) +  NEEDBYTES(intgBits) + 1;
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE int GetNeedBits(int base10Precision) {
//        int len = base10Precision / DIG_PER_INT32 * 32;
//        switch (base10Precision % DIG_PER_INT32) {
//            case 0:
//                len += 0;
//                break;
//            case 1:
//                len += 4;
//                break;
//            case 2:
//                len += 7;
//                break;
//            case 3:
//                len += 10;
//                break;
//            case 4:
//                len += 14;
//                break;
//            case 5:
//                len += 17;
//                break;
//            case 6:
//                len += 20;
//                break;
//            case 7:
//                len += 24;
//                break;
//            case 8:
//                len += 27;
//                break;
//        }
//        return len;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE int GetValidElementsCount( uint16_t precision, uint16_t scale )
//    {
//        return NEEDELEMENTS( precision - scale ) + NEEDELEMENTS( scale );
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::cast(const Decimal &v) {
//        if (frac >= v.frac) {
//            SET_ERR(error, GET_ERR(v.error));
//            int shift = NEEDELEMENTS(frac) - NEEDELEMENTS(v.frac);
//            for (int i = 0; i < shift; ++i) {
//                values[i] = 0;
//            }
//            for (int i = shift; i < NUM_TOTAL_DIG; ++i) {
//                values[i - shift] = v.values[i];
//            }
//            if (intg < v.intg) {
//                CheckOverFlow();
//            }
//        } else {
//            if (!v.isFracZero()) {
//                int shift = NEEDELEMENTS(v.frac) - NEEDELEMENTS(frac);
//                for (int i = 0; i < shift; ++i) {
//                    values[i] = 0;
//                }
//                for (int i = shift; i < NUM_TOTAL_DIG; ++i) {
//                    values[i] = v.values[i - shift];
//                }
//                bool neg = *this < 0;
//                if (neg) {
//                    Negate();
//                }
//                //scale down
//                int scale = frac;
//                if ( scale >= DIG_PER_INT32) {
//                    scale %= DIG_PER_INT32;
//                }
//                if (scale) {
//                    // scale 5: 123456789 -> 123460000
//                    values[INDEX_LAST_DIG] = values[INDEX_LAST_DIG] / GetPowers10( DIG_PER_INT32 - scale) * GetPowers10( DIG_PER_INT32 - scale);
//                }
//
//                //check the carry if cast
//                //scale 9, check 1 of next value
//                if (++scale == 1) {
//                    //use shift as index of values later, change check frac value index
//                    --shift;
//                }
//                scale = DIG_PER_INT32 - scale;
//                if (aries_abs(v.values[INDEX_LAST_DIG - shift] / GetPowers10(scale)) % 10 >= 5) {
//                    int max = GetPowers10( DIG_PER_INT32);
//                    int carry = scale + 1 == DIG_PER_INT32 ? 1 : GetPowers10( scale + 1);
//                    for (int i = INDEX_LAST_DIG; i >= 0; --i) {
//                        values[i] += carry;
//                        if (values[i] < max) {
//                            carry = 0;
//                            break;
//                        }
//                        carry = 1;
//                        values[i] = 0;
//                    }
//                    // check highest one
//                    if (carry == 1) {
//                        values[0] = max;
//                    }
//                }
//                if (neg) {
//                    Negate();
//                }
//            }
//            CheckOverFlow();
//        }
//        assert(intg + frac <= SUPPORTED_MAX_PRECISION && frac <= SUPPORTED_MAX_SCALE);
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::truncate( int p ) {
//        uint16_t frac0 = frac, intg0 = intg;
//        CalcInnerTruncatePrecision(p);
//        CalcTruncatePrecision(p);
//        if (p > 0) {
//            p = frac;
//        } else {
//            if (-p >= intg0) {
//                //result should be zero
//                p = -(NEEDELEMENTS(intg0) + NEEDELEMENTS(frac0)) * DIG_PER_INT32;
//            }
//        }
//        int shift = p >= 0 ? NEEDELEMENTS(frac0) - NEEDELEMENTS(p) : NEEDELEMENTS(frac0);
//        if (shift > 0) {
//            for ( int i = INDEX_LAST_DIG - shift; i >= 0; --i ) {
//                values[i + shift] = values[i];
//            }
//            for ( int i = 0; i < shift; ++i )
//            {
//                values[i] = 0;
//            }
//        } else if (shift < 0) {
//            for ( int i = -shift; i < NUM_TOTAL_DIG; ++i ) {
//                values[i + shift] = values[i];
//            }
//            for ( int i = NUM_TOTAL_DIG + shift; i < NUM_TOTAL_DIG; ++i )
//            {
//                values[i] = 0;
//            }
//        }
//        if (frac > p) {
//            int cutPowersN = p > 0 ? (DIG_PER_INT32 - p) % DIG_PER_INT32 : -p;
//            int cutInt = cutPowersN / DIG_PER_INT32;
//            int cutPowers10 = cutPowersN % DIG_PER_INT32;
//            if (cutInt) {
//                int cutStartIndex = INDEX_LAST_DIG - (cutPowers10 ? 1 : 0);
//                for (int i = cutStartIndex; i > cutStartIndex - cutInt; --i) {
//                    values[i] = 0;
//                }
//            }
//            if (cutPowers10) {
//                values[INDEX_LAST_DIG] -= values[INDEX_LAST_DIG] % GetPowers10(cutPowers10);
//            }
//        }
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcTruncTargetPrecision( int p ) {
//        frac = p >= 0 ? aries_min(p, SUPPORTED_MAX_SCALE) : 0;
//        if (-p >= intg) {
//            intg = 1;
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcTruncatePrecision( int p ) {
//        if (GET_CALC_INTG(mode) == 0 && GET_CALC_FRAC(error) == 0) {
//            SET_CALC_INTG(mode, intg);
//            SET_CALC_FRAC(error, frac);
//        }
//        uint16_t frac0 = p >= 0 ? aries_min(p, SUPPORTED_MAX_SCALE) : 0;
//        uint16_t intg0 = GET_CALC_INTG(mode);
//        if (-p >= intg0) {
//            intg0 = 1;
//        }
//        uint8_t e = 0;
//        FIX_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
//        SET_CALC_INTG(mode, intg0);
//        SET_CALC_FRAC(error,frac0);
//        SET_ERR(error, e);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerTruncatePrecision( int p ) {
//        uint16_t frac0 = p >= 0 ? aries_min(p, SUPPORTED_MAX_SCALE) : 0;
//        uint16_t intg0 = intg;
//        if (-p >= intg) {
//            intg0 = 1;
//        }
//        uint16_t frac1, frac2;
//        frac1 = frac2 = NEEDELEMENTS(frac0);
//        uint16_t intg1, intg2;
//        intg1 = intg2 = NEEDELEMENTS(intg0);
//        uint8_t e = 0;
//        FIX_INTG_FRAC_ERROR(INNER_MAX_PRECISION_INT32_NUM, intg1, frac1, e);
//        SET_PREC_SCALE_VALUE(frac, frac0, frac1, frac2);
//        SET_PREC_SCALE_VALUE(intg, intg0, intg1, intg2);
//        SET_ERR(error, e);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal::operator bool() const {
//        return !isZero();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::operator-() {
//        Decimal decimal(*this);
//        #pragma unroll
//        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//            decimal.values[i] = -decimal.values[i];
//        }
//        return decimal;
//    }
//
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int8_t v) {
//        Decimal tmp(v);
//        SET_MODE(tmp.mode, GET_MODE(mode));
//        *this = tmp;
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int16_t v) {
//        Decimal tmp(v);
//        SET_MODE(tmp.mode, GET_MODE(mode));
//        *this = tmp;
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int32_t v) {
//        Decimal tmp(v);
//        SET_MODE(tmp.mode, GET_MODE(mode));
//        *this = tmp;
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(int64_t v) {
//        Decimal tmp(v);
//        SET_MODE(tmp.mode, GET_MODE(mode));
//        *this = tmp;
//        return *this;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint8_t v) {
//        Decimal tmp(v);
//        SET_MODE(tmp.mode, GET_MODE(mode));
//        *this = tmp;
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint16_t v) {
//        Decimal tmp(v);
//        SET_MODE(tmp.mode, GET_MODE(mode));
//        *this = tmp;
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint32_t v) {
//        Decimal tmp(v);
//        SET_MODE(tmp.mode, GET_MODE(mode));
//        *this = tmp;
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator=(uint64_t v) {
//        Decimal tmp(v);
//        SET_MODE(tmp.mode, GET_MODE(mode));
//        *this = tmp;
//        return *this;
//    }
//
//    //for decimal
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, const Decimal &right) {
//        int temp;
//        if (ALIGNED(left.frac, right.frac)) {
//            #pragma unroll
//            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//                if ((temp = (left.values[i] - right.values[i]))) {
//                    return temp > 0;
//                }
//            }
//        } else {
//            Decimal l(left);
//            Decimal r(right);
//            l.AlignAddSubData(r);
//            #pragma unroll
//            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//                if ((temp = (l.values[i] - r.values[i]))) {
//                    return temp > 0;
//                }
//            }
//        }
//        return false;
//    }
//
   // ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, const Decimal &right) {
       // return !(left < right);
   // }

   // ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, const Decimal &right) {
       
       // return false;
   // }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, const Decimal &right) {
//        return !(left > right);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, const Decimal &right) {
//        if (ALIGNED(left.frac, right.frac)) {
//            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//                if (left.values[i] - right.values[i]) {
//                    return false;
//                }
//            }
//        } else {
//            Decimal l(left);
//            Decimal r(right);
//            l.AlignAddSubData(r);
//            #pragma unroll
//            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//                if (l.values[i] - r.values[i]) {
//                    return false;
//                }
//            }
//        }
//        return true;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, const Decimal &right) {
//        return !(left == right);
//    }
//
//    // for int8_t
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int8_t left, const Decimal &right) {
//        return (int32_t) left > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int8_t left, const Decimal &right) {
//        return (int32_t) left >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int8_t left, const Decimal &right) {
//        return (int32_t) left < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int8_t left, const Decimal &right) {
//        return (int32_t) left <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int8_t left, const Decimal &right) {
//        return (int32_t) left == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int8_t left, const Decimal &right) {
//        return !(left == right);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int8_t right) {
//        return left > (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int8_t right) {
//        return left >= (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int8_t right) {
//        return left < (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int8_t right) {
//        return left <= (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int8_t right) {
//        return left == (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int8_t right) {
//        return left != (int32_t) right;
//    }
//
//    // for uint8_t
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint8_t left, const Decimal &right) {
//        return (uint32_t) left > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint8_t left, const Decimal &right) {
//        return (uint32_t) left >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint8_t left, const Decimal &right) {
//        return (uint32_t) left < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint8_t left, const Decimal &right) {
//        return (uint32_t) left <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint8_t left, const Decimal &right) {
//        return (uint32_t) left == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint8_t left, const Decimal &right) {
//        return !(left == right);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint8_t right) {
//        return left > (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint8_t right) {
//        return left >= (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint8_t right) {
//        return left < (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint8_t right) {
//        return left <= (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint8_t right) {
//        return left == (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint8_t right) {
//        return left != (uint32_t) right;
//    }
//
//    //for int16_t
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int16_t left, const Decimal &right) {
//        return (int32_t) left > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int16_t left, const Decimal &right) {
//        return (int32_t) left >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int16_t left, const Decimal &right) {
//        return (int32_t) left < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int16_t left, const Decimal &right) {
//        return (int32_t) left <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int16_t left, const Decimal &right) {
//        return (int32_t) left == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int16_t left, const Decimal &right) {
//        return (int32_t) left != right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int16_t right) {
//        return left > (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int16_t right) {
//        return left >= (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int16_t right) {
//        return left < (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int16_t right) {
//        return left <= (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int16_t right) {
//        return left == (int32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int16_t right) {
//        return left != (int32_t) right;
//    }
//
//    //for uint16_t
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint16_t left, const Decimal &right) {
//        return (uint32_t) left > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint16_t left, const Decimal &right) {
//        return (uint32_t) left >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint16_t left, const Decimal &right) {
//        return (uint32_t) left < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint16_t left, const Decimal &right) {
//        return (uint32_t) left <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint16_t left, const Decimal &right) {
//        return (uint32_t) left == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint16_t left, const Decimal &right) {
//        return (uint32_t) left != right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint16_t right) {
//        return left > (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint16_t right) {
//        return left >= (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint16_t right) {
//        return left < (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint16_t right) {
//        return left <= (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint16_t right) {
//        return left == (uint32_t) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint16_t right) {
//        return left != (uint32_t) right;
//    }
//
//    //for int32_t
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d != right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int32_t right) {
//        Decimal d(right);
//        return left > d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int32_t right) {
//        Decimal d(right);
//        return left >= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int32_t right) {
//        Decimal d(right);
//        return left < d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int32_t right) {
//        Decimal d(right);
//        return left <= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int32_t right) {
//        Decimal d(right);
//        return left == d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int32_t right) {
//        Decimal d(right);
//        return left != d;
//    }
//
//    //for uint32_t
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint32_t left, const Decimal &right) {
//        Decimal d(left);
//        return d != right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint32_t right) {
//        Decimal d(right);
//        return left > d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint32_t right) {
//        Decimal d(right);
//        return left >= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint32_t right) {
//        Decimal d(right);
//        return left < d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint32_t right) {
//        Decimal d(right);
//        return left <= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint32_t right) {
//        Decimal d(right);
//        return left == d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint32_t right) {
//        Decimal d(right);
//        return left != d;
//    }
//
//    //for int64_t
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(int64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(int64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(int64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d != right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, int64_t right) {
//        Decimal d(right);
//        return left > d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, int64_t right) {
//        Decimal d(right);
//        return left >= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, int64_t right) {
//        Decimal d(right);
//        return left < d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, int64_t right) {
//        Decimal d(right);
//        return left <= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, int64_t right) {
//        Decimal d(right);
//        return left == d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, int64_t right) {
//        Decimal d(right);
//        return left != d;
//    }
//
//    //for uint64_t
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint64_t left, const Decimal &right) {
//        Decimal d(left);
//        return d != right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, uint64_t right) {
//        Decimal d(right);
//        return left > d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, uint64_t right) {
//        Decimal d(right);
//        return left >= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, uint64_t right) {
//        Decimal d(right);
//        return left < d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, uint64_t right) {
//        Decimal d(right);
//        return left <= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, uint64_t right) {
//        Decimal d(right);
//        return left == d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, uint64_t right) {
//        Decimal d(right);
//        return left != d;
//    }
//
//    //for float
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(float left, const Decimal &right) {
//        return (double) left > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(float left, const Decimal &right) {
//        return (double) left >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(float left, const Decimal &right) {
//        return (double) left < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(float left, const Decimal &right) {
//        return (double) left <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(float left, const Decimal &right) {
//        return (double) left == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(float left, const Decimal &right) {
//        return (double) left != right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, float right) {
//        return left > (double) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, float right) {
//        return left >= (double) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, float right) {
//        return left < (double) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, float right) {
//        return left <= (double) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, float right) {
//        return left == (double) right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, float right) {
//        return left != (double) right;
//    }
//
//    //for double
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(double left, const Decimal &right) {
//        return left > right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(double left, const Decimal &right) {
//        return left >= right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(double left, const Decimal &right) {
//        return left < right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(double left, const Decimal &right) {
//        return left <= right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(double left, const Decimal &right) {
//        return left == right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(double left, const Decimal &right) {
//        return left != right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal &left, double right) {
//        return left.GetDouble() > right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal &left, double right) {
//        return left.GetDouble() >= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal &left, double right) {
//        return left.GetDouble() < right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal &left, double right) {
//        return left.GetDouble() <= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal &left, double right) {
//        return left.GetDouble() == right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal &left, double right) {
//        return left.GetDouble() != right;
//    }
//
//    // for add
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerAddPrecision(const Decimal& d) {
//        uint16_t frac0 = aries_min(aries_max(frac, d.frac), SUPPORTED_MAX_SCALE);
//        uint16_t intg0 = aries_max(intg, d.intg);
//        int highestV1, highestV2, i1 = GetRealIntgSize(highestV1), i2 = d.GetRealIntgSize(highestV2);
//        if (aries_max(i1, i2) >= NEEDELEMENTS(intg0)) {
//            int value = i1 > i2 ? highestV1 : i1 < i2 ? highestV2 : highestV1 + highestV2;
//            int maxIntg = intg0 % DIG_PER_INT32;
//            if (maxIntg == 0) {
//                maxIntg = DIG_PER_INT32;
//            }
//            if (value && (aries_abs(value) >= GetPowers10(maxIntg) - 1)) {
//                intg0 += 1;
//            }
//        }
//        uint16_t frac1, frac2;
//        frac1 = frac2 = NEEDELEMENTS(frac0);
//        uint16_t intg1, intg2;
//        intg1 = intg2 = NEEDELEMENTS(intg0);
//        uint8_t e = 0;
//        FIX_INTG_FRAC_ERROR(INNER_MAX_PRECISION_INT32_NUM, intg1, frac1, e);
//        SET_PREC_SCALE_VALUE(frac, frac0, frac1, frac2);
//        SET_PREC_SCALE_VALUE(intg, intg0, intg1, intg2);
//        SET_ERR(error, e);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcAddPrecision(const Decimal &d) {
//        uint16_t frac0 = aries_min(aries_max(GET_CALC_FRAC(error), GET_CALC_FRAC(d.error)), SUPPORTED_MAX_SCALE);
//        uint16_t intg0 = aries_max(GET_CALC_INTG(mode), GET_CALC_INTG(d.mode));
//        int highestV1, highestV2, i1 = GetRealIntgSize(highestV1), i2 = d.GetRealIntgSize(highestV2);
//        if (aries_max(i1, i2) >= NEEDELEMENTS(intg0)) {
//            int value = i1 > i2 ? highestV1 : i1 < i2 ? highestV2 : highestV1 + highestV2;
//            int maxIntg = intg0 % DIG_PER_INT32;
//            if (maxIntg == 0) {
//                maxIntg = DIG_PER_INT32;
//            }
//            if (value && (aries_abs(value) >= GetPowers10(maxIntg) - 1)) {
//                intg0 += 1;
//            }
//        }
//        uint8_t e = 0;
//        FIX_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
//        SET_CALC_INTG(mode, intg0);
//        SET_CALC_FRAC(error,frac0);
//        SET_ERR(error, e);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcAddTargetPrecision( const Decimal& d ) {
//        uint16_t frac0 = aries_min(aries_max(frac, d.frac), SUPPORTED_MAX_SCALE);
//        uint16_t intg0 = aries_max(intg, d.intg) + 1;
//        uint8_t e = 0;
//        FIX_TAGET_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
//        intg = intg0;
//        frac = frac0;
//        error = e;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::AddBothPositiveNums(Decimal &d) {
//        AlignAddSubData(d);
//        //add
//        int32_t carry = 0;
//        for (int i = INDEX_LAST_DIG; i >= 0; i--) {
//            values[i] += d.values[i];
//            values[i] += carry;
//            if (values[i] >= PER_DEC_MAX_SCALE) {
//                carry = 1;
//                values[i] -= PER_DEC_MAX_SCALE;
//            } else {
//                carry = 0;
//            }
//        }
//        //        CheckOverFlow();
//        return *this;
//    }
//
	//lixin  更改+=函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(const Decimal &d) {
		// printf("**************operator+=*************start\n"); 
		// printf("beforeAlign\n");
		// printf("operator+=::x :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",sign,prec,frac,v[4],v[3],v[2],v[1],v[0]);  
		// printf("operator+=::d :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",d.sign,d.prec,d.frac,d.v[4],d.v[3],d.v[2],d.v[1],d.v[0]);  
		
		//将const d 拷贝到 added 中
		Decimal added(d);
		//进行对齐操作
		added.AlignAddSubData(*this);
		
		// printf("afterAlign\n");
		// printf("operator+=::x :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",sign,prec,frac,v[4],v[3],v[2],v[1],v[0]);  
		// printf("operator+=::added :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",added.sign,added.prec,added.frac,added.v[4],added.v[3],added.v[2],added.v[1],added.v[0]);  
		
		int32_t temp[NUM_TOTAL_DIG];
		
		for(int i=0;i<5;i++){
			temp[i] = v[i];
		}
		
		//通过两个加数的符号计算
		if(sign ^ added.sign == 0){
		//	printf("var_add::两个加数符号相同\n"); 
			//符号相同，绝对值相加，sign不变
			abs_add(temp, added.v, v);
		}else{
		//	printf("var_add::两个加数符号不同\n");  
			//符号不相，绝对值相减
			abs_sub(temp, added.v, v);
			//sign的符号 与 绝对值较大的符号相同
			sign = (abs_cmp(temp, added.v) > 0 && !added.sign) || (abs_cmp(temp, added.v) < 0 && added.sign);
		}
		// printf("**************operator+=*************end\n"); 
        return *this;
    }
//
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(int8_t i) {
//        Decimal d(i);
//        return *this += d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(int16_t i) {
//        Decimal d(i);
//        return *this += d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(int32_t i) {
//        Decimal d(i);
//        return *this += d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(int64_t i) {
//        Decimal d(i);
//        return *this += d;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(uint8_t i) {
//        Decimal d(i);
//        return *this += d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(uint16_t i) {
//        Decimal d(i);
//        return *this += d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(uint32_t i) {
//        Decimal d(i);
//        return *this += d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator+=(uint64_t i) {
//        Decimal d(i);
//        return *this += d;
//    }
//
//    //double / float
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator+=(const float &f) {
//        return *this += (double) f;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator+=(const double &l) {
//        return GetDouble() + l;
//    }
//
//    //self operator
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator++() {
//        Decimal d((int8_t) 1);
//        *this += d;
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::operator++(int32_t) {
//        Decimal d((int8_t) 1);
//        *this += d;
//        return *this;
//    }
//
//    //signed
	//lixin  更改加法函数
    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, const Decimal &right) {
        //将 const left 赋值到temp进行操作
		Decimal tmp(left);
		//printf("left :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",left.sign,left.prec,left.frac,left.v[4],left.v[3],left.v[2],left.v[1],left.v[0]);  
		//printf("right :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",right.sign,right.prec,right.frac,right.v[4],right.v[3],right.v[2],right.v[1],right.v[0]);  
        return tmp += right;
    }

//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, int8_t right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, int16_t right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, int32_t right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, int64_t right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(int8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(int16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(int32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(int64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, uint8_t right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, uint16_t right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, uint32_t right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(const Decimal &left, uint64_t right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(uint8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(uint16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(uint32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator+(uint64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp += right;
//    }
//
//    //double / float
//    ARIES_HOST_DEVICE_NO_INLINE double operator+(const Decimal &left, float right) {
//        return left.GetDouble() + right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator+(const Decimal &left, double right) {
//        return left.GetDouble() + right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator+(float left, const Decimal &right) {
//        return left + right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator+(double left, const Decimal &right) {
//        return left + right.GetDouble();
//    }
//
//    // for sub
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcSubPrecision(const Decimal &d) {
//        CalcAddPrecision(d);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcSubTargetPrecision(const Decimal &d) {
//        CalcAddTargetPrecision(d);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerSubPrecision( const Decimal &d ) {
//        CalcInnerAddPrecision(d);
//    }
//
//    // op1 and op2 are positive
//    ARIES_HOST_DEVICE_NO_INLINE int32_t Decimal::CompareInt(int32_t *op1, int32_t *op2) {
//        int32_t res = 0;
//        #pragma unroll
//        for (int i = 0; i < NUM_TOTAL_DIG && res == 0; i++) {
//            res = op1[i] - op2[i];
//        }
//        return res;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::SubBothPositiveNums(Decimal &d) {
//        int sign = 1;
//        int32_t *p1 = (int32_t *) values, *p2 = (int32_t *) d.values;
//        AlignAddSubData(d);
//        int32_t r = CompareInt(p1, p2);
//        if (r == 0) {
//            #pragma unroll
//            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//                values[i] = 0;
//            }
//            return *this;
//        } else if (r < 0) {
//            int32_t *tmp;
//            tmp = p1;
//            p1 = p2;
//            p2 = tmp;
//            sign = -1;
//        }
//        //sub
//        int32_t carry = 0; //借位
//        for (int i = INDEX_LAST_DIG; i >= 0; i--) {
//            p1[i] -= p2[i];
//            p1[i] -= carry;
//            if (p1[i] < 0) {
//                p1[i] += PER_DEC_MAX_SCALE;
//                carry = 1;
//            } else {
//                carry = 0;
//            }
//        }
//        if (p1 != values) {
//            #pragma unroll
//            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//                values[i] = p1[i];
//            }
//        }
//        if (sign == -1) {
//            Negate();
//        }
//        return *this;
//    }
//
	//lixin  更改-=函数
   ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(const Decimal &d) {
	   
	    //printf("**************operator-=*************start\n"); 
		//printf("beforeAlign\n");
		//printf("operator-=::x :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",sign,prec,frac,v[4],v[3],v[2],v[1],v[0]);  
		//printf("operator-=::d :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",d.sign,d.prec,d.frac,d.v[4],d.v[3],d.v[2],d.v[1],d.v[0]);  
	
		//将d的值 拷贝到 added
	   Decimal added(d);
	   
	   //对齐
	   added.AlignAddSubData(*this);
	   
	   	//printf("afterAlign\n");
		//printf("operator-=::x :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",sign,prec,frac,v[4],v[3],v[2],v[1],v[0]);  
		//printf("operator-=::added :: sign = %d,prec = %d,frac = %d,v = %d %d %d %d %d\n",added.sign,added.prec,added.frac,added.v[4],added.v[3],added.v[2],added.v[1],added.v[0]);  
		
		//将减数取反
	   if(added.sign == 0){
			added.sign = 1;
	   }
	   else{
			added.sign = 0;
	   }
	   
	   int32_t temp[NUM_TOTAL_DIG];
		
		for(int i=0;i<5;i++){
			temp[i] = v[i];
		}
		
		if(sign ^ added.sign == 0){
		//	printf("operator-=::被减数与减数符号不同\n");
			//被减数与减数符号不同 符号不变 |a - b| = |a| + |b| 		
			abs_add(temp, added.v, v);
		}else{
		//	printf("operator-=::被减数与减数符号相同\n");
			//被减数与减数符号相同 符号变为绝对值 |a - b| = |a| - |b|  				
			abs_sub(temp,added.v, v);
			// ( |a| > |b| 并且 b是正数 ) || ( |a| < |b| 并且 b是负数 ) 时 结果为负数 反之 为正数
			sign = (abs_cmp(temp, added.v) > 0 && !added.sign) || (abs_cmp(temp, added.v) < 0 && added.sign);
		}

		//		printf("**************operator+=*************end\n"); 

       return *this;
   }
//
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(int8_t i) {
//        Decimal d(i);
//        return *this -= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(int16_t i) {
//        Decimal d(i);
//        return *this -= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(int32_t i) {
//        Decimal d(i);
//        return *this -= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(int64_t i) {
//        Decimal d(i);
//        return *this -= d;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(uint8_t i) {
//        Decimal d(i);
//        return *this -= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(uint16_t i) {
//        Decimal d(i);
//        return *this -= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(uint32_t i) {
//        Decimal d(i);
//        return *this -= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator-=(uint64_t i) {
//        Decimal d(i);
//        return *this -= d;
//    }
//
//    //double / float
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator-=(const float &f) {
//        return GetDouble() - f;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator-=(const double &l) {
//        return GetDouble() - l;
//    }
//
//    //self operator
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator--() {
//        Decimal d((int8_t) 1);
//        return *this -= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::operator--(int32_t) {
//        Decimal tmp(*this);
//        Decimal d((int8_t) 1);
//        return tmp -= d;
//    }
//

	//lixin  更改减法函数
   ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, const Decimal &right) {
       Decimal tmp(left);
       return tmp -= right;
   }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, int8_t right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, int16_t right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, int32_t right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, int64_t right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(int8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(int16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(int32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(int64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, uint8_t right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, uint16_t right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, uint32_t right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(const Decimal &left, uint64_t right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(uint8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(uint16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(uint32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator-(uint64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp -= right;
//    }
//
//    //double / float
//    ARIES_HOST_DEVICE_NO_INLINE double operator-(const Decimal &left, const float right) {
//        return left.GetDouble() - right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator-(const Decimal &left, const double right) {
//        return left.GetDouble() - right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator-(const float left, const Decimal &right) {
//        return left - right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator-(const double left, const Decimal &right) {
//        return left - right.GetDouble();
//    }
//
//    // for multiple
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerMulPrecision(const Decimal& d) {
//        uint16_t frac0 = aries_min(frac + d.frac, SUPPORTED_MAX_SCALE);
//        uint16_t frac1, frac2;
//        frac1 = frac2 = NEEDELEMENTS(frac0);
//        uint16_t intg0 = intg + d.intg;
//        uint16_t intg1, intg2;
//        intg1 = intg2 = NEEDELEMENTS(intg0);
//        uint8_t e = 0;
//        FIX_INTG_FRAC_ERROR(INNER_MAX_PRECISION_INT32_NUM, intg1, frac1, e);
//        SET_PREC_SCALE_VALUE(frac, frac0, frac1, frac2);
//        SET_PREC_SCALE_VALUE(intg, intg0, intg1, intg2);
//        SET_ERR(error, e);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcMulPrecision(const Decimal &d) {
//        uint16_t frac0 = aries_min(GET_CALC_FRAC(error) + GET_CALC_FRAC(d.error), SUPPORTED_MAX_SCALE);
//        uint16_t intg0 = GET_CALC_INTG(mode) + GET_CALC_INTG(d.mode);
//        uint8_t e = 0;
//        FIX_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
//        SET_CALC_INTG(mode, intg0);
//        SET_CALC_FRAC(error,frac0);
//        SET_ERR(error, e);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcMulTargetPrecision(const Decimal &d) {
//        uint16_t frac0 = aries_min(frac + d.frac, SUPPORTED_MAX_SCALE);
//        uint16_t intg0 = intg + intg;
//        uint8_t e = 0;
//        FIX_TAGET_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
//        intg = intg0;
//        frac = frac0;
//        error = e;
//    }
//
	//lixin  更改*=函数
   ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(const Decimal &d) {
	   
	   //printf("**********************var_mul start****************\n");
	   
	   //溢出位
	   int32_t overflow = 0;
	   //TODO prec尚未定义
       prec = prec;
	   
	   //  小数点位数 ( x * pow(10,x.frac) ) * ( y * pow(10,y.frac) ) = x * y * pow(10, x.frac + y.frac)
       frac = frac + d.frac;
	   //  符号
       sign = sign ^ d.sign;
	   	
	   //printf("var_mul::this::prec=:%d,frac=%d,sign=%d\n",prec,frac,sign);
		
		//中间结果
	   int32_t inner_res[NUM_TOTAL_DIG*2];
	   
	   //绝对相乘结果放在 inner_res 中
	   abs_mul(v, d.v, inner_res);
	   
	   //将结果赋值到res.v 
		for(int i = 0; i < NUM_TOTAL_DIG; i++){
			v[i] = inner_res[i];
		//	printf("var_mul::this::v=:%d\n",v[i]);
			
		}
		
		//TODO 判断溢出
		for(int i = NUM_TOTAL_DIG; i < 2*NUM_TOTAL_DIG; i++){
			overflow = (overflow || inner_res[i]);
		}
		
		//printf("**********************var_mul end****************\n");
       return *this;
   }
//
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(int8_t i) {
//        Decimal tmp(i);
//        return *this *= tmp;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(int16_t i) {
//        Decimal tmp(i);
//        return *this *= tmp;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(int32_t i) {
//        Decimal tmp(i);
//        return *this *= tmp;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(int64_t i) {
//        Decimal tmp(i);
//        return *this *= tmp;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(uint8_t i) {
//        Decimal tmp(i);
//        return *this *= tmp;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(uint16_t i) {
//        Decimal tmp(i);
//        return *this *= tmp;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(uint32_t i) {
//        Decimal tmp(i);
//        return *this *= tmp;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator*=(uint64_t i) {
//        Decimal tmp(i);
//        return *this *= tmp;
//    }
//
//    //double / float
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator*=(const float &f) {
//        return GetDouble() * f;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator*=(const double &d) {
//        return GetDouble() * d;
//    }
//

//    //two operators
	// lixin 乘法改写
   ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, const Decimal &right) {
       Decimal tmp(left);
       return tmp *= right;
   }
   
//
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, int8_t right) {
//        Decimal tmp(right);
//        return tmp *= left;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, int16_t right) {
//        Decimal tmp(right);
//        return tmp *= left;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, int32_t right) {
//        Decimal tmp(right);
//        return tmp *= left;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, int64_t right) {
//        Decimal tmp(right);
//        return tmp *= left;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(int8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp *= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(int16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp *= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(int32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp *= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(int64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp *= right;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, uint8_t right) {
//        Decimal tmp(right);
//        return tmp *= left;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, uint16_t right) {
//        Decimal tmp(right);
//        return tmp *= left;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, uint32_t right) {
//        Decimal tmp(right);
//        return tmp *= left;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(const Decimal &left, uint64_t right) {
//        Decimal tmp(right);
//        return tmp *= left;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(uint8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp *= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(uint16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp *= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(uint32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp *= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator*(uint64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp *= right;
//    }
//
//    //double / float
//    ARIES_HOST_DEVICE_NO_INLINE double operator*(const Decimal &left, const float right) {
//        return left.GetDouble() * right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator*(const Decimal &left, const double right) {
//        return left.GetDouble() * right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator*(const float left, const Decimal &right) {
//        return left * right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator*(const double left, const Decimal &right) {
//        return left * right.GetDouble();
//    }
//
//    // for division
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerDivPrecision(const Decimal& d) {
//        uint16_t frac0 = aries_min(frac + DIV_FIX_INNER_FRAC, SUPPORTED_MAX_SCALE);
//        int highestV1, highestV2, prec1 = GetRealPrecision(highestV1), prec2 = d.GetRealPrecision(highestV2);
//        int16_t intg0 = prec1 - frac - (prec2 - d.frac) + (highestV1 >= highestV2);
//        if (intg0 < 0) {
//            intg0 = 0;
//        }
//        uint16_t frac1, frac2;
//        frac1 = frac2 = NEEDELEMENTS(frac0);
//        uint16_t intg1, intg2;
//        intg1 = intg2 = NEEDELEMENTS(intg0);
//        uint8_t e = 0;
//        FIX_INTG_FRAC_ERROR(INNER_MAX_PRECISION_INT32_NUM, intg1, frac1, e);
//        SET_PREC_SCALE_VALUE(frac, frac0, frac1, frac2);
//        SET_PREC_SCALE_VALUE(intg, intg0, intg1, intg2);
//        SET_ERR(error, e);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcDivPrecision( const Decimal &d ) {
//        uint16_t frac0 = aries_min(GET_CALC_FRAC(error) + DIV_FIX_EX_FRAC, SUPPORTED_MAX_SCALE);
//        int highestV1, highestV2, prec1 = GetRealPrecision(highestV1), prec2 = d.GetRealPrecision(highestV2);
//        int16_t intg0 = prec1 - GET_CALC_FRAC(error) - (prec2 - GET_CALC_FRAC(d.error)) + (highestV1 >= highestV2);
//        if (intg0 < 0) {
//            intg0 = 0;
//        }
//        uint8_t e = 0;
//        FIX_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
//        SET_CALC_INTG(mode, intg0);
//        SET_CALC_FRAC(error,frac0);
//        SET_ERR(error, e);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcDivTargetPrecision( const Decimal &d ) {
//        uint16_t frac0 = aries_min(frac + DIV_FIX_EX_FRAC, SUPPORTED_MAX_SCALE);
//        uint16_t intg0 = aries_min(intg + d.frac, SUPPORTED_MAX_PRECISION);
//        uint8_t e = 0;
//        FIX_TAGET_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
//        intg = intg0;
//        frac = frac0;
//        error = e;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator>>(int n) {
//        int shiftDigits = n % DIG_PER_INT32;
//        int shiftInt = n / DIG_PER_INT32;
//        if (shiftDigits) {
//            int lower = GetPowers10(shiftDigits);
//            int higher = GetPowers10( DIG_PER_INT32 - shiftDigits);
//            int carry = 0, temp = 0;
//            #pragma unroll
//            for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//                if (values[i] != 0) {
//                    temp = values[i] % lower;
//                    values[i] = values[i] / lower;
//                } else {
//                    temp = 0;
//                }
//                if (carry) {
//                    values[i] += carry * higher;
//                }
//                carry = temp;
//            }
//        }
//        if (shiftInt) {
//            for (int i = INDEX_LAST_DIG; i >= shiftInt; i--) {
//                values[i] = values[i - shiftInt];
//            }
//            for (int i = 0; i < shiftInt; i++) {
//                values[i] = 0;
//            }
//        }
//        //for check
//        for (int i = 0; i < shiftInt; i++) {
//            assert(values[i] == 0);
//        }
//        if (shiftDigits) {
//            int lower = GetPowers10(shiftDigits);
//            assert(values[shiftInt] / lower == 0);
//        }
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator<<(int n) {
//        int shiftDigits = n % DIG_PER_INT32;
//        int shiftInt = n / DIG_PER_INT32;
//        int lower = GetPowers10( DIG_PER_INT32 - shiftDigits);
//        int higher = GetPowers10(shiftDigits);
//        if (shiftDigits) {
//            int carry = 0, temp = 0;
//            for (int i = INDEX_LAST_DIG; i >= 0; i--) {
//                if (values[i] != 0) {
//                    temp = values[i] / lower;
//                    values[i] = values[i] % lower * higher;
//                } else {
//                    temp = 0;
//                }
//                if (carry) {
//                    values[i] += carry;
//                }
//                carry = temp;
//            }
//        }
//        if (shiftInt) {
//            for (int i = 0; i < NUM_TOTAL_DIG - shiftInt; i++) {
//                values[i] = values[i + shiftInt];
//            }
//            for (int i = NUM_TOTAL_DIG - shiftInt; i < NUM_TOTAL_DIG; i++) {
//                values[i] = 0;
//            }
//        }
//        intg += n;
//        return *this;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::UpdateIntgDigits() {
//        int validPos = 0;
//        for ( validPos = 0; validPos < NUM_TOTAL_DIG; ++validPos )
//        {
//            if (values[validPos]) {
//                break;
//            }
//        }
//        int intg0 = NUM_TOTAL_DIG - validPos - NEEDELEMENTS(frac);
//        if (intg0 <= 0) {
//            intg = 0;
//        } else {
//            int v = aries_abs(values[validPos]);
//            int digit = 1;
//            while(v >= GetPowers10(digit) && ++digit < DIG_PER_INT32);
//            intg = (intg0 - 1) * DIG_PER_INT32 + digit;
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE int Decimal::GetRealPrecision(int &highestValue) const {
//        int validPos = 0;
//        for ( ; validPos < NUM_TOTAL_DIG; ++validPos )
//        {
//            if (values[validPos]) {
//                break;
//            }
//        }
//        int prec0 = NUM_TOTAL_DIG - validPos;
//        if (prec0 <= 0) {
//            highestValue = 0;
//            return 0;
//        }
//        int v = aries_abs(values[validPos]);
//        int digit = 1;
//        while(v >= GetPowers10(digit) && ++digit < DIG_PER_INT32);
//        highestValue = v / GetPowers10(digit - 1);
//        if (frac == 0) {
//            return digit + (prec0 - 1) * DIG_PER_INT32;
//        } else {
//            int lastFrac = frac % DIG_PER_INT32;
//            return digit + (prec0 - 2) * DIG_PER_INT32 + (lastFrac == 0 ? DIG_PER_INT32 :  lastFrac);
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckAndSetCalcPrecision() {
//        CheckAndSetRealPrecision();
//        if (GET_CALC_FRAC(error) == 0 && GET_CALC_INTG(mode) == 0) {
//            SET_CALC_FRAC(error, frac);
//            SET_CALC_INTG(mode, intg);
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckAndSetRealPrecision() {
//        int highest;
//        int prec = GetRealPrecision(highest);
//        intg = prec - frac;
//        if ((intg & 0x80) > 0) {
//            intg = 0;
//        }
//        if (intg == 0 && frac == 0) {
//            intg = 1;
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE int Decimal::GetRealIntgSize(int &highestValue) const {
//        int validPos = 0;
//        for ( ; validPos < NUM_TOTAL_DIG; ++validPos )
//        {
//            if (values[validPos]) {
//                break;
//            }
//        }
//        int intg0 = NUM_TOTAL_DIG - validPos - NEEDELEMENTS(frac);
//        if (intg0 <= 0) {
//            highestValue = 0;
//            intg0 = 0;
//        } else {
//            highestValue = values[validPos];
//        }
//        return intg0;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::GenIntDecimal(int shift) {
//        int n = shift;
//        if (frac) {
//            n -= DIG_PER_INT32 - frac % DIG_PER_INT32;
//        }
//        if (n > 0) {
//            *this << n;
//        } else if (n < 0) {
//            *this >> (-n);
//        }
//        frac = 0;
//        UpdateIntgDigits();
//        return *this;
//    }
//
	//lixin  更改取中值函数
   ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::HalfIntDecimal(const Decimal d1, const Decimal d2) {
       Decimal tmp(d1);
       tmp += d2;
       int32_t rds = 0;
       int64_t t[NUM_TOTAL_DIG];
       #pragma unroll
       for (int i = 0; i < NUM_TOTAL_DIG; i++) {
           t[i] = tmp.v[i];
       }
	   
       #pragma unroll
       for (int i = NUM_TOTAL_DIG-1; i >=0 ; i--) {
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
           tmp.v[i] = t[i];
       }
       return tmp;
   }
//
	//lixin  更改DivInt函数
   ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::DivInt(const Decimal ds, const Decimal dt, Decimal &residuel) {
	   
	   // printf("\n*****************进入DivInt*******************\n");
		// printf("被除数::frac = %d,values = %09d %09d %09d %09d %09d\n",ds.frac,ds.v[4],ds.v[3],ds.v[2],ds.v[1],ds.v[0]);
        // printf("  除数::frac = %d,values = %09d %09d %09d %09d %09d\n",dt.frac,dt.v[4],dt.v[3],dt.v[2],dt.v[1],dt.v[0]);
          
			   Decimal zero("0");
		
		//被除数为0
       if (ds.isZero()) {
		   aries_memset(residuel.v, 0x00, sizeof(residuel.v));
           return ds;
       }
	   
	   //确定商的范围
       int q = getDecimalLen(ds.v) - getDecimalLen(dt.v);
	 
	   char qmaxNum[NUM_TOTAL_DIG * DIG_PER_INT32];
	   char qminNum[NUM_TOTAL_DIG * DIG_PER_INT32];
	   qminNum[0] = '1';
	   qmaxNum[0] = '9';
	   for(int i=1; i<q ; i++){
			qmaxNum[i] = '9';
			qminNum[i] = '0';
	   }
	   qmaxNum[q] ='9';
	   qmaxNum[q+1] = '\0';
	   qminNum[q] = '\0';
	   
       Decimal qmax(qmaxNum), qmin(qminNum), qmid("0"), rsdmax("0"), rsdmin("0"), rsdtmp("0");
       
	   
	   // printf("qmax::frac = %d,values = %d %d %d %d %d\n",qmax.frac,qmax.v[4],qmax.v[3],qmax.v[2],qmax.v[1],qmax.v[0]);
       // printf("qmin::frac = %d,values = %d %d %d %d %d\n",qmin.frac,qmin.v[4],qmin.v[3],qmin.v[2],qmin.v[1],qmin.v[0]);
        
       rsdmax = ds - qmax * dt;
	   
	
	   
       if ( rsdmax.sign == 0){
           residuel = rsdmax;
           return qmax;
       }
	   
       rsdmin = ds - qmin * dt;
	   
       if (abs_cmp(rsdmin.v,zero.v) == 0) {
           residuel = 0;
           return qmin;
       }
       assert(abs_cmp(rsdmin.v,zero.v) > 0);

	   //利用二分找出商
       // clock_t st, ed, acc = 0;
       int iter = 0;
       // st = clock();
       while (abs_cmp(qmin.v,qmax.v) < 0) {
		   // printf("\n进入qmin、qmax循环\n");
           iter++;
 
           qmid = HalfIntDecimal(qmax, qmin);
		   
		   // printf("qmin::frac = %d,values = %d %d %d %d %d\n",qmin.frac,qmin.v[4],qmin.v[3],qmin.v[2],qmin.v[1],qmin.v[0]);
		   // printf("qmax::frac = %d,values = %d %d %d %d %d\n",qmax.frac,qmax.v[4],qmax.v[3],qmax.v[2],qmax.v[1],qmax.v[0]);
		   // printf("qmid::frac = %d,values = %d %d %d %d %d\n",qmid.frac,qmid.v[4],qmid.v[3],qmid.v[2],qmid.v[1],qmid.v[0]);
		   
           // st = clock();
           if (abs_cmp(qmid.v,qmin.v) == 0) {
               break;
           }
           // ed = clock();
           // acc += (ed - st);
           rsdtmp = ds - qmid * dt;
		   
		   // printf("rsdtmp::sign = %d,values = %d %d %d %d %d\n",rsdtmp.sign,rsdtmp.v[4],rsdtmp.v[3],rsdtmp.v[2],rsdtmp.v[1],rsdtmp.v[0]);
		   
           // st = clock();
           if (rsdtmp.isZero()) {
			   aries_memset(rsdmin.v, 0x00, sizeof(rsdmin.v));
			   qmin.CopyValue(qmid);
               break;
           } else if (rsdtmp.sign==0) {
               rsdmin = rsdtmp;
               qmin = qmid;
           } else {
               rsdmax = rsdtmp;
               qmax = qmid;
           }
           // ed = clock();
           // acc += (ed - st);
       }
       // ed = clock();
       // printf("    divInt loop %ld, iter %d times\n", ed - st, iter);
       residuel = rsdmin;
       return qmin;
   }
//
	//lixin 改写按int位相除法
   ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::DivByInt(const Decimal &d, int shift, bool isMod) {
       int dvt = d.v[0];	//INDEX_LAST_DIG = 4
       int remainder = 0;
	   
	  int32_t temp[NUM_TOTAL_DIG];
		aries_memcpy(temp, v, sizeof(v));
		abs_lshift(temp,NUM_TOTAL_DIG,shift,v);

       #pragma unroll
       for (int i = INDEX_LAST_DIG; i >=0 ; i--) {
           if (v[i] || remainder) {
               int64_t tmp = (int64_t) v[i] + (int64_t) remainder * PER_DEC_MAX_SCALE;
               v[i] = tmp / dvt;
               remainder = tmp % dvt;
           }
       }
	   
	   
       if (isMod) {
           // *this = remainder;
       } else {
		   if (remainder * 10 - dvt > 0 ) {
			   Decimal oneNUM("1");
				*this += oneNUM;
		   }
       }
       return *this;
   }
//
	//lixin 改写转换为int64相除法
   ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::DivByInt64(const Decimal &divisor, int shift, bool isMod) {
       int64_t dvs = ToInt64();
       while (shift > DIG_PER_INT32) {
           dvs *= GetPowers10(DIG_PER_INT32);
           shift -= DIG_PER_INT32;
       }
       dvs *= GetPowers10(shift);
       int64_t dvt = divisor.ToInt64();
       int64_t res = isMod ? (dvs % dvt) : (dvs / dvt + (((dvs % dvt) << 1) >= dvt ? 1 : 0));
	   
	   int i=0;
	   while(res>PER_DEC_MAX_SCALE){
			v[i++] = res % PER_DEC_MAX_SCALE;
			res = res / PER_DEC_MAX_SCALE;
	   }
	   v[i] = res;
       return *this;
   }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::Negate() {
//        #pragma unroll
//        for (int i = 0; i < NUM_TOTAL_DIG; i++) {
//            values[i] = -values[i];
//        }
//        return *this;
//    }
//
	
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::IntToFrac(int fracDigits) {
//        int frac0 = NEEDELEMENTS(fracDigits);
//
//        Decimal intgPart(*this);
//        intgPart >> (fracDigits);
//        Decimal fracPart(*this);
//        fracPart << ( DIG_PER_INT32 * NUM_TOTAL_DIG - fracDigits);
//        for (int i = 0; i < NUM_TOTAL_DIG - frac0; i++) {
//            values[i] = intgPart.values[i + frac0];
//        }
//        int fracBase = NUM_TOTAL_DIG - frac0;
//        for (int i = fracBase; i < NUM_TOTAL_DIG; i++) {
//            values[i] = fracPart.values[i - fracBase];
//        }
//        frac = fracDigits;
//        UpdateIntgDigits();
//        return *this;
//    }
//
	//lixin 未改动
   ARIES_HOST_DEVICE_NO_INLINE void Decimal::CopyValue(Decimal &d) {
       #pragma unroll
       for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
           v[i] = d.v[i];
       }
   }
//
	//lixin 改写DivOrMod函数
   ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::DivOrMod( const Decimal &d, bool isMod ) {

		// printf("*****************DivOrMod开始***************\n");
		// printf("1.0::被除数::frac = %d,values = %09d %09d %09d %09d %09d\n",frac,v[4],v[3],v[2],v[1],v[0]);
		// printf("1.0::  除数::frac = %d,values = %09d %09d %09d %09d %09d\n",d.frac,d.v[4],d.v[3],d.v[2],d.v[1],d.v[0]);
       
		//将 被除数与除数 分别拷贝在 divitend 和 divisor中
       Decimal divitend(*this);
       Decimal divisor(d);
	   
	    if (isZero()) {
           return *this;
       }
	   //TODO 除数为零
		if (d.isZero()) {
           return *this;
       }

	   	 //确定符号
		sign = sign ^ d.sign;
		
		//判断是否为mod
       if (isMod)
       {
           // CalcModPrecision(divisor);
           // CalcInnerModPrecision(divisor);
       } else {
		   //计算精度 遵循old.cu的精度
           frac = frac - DIV_FIX_INNER_FRAC;
       }
	
		// printf("\n符号、精度计算之后\n");
		// printf("this::sign = %d,frac = %d\n",sign,frac);
		
		//被除数为零直接返回0
       
	   //TODO 除数为零


		
		//将被除数与除数化为整数
		divitend.frac = 0;
		divisor.frac = 0;
		divitend.sign = 0;
		divisor.sign = 0;
		
		// printf("\n化为整数之后\n");
		// printf("2.0::被除数::divitend::frac = %d,values = %09d %09d %09d %09d %09d\n",divitend.frac,divitend.v[4],divitend.v[3],divitend.v[2],divitend.v[1],divitend.v[0]);
        // printf("2.0::  除数:: divisor::frac = %d,values = %09d %09d %09d %09d %09d\n",divisor.frac,divisor.v[4],divisor.v[3],divisor.v[2],divisor.v[1],divisor.v[0]);
		
		//提高精度 多出6个精度 遵循old
       int shift = -(d.frac) + DIV_FIX_INNER_FRAC;
	// printf("shift = d.frac + DIV_FIX_INNER_FRAC = %d + %d = %d\n",d.frac,DIV_FIX_INNER_FRAC,shift);
	
       if (!isMod) {
           // TODO 不够除怎么办
           // if (divitend.intg + shift < divisor.intg) {
               // aries_memset(values, 0x00, sizeof(values));
               // return *this;
           // }
       } else {
           shift = 0;
       }
       
       Decimal res("0");
		if (getDecimalLen(divitend.v) + shift <= DIG_PER_INT64 &&  getDecimalLen(divisor.v)<= DIG_PER_INT64) { 
			// printf("转化为64位计算\n");
            res = divitend.DivByInt64(divisor, shift, isMod);
        } else if (getDecimalLen(divisor.v) <= DIG_PER_INT32) {
			// printf("转化为32位计算\n");
            res = divitend.DivByInt(divisor, shift, isMod);
        }
       else{
		   // printf("二分计算\n");
		   
		   //待左移的量
           int tmpEx = shift;
		   //左移量 因为不能一下子完成左移 可能需要分多次
           int nDigits = 0;
	
			// printf("tmpEx = shift = %d\n",shift);
			
			//tmpRes保存中间结果
           Decimal tmpRes("0");
           for (; tmpEx > 0;) {
               //iter++;
			   
			   //获取被除数的值的长度
               int divitendLen = getDecimalLen(divitend.v);
			   //这样能算出一次性最多左移的大小	-1此处遵循old
               nDigits = INNER_MAX_PRECISION - divitendLen - 1;
			   
			   // printf("nDigits = INNER_MAX_PRECISION - divitendLen - 1 = %d - %d - 1 = %d\n",INNER_MAX_PRECISION,divitendLen,nDigits);
				
				//可左移的量 比 待左移的量大
               if (nDigits > tmpEx) {
				   // printf("nDigits = %d > tmpEx = %d  :: nDigits = tmpEx = %d\n",nDigits,tmpEx,tmpEx);
                   nDigits = tmpEx;
               }
			   //此次左移 nDigits 
               tmpEx -= nDigits;
			   
			   //左移
			   int32_t temp[NUM_TOTAL_DIG];
			   aries_memcpy(temp, divitend.v, sizeof(v));
			   abs_lshift(temp,NUM_TOTAL_DIG,nDigits,divitend.v);
			   // printf("divitend左移nDigits位::frac = %d,values = %09d %09d %09d %09d %09d\n",divitend.frac,divitend.v[4],divitend.v[3],divitend.v[2],divitend.v[1],divitend.v[0]);
            
				//除法
               tmpRes = DivInt(divitend, divisor, divitend);
				// printf("DivInt::的结果  商 ::   tmpRes::frac = %d,values = %09d %09d %09d %09d %09d\n",tmpRes.frac,tmpRes.v[4],tmpRes.v[3],tmpRes.v[2],tmpRes.v[1],tmpRes.v[0]);
				// printf("DivInt::的结果 余数:: divitend::frac = %d,values = %09d %09d %09d %09d %09d\n",divitend.frac,divitend.v[4],divitend.v[3],divitend.v[2],divitend.v[1],divitend.v[0]);
				// printf("****************DivInt  结束********************\n\n");

               if (!res.isZero()) {
				   //res左移nDigits位
				   // printf("res::before::frac = %d,values = %09d %09d %09d %09d %09d\n",res.frac,res.v[4],res.v[3],res.v[2],res.v[1],res.v[0]);
				    int32_t tempRes[NUM_TOTAL_DIG];
					aries_memcpy(tempRes, res.v, sizeof(v));
					abs_lshift(tempRes,NUM_TOTAL_DIG,nDigits,res.v);
					
					// printf("res::after::frac = %d,values = %09d %09d %09d %09d %09d\n",res.frac,res.v[4],res.v[3],res.v[2],res.v[1],res.v[0]);
               }
			   // ( a * pow(10,t) + b) / c = a/c*pow(10,t) + b/c 
               res += tmpRes;
			   // printf("res+=tmpRes::frac = %d,values = %09d %09d %09d %09d %09d\n",res.frac,res.v[4],res.v[3],res.v[2],res.v[1],res.v[0]);
              
           }
		   // printf("res+=tmpRes::frac = %d,values = %09d %09d %09d %09d %09d\n",res.frac,res.v[4],res.v[3],res.v[2],res.v[1],res.v[0]);
              
           //check if need round up
           if (isMod) {
               res = divitend;
           } else {
			   //进行四舍五入
			   Decimal doubleDivitend = divitend + divitend;
               if (abs_cmp(doubleDivitend.v , divisor.v) >0) {
                   Decimal oneNUM("1");
				   res += oneNUM;
               }
           }
		}
		   // printf("res+=tmpRes::frac = %d,values = %09d %09d %09d %09d %09d\n",res.frac,res.v[4],res.v[3],res.v[2],res.v[1],res.v[0]);
              
		//将res的值赋值到this中	
       CopyValue(res);

       return *this;
   }
//
	//lixin 改写/=函数
   ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(const Decimal &d) {
       return DivOrMod(d);
   }
//
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(int8_t i) {
//        Decimal d(i);
//        return *this /= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(int16_t i) {
//        Decimal d(i);
//        return *this /= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(int32_t i) {
//        Decimal d(i);
//        return *this /= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(int64_t i) {
//        Decimal d(i);
//        return *this /= d;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(uint8_t i) {
//        Decimal d(i);
//        return *this /= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(uint16_t i) {
//        Decimal d(i);
//        return *this /= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(uint32_t i) {
//        Decimal d(i);
//        return *this /= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator/=(uint64_t i) {
//        Decimal d(i);
//        return *this /= d;
//    }
//
//    //double / float
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator/=(const float &f) {
//        return GetDouble() / f;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator/=(const double &d) {
//        return GetDouble() / d;
//    }
//
//    //two operators
	//lixin 改写/函数
   ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, const Decimal &right) {
       Decimal tmp(left);
       return tmp /= right;
   }
//
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, int8_t right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, int16_t right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, int32_t right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, int64_t right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(int8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(int16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(int32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(int64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, uint8_t right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, uint16_t right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, uint32_t right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(const Decimal &left, uint64_t right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(uint8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(uint16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(uint32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator/(uint64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp /= right;
//    }
//
//    //double / float
//    ARIES_HOST_DEVICE_NO_INLINE double operator/(const Decimal &left, const float right) {
//        return left.GetDouble() / right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator/(const Decimal &left, const double right) {
//        return left.GetDouble() / right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator/(const float left, const Decimal &right) {
//        return left / right.GetDouble();
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator/(const double left, const Decimal &right) {
//        return left / right.GetDouble();
//    }
//
//    // for mod
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcModPrecision( const Decimal &d ) {
//        int i = 0;
//        uint8_t frac0 = GET_CALC_FRAC(error), frac1 = GET_CALC_FRAC(d.error), intg0;
//        if (frac0 < frac1) {
//            frac0 = frac1;
//        } else {
//            i = frac0 - frac1;
//        }
//        intg0 = GET_CALC_INTG(d.mode) + i;
//        SET_CALC_INTG(mode, intg0);
//        SET_CALC_FRAC(error, frac0);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcModTargetPrecision( const Decimal &d ) {
//        int i = 0;
//        uint8_t frac0 = frac, frac1 = d.frac, intg0;
//        if (frac0 < frac1) {
//            frac0 = frac1;
//        } else {
//            i = frac0 - frac1;
//        }
//        intg0 = d.intg + i;
//        uint8_t e;
//        FIX_TAGET_INTG_FRAC_ERROR(SUPPORTED_MAX_PRECISION, intg0, frac0, e);
//        intg = intg0;
//        frac = frac0;
//        error = e;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcInnerModPrecision( const Decimal &d ) {
//        int i = 0;
//        if (frac < d.frac) {
//            frac = d.frac;
//        } else {
//            i = frac - d.frac;
//        }
//        intg = d.intg + i;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(const Decimal& d) {
//        return DivOrMod(d, true);
//    }
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(int8_t i) {
//        Decimal d(i);
//        return *this %= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(int16_t i) {
//        Decimal d(i);
//        return *this %= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(int32_t i) {
//        Decimal d(i);
//        return *this %= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(int64_t i) {
//        Decimal d(i);
//        return *this %= d;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(uint8_t i) {
//        Decimal d(i);
//        return *this %= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(uint16_t i) {
//        Decimal d(i);
//        return *this %= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(uint32_t i) {
//        Decimal d(i);
//        return *this %= d;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal &Decimal::operator%=(uint64_t i) {
//        Decimal d(i);
//        return *this %= d;
//    }
//
//    //double % float
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator%=(const float &f) {
//        return fmod(GetDouble(), f);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::operator%=(const double &d) {
//        return fmod(GetDouble(), d);
//    }
//
//    //two operators
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    //signed
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, int8_t right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, int16_t right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, int32_t right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, int64_t right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(int8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(int16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(int32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(int64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    //unsigned
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, uint8_t right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, uint16_t right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, uint32_t right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(const Decimal &left, uint64_t right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(uint8_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(uint16_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(uint32_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE Decimal operator%(uint64_t left, const Decimal &right) {
//        Decimal tmp(left);
//        return tmp %= right;
//    }
//
//    //double % float
//    ARIES_HOST_DEVICE_NO_INLINE double operator%(const Decimal &left, const float right) {
//        return fmod(left.GetDouble(), right);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator%(const Decimal &left, const double right) {
//        return fmod(left.GetDouble(), right);
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator%(const float left, const Decimal &right) {
//        return fmod((double)left, right.GetDouble());
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE double operator%(const double left, const Decimal &right) {
//        return fmod((double)left, right.GetDouble());
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isFracZero() const {
//        for (int i = INDEX_LAST_DIG - NEEDELEMENTS(frac); i <= INDEX_LAST_DIG; ++i) {
//            if (values[i]) {
//                return false;
//            }
//        }
//        return true;
//    }
//
	//lixin 判断是否为0
   ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isZero() const {
       for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
           if (v[i] != 0) {
               return false;
           }
       }
       return true;
   }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isLessZero() const {
//        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
//            if (values[i] < 0) {
//                return true;
//            }
//        }
//        return false;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isLessEqualZero() const {
//        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
//            if (values[i] > 0) {
//                return false;
//            }
//        }
//        return true;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isGreaterZero() const {
//        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
//            if (values[i] > 0) {
//                return true;
//            }
//        }
//        return false;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isGreaterEqualZero() const {
//        for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
//            if (values[i] < 0) {
//                return false;
//            }
//        }
//        return true;
//    }
//
	//10的i次方
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
//
//    ARIES_HOST_DEVICE_NO_INLINE int32_t Decimal::GetFracMaxTable(int i) const {
//        int32_t res = 0;
//        switch (i) {
//            case 0:
//                res = 900000000;
//                break;
//            case 1:
//                res = 990000000;
//                break;
//            case 2:
//                res = 999000000;
//                break;
//            case 3:
//                res = 999900000;
//                break;
//            case 4:
//                res = 999990000;
//                break;
//            case 5:
//                res = 999999000;
//                break;
//            case 6:
//                res = 999999900;
//                break;
//            case 7:
//                res = 999999990;
//                break;
//            case 8:
//                res = 999999999;
//                break;
//            default:
//                break;
//        }
//        return res;
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::GenMaxDecByPrecision() {
//        int index = NUM_TOTAL_DIG - NEEDELEMENTS(intg) - NEEDELEMENTS(frac);
//        // clear no use values
//        for (int i = 0; i < index; i++) {
//            values[i] = 0;
//        }
//        int firstDigits = intg % DIG_PER_INT32;
//        if (firstDigits) {
//            values[index++] = GetPowers10(firstDigits) - 1;
//        }
//        int32_t overPerDec = PER_DEC_MAX_SCALE - 1;
//        for (int i = index; i < NUM_TOTAL_DIG; i++) {
//            values[i] = overPerDec;
//        }
//        //replace last frac if necessary
//        if (frac) {
//            int lastDigits = frac % DIG_PER_INT32;
//            if (lastDigits) {
//                values[INDEX_LAST_DIG] = GetFracMaxTable(lastDigits - 1);
//            }
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::GenMinDecByPrecision() {
//        int index = NUM_TOTAL_DIG - NEEDELEMENTS(intg) - NEEDELEMENTS(frac);
//        // clear no use values
//        for (int i = 0; i < index; i++) {
//            values[i] = 0;
//        }
//        if (intg) {
//            int firstDigits = intg % DIG_PER_INT32;
//            if (firstDigits) {
//                values[index++] = GetPowers10(firstDigits - 1);
//            } else {
//                values[index++] = GetPowers10( DIG_PER_INT32 - 1);
//            }
//        } else if (frac) {
//            values[index++] = GetPowers10( DIG_PER_INT32 - 1);
//        }
//        for (int i = index; i < NUM_TOTAL_DIG; i++) {
//            values[i] = 0;
//        }
//    }
//
//    ARIES_HOST_DEVICE_NO_INLINE void Decimal::TransferData(const Decimal *v) {
//        if (intg >= v->intg && frac >= v->frac) {
//            SET_MODE(mode, GET_MODE(v->mode));
//            SET_ERR(error, GET_ERR(v->error));
//            int shift = NEEDELEMENTS(frac) - NEEDELEMENTS(v->frac);
//            for (int i = shift; i < NUM_TOTAL_DIG; i++) {
//                values[i - shift] = v->values[i];
//            }
//        } else {
//            assert(0);
//            SET_MODE(mode, GET_MODE(v->mode));
//            SET_ERR(error, ERR_OVER_FLOW);
//        }
//        assert(intg + frac <= SUPPORTED_MAX_PRECISION && frac <= SUPPORTED_MAX_SCALE);
//    }
//
	//lixin  更改对齐函数
   ARIES_HOST_DEVICE_NO_INLINE void Decimal::AlignAddSubData(Decimal &d) {
	   //printf("**************AlignAddSubData************* start\n");
	   
	   //v 数组 的长度
	   int len = NUM_TOTAL_DIG;
	   
	   //如果两个数本身就已经对齐直接返回
       if (frac == d.frac) {
           //do nothing
		   //printf("frac == d.frac\n");
           return;
       }

	   //小阶向大阶对齐	-6 -3 将 -3 变为 -6
       if (frac > d.frac) {
		   //printf("frac = %d <  d.frac = %d\n",frac,d.frac);
		   //将 v 的值复制到新的数组中
           int32_t temp[NUM_TOTAL_DIG];
		   aries_memcpy(temp, v, sizeof(v));
		   //printf("yy 左移 %d 位\n",frac-d.frac);
		   //将 新的数组temp 左移后 再 复制到 v中
           abs_lshift(temp,len,frac-d.frac,v);
		   //更改frac
		   frac = d.frac;
       } else {
		   //printf("frac = %d >  d.frac = %d\n",frac,d.frac);
		   //将 d.v 的值复制到新的数组中
            int32_t temp[NUM_TOTAL_DIG];
			aries_memcpy(temp, d.v, sizeof(d.v));
			//printf("xx 左移 %d 位\n",d.frac-frac);
			abs_lshift(temp,len,d.frac-frac,d.v);
			//将 新的数组temp 左移后 再 复制到 d.v中
			//更改frac
			d.frac = frac;
       }
	    //printf("**************AlignAddSubData************* end\n");
   }
//

//
//    ARIES_HOST_DEVICE_NO_INLINE double Decimal::GetDouble() const {
//        double z = 0;
//        int frac0 = NEEDELEMENTS(frac);
//        for (int i = 0; i < NUM_TOTAL_DIG - frac0; i++) {
//            if (values[i]) {
//                z += values[i];
//            }
//            if (z) {
//                z *= PER_DEC_MAX_SCALE;
//            }
//        }
//        //handle scale part
//        double s = 0;
//        for (int i = NUM_TOTAL_DIG - frac0; i < NUM_TOTAL_DIG; i++) {
//            if (values[i]) {
//                s += values[i];
//            }
//            if (s) {
//                s *= PER_DEC_MAX_SCALE;
//            }
//        }
//        for (int i = 0; i < frac0; i++) {
//            s /= PER_DEC_MAX_SCALE;
//        }
//        z += s;
//        return z / PER_DEC_MAX_SCALE;
//    }
//
	//lixin 改写转换为int64
   ARIES_HOST_DEVICE_NO_INLINE int64_t Decimal::ToInt64() const {
       //only 2 digits are valid and no frac part
       int64_t res = v[0];	//INDEX_LAST_DIG=4
       if (v[1]) {
           res += (int64_t) v[1] * PER_DEC_MAX_SCALE;
       }
       return res;
   }
//    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::CheckIfValidStr2Dec(char * str)
//    {
//        if (*str == '-') ++str;
//        for ( int i = 0; i < aries_strlen(str); ++i )
//        {
//            if (aries_is_digit(str[i]))
//            {
//                continue;
//            }
//            if (str[i] == '.')
//            {
//                continue;
//            }
//            return false;
//        }
//        return true;
//    }

	//lixin  更改初始化函数
    ARIES_HOST_DEVICE_NO_INLINE void Decimal::initialize(uint32_t ig, uint32_t fc, uint32_t m) {
        
		//printf("initialize *** start\n");
		sign = ig;
        prec = fc;
        frac = m;
        
        aries_memset(v, 0x00, sizeof(v));
		//printf("%d %d %d %d\n",sign,prec,frac,v[0]);
		//printf("initialize *** end\n");
    }
    
	//lixin  更改string转decimal函数
    ARIES_HOST_DEVICE_NO_INLINE bool Decimal::StringToDecimal( char * str )
    {
		//printf("StringToDecimal *** start\n");
        //如果检验数字格式不合格	
//        if (!CheckIfValidStr2Dec(str))
//        {
//            return false;
//        }
        
		//转化为decimal
		if(str[0] == '-'){
			sign = 1;
			++str;
		}else
			sign = 0;

		char *pos = aries_strchr(str, '.');
		
		//printf("beforestr = %s\n",str);
		if(pos == 0){
			frac = 0;
		}else{
			frac = -(aries_strlen(str) - (pos - str)  - 1);
			str = aries_strerase(str,(pos - str));
		}
		//printf("afterstr = %s\n",str);
		
		//printf("%d %d %d %d\n",sign,prec,frac,v[0]);
		
		int i = 0;
		int n = aries_strlen(str) / DIG_PER_INT32;
		int mod = aries_strlen(str) % DIG_PER_INT32;
		if( mod != 0){
			v[n] = aries_atoi(str,str+mod);
			str += mod;
		}
		while(aries_strlen(str) >= DIG_PER_INT32){
			i++;
			v[n-i] = aries_atoi(str,str+DIG_PER_INT32);
			str += DIG_PER_INT32;
		}
		
		for( i =0 ; i < 5 ; i++){
			//printf("v[%d] = %d\n",i,v[i]);
		}	
		//printf("StringToDecimal *** end\n");
        return true;
    }



} //end namespace aries_acc

/**/
