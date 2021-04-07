#ifndef _UTILS_H
#define _UTILS_H

#include <cstdio>
#include <functional>
#include <sys/time.h>
#include <vector>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*
 * Usage:
 *   CPUTimer cpuTimer;
 *   double cpuPerf = cpuTimer.timing( [&](){
 *        *** CODE ***
 *   });
 */
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


/*
 * Usage:
 *  GPUTimer gpuTimer;
 *  float gpuPerf = gpuTimer.timing( [&](){
 *       *** CODE ***
 *       kernel<<<m, n>>>(args, ...);
 *       *** CODE ***
 *  });
 */
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

/*
 * readLines: read and process lines from a file
 *  datafile: the input file
 *       lnr: the number of lines read from the file
 *  lineProc: the processing function, which is supposed to be lineProc(std::string)
 */
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

/*
 * extract fields from a line
 *    line: the input line
 *     idx: the indices of fields that are extracted
 *   start: the start field index
 * firstContainer and containers: the containers hold the extracted fields
 *
 * Usage:
 *  readLines(datafile, (uint64_t)-1, [&](std::string l) {
 *       extractFields(l, {4, 5, 6, 7}, 0, q_str, e_str, d_str, t_str);
 *   });
 *
 *  Read lines from the datafile, and for each line, field 4, 5, 6, 7 are appended into q_str, e_str, d_str, t_str;
 */

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


#endif
