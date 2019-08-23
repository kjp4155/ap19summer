// TIMESTAMP V2
#define MAXDEV 4
#define MAX_BATCH_SIZE 256
int NDEV = 1;
int BATCH_SIZE = 64;

#define CL_SILENCE_DEPRECATION

#include <CL/cl.h>

#include "music.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

//////////////////////// CUSTOM TIMER DEFINITION START /////////////////////////

#include <time.h>
#define MAXTIMER 64

int timerCnt = 0;
char timerName[MAXTIMER][128];
struct timespec startTime[MAXTIMER];
double totalTime[MAXTIMER];
void startTimer(char *s) {
    for (int i = 0; i < timerCnt; i++) {
        if( strcmp(timerName[i], s) == 0 ){
            clock_gettime(CLOCK_REALTIME, &startTime[i]);
            return;
        }
    }
    strcpy(timerName[timerCnt], s);
    clock_gettime(CLOCK_REALTIME, &startTime[timerCnt]);
    timerCnt++;
}

double stopTimer(char *s) {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    int idx = -1;
    
    for (int i = 0; i < timerCnt; i++) {
        if( strcmp(timerName[i], s) == 0 ) {
            idx = i; break;
        } 
    }

    double elapsed = (tp.tv_sec - startTime[idx].tv_sec) + (tp.tv_nsec - startTime[idx].tv_nsec) * 1e-9;
    totalTime[idx] += elapsed;

    return elapsed;
}

void printAllTimer() {
    for (int i = 0; i < timerCnt; i++) {
        fprintf(stderr, "[%s] %.2f sec\n", timerName[i], totalTime[i]);
    }
}

//////////////////////// CUSTOM TIMER DEFINITION END /////////////////////////

/* Error String from https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes */
const char *getErrorString(cl_int error){
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

void printErr(cl_int errNum, const char *str) {
    if (errNum == CL_SUCCESS) return;
    fprintf(stderr, "Error at %s: %s\n",str, getErrorString(errNum));
    exit(0);
}

void readEntireFile(const char *fname, char** str){
    FILE *f = fopen(fname, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if( fsize == 0 ){
        fprintf(stderr, "EMPTY FILE\n");
        exit(0);
    }

    *str = (char*) malloc(fsize + 1);
    fread(*str, 1, fsize, f);
    fclose(f);

    (*str)[fsize] = 0;
}


cl_device_id deviceIds[MAXDEV];
cl_uint numDevices = -1;
cl_int errNum;
cl_uint numPlatforms;
cl_platform_id platformId;
cl_program program[MAXDEV];
cl_command_queue commandQueue[MAXDEV];
cl_context context = 0;
int cnt = 0;
const int NETWORK_SIZE_IN_BYTES2 = 202875232;
cl_mem cl_network[MAXDEV];

cl_mem bir[2][MAXDEV], biz[2][MAXDEV], bin[2][MAXDEV], bhr[2][MAXDEV], bhz[2][MAXDEV], bhn[2][MAXDEV];
cl_mem bir[2][MAXDEV], biz[2][MAXDEV], bin[2][MAXDEV], bhr[2][MAXDEV], bhz[2][MAXDEV], bhn[2][MAXDEV];
cl_mem encoderw[MAXDEV], encoderb[MAXDEV], decoderw[MAXDEV], decoderb[MAXDEV];
cl_mem wirT[2][MAXDEV], wizT[2][MAXDEV], winT[2][MAXDEV], whrT[2][MAXDEV], whzT[2][MAXDEV], whnT[2][MAXDEV];
cl_mem wirT[2][MAXDEV], wizT[2][MAXDEV], winT[2][MAXDEV], whrT[2][MAXDEV], whzT[2][MAXDEV], whnT[2][MAXDEV];
cl_mem encoderwT[MAXDEV], encoderbT[MAXDEV], decoderwT[MAXDEV], decoderbT[MAXDEV];
cl_mem cl_in[MAXDEV], cl_x[MAXDEV], cl_h1[MAXDEV], cl_h2[MAXDEV], cl_o1[MAXDEV], cl_o2[MAXDEV], cl_out[MAXDEV], cl_hidden[MAXDEV];
cl_mem cl_r[MAXDEV], cl_z[MAXDEV], cl_n[MAXDEV], cl_h[MAXDEV];
cl_mem cl_zero[MAXDEV];

cl_mem batchIn[MAXDEV], batchX[MAXDEV], batchH1[MAXDEV], batchH2[MAXDEV], batchO1[MAXDEV], batchO2[MAXDEV], batchOut[MAXDEV];
cl_mem batchR[MAXDEV], batchZ[MAXDEV], batchN[MAXDEV], batchH[MAXDEV], batchT1[MAXDEV], batchT2[MAXDEV];
cl_mem batchOutput[MAXDEV];
cl_mem batchIter[MAXDEV], batchValid[MAXDEV];


cl_kernel kernel_fc[MAXDEV], kernel_matadd_sigmoid[MAXDEV], kernel_sample[MAXDEV], kernel_matadd_tanh[MAXDEV];
cl_kernel kernel_matproduct[MAXDEV], kernel_calcH[MAXDEV], kernel_linear[MAXDEV], kernel_transpose[MAXDEV];
cl_kernel kernel_sample2[MAXDEV], kernel_addone[MAXDEV];


static void bufferToHost(cl_mem mem, float *f, int sz, int dev) {
    errNum = clEnqueueReadBuffer(commandQueue[dev], mem, CL_FALSE, 0, sz * sizeof(float),
                                 f, 0, NULL, NULL);
    printErr(errNum, "clEnqueueReadBuffer");
}

static void hostToBuffer(cl_mem mem, float *f, int sz, int dev) {
    errNum = clEnqueueWriteBuffer(commandQueue[dev], mem, CL_FALSE, 0, sz * sizeof(float), 
                                  f, 0, NULL, NULL);
    printErr(errNum, "clEnqueueWriteBuffer");
}

static void copyBuffer(cl_mem dst, cl_mem src, int sz, int dev) {
    errNum = clEnqueueCopyBuffer(commandQueue[dev], src, dst, 0, 0,
                                 sz * sizeof(float), 0, NULL, NULL);
    printErr(errNum, "clEnqueueCopyBuffer1");
}


float zero[2048 * MAX_BATCH_SIZE];
int zeroInt[5] = {0, 0, 0, 0, 0}, oneInt[5] = {1, 1, 1, 1, 1};
// Scheduler..
typedef struct {
    float *input, *output;
    int len;
    int done;
} Job;

typedef struct {
    int occupancy;
    int jobIdx[MAX_BATCH_SIZE+10], iter[MAX_BATCH_SIZE+10], valid[MAX_BATCH_SIZE+10];
} GPU;


Job jobs[100050];
GPU gpu[MAXDEV];
int cursor[MAXDEV] = {0, 0, 0, 0};
int low[MAXDEV], high[MAXDEV];
int devcnt[MAXDEV], jobcnt[MAXDEV], joblist[100050];
static void cl_linear(cl_mem out, cl_mem A, cl_mem B, cl_mem bias,
                      int N, int M, int K, int dev)
{
    errNum = clSetKernelArg(kernel_linear[dev], 0, sizeof(cl_mem), &out);
    errNum = clSetKernelArg(kernel_linear[dev], 1, sizeof(cl_mem), &A);
    errNum = clSetKernelArg(kernel_linear[dev], 2, sizeof(cl_mem), &B);
    errNum = clSetKernelArg(kernel_linear[dev], 3, sizeof(cl_mem), &bias);
    errNum = clSetKernelArg(kernel_linear[dev], 4, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_linear[dev], 5, sizeof(cl_int), &M);
    errNum = clSetKernelArg(kernel_linear[dev], 6, sizeof(cl_int), &K);

    printErr(errNum, "clSetKernelArg linear");

    size_t globalWorkSize[2] = {M/4, N/4}; // 열 먼저 넣기
    size_t localWorkSize[2] = {16, 16};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_linear[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel linear");
    devcnt[dev]++;
}

static void cl_fc(cl_mem out, cl_mem A, cl_mem B, cl_mem bias, int N, int M, int K, int dev)
{
    errNum = clSetKernelArg(kernel_fc[dev], 0, sizeof(cl_mem), &out);
    errNum = clSetKernelArg(kernel_fc[dev], 1, sizeof(cl_mem), &A);
    errNum = clSetKernelArg(kernel_fc[dev], 2, sizeof(cl_mem), &B);
    errNum = clSetKernelArg(kernel_fc[dev], 3, sizeof(cl_mem), &bias);
    errNum = clSetKernelArg(kernel_fc[dev], 4, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_fc[dev], 5, sizeof(cl_int), &M);
    errNum = clSetKernelArg(kernel_fc[dev], 6, sizeof(cl_int), &K);

    printErr(errNum, "clSetKernelArg fc");

    size_t globalWorkSize[2] = {M, N}; // 열 먼저 넣기
    size_t localWorkSize[2] = {8, 8};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_fc[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel fc");
    devcnt[dev]++;
}

static void cl_matproduct(cl_mem A, cl_mem B, cl_mem C, int N, int M, int dev)
{
    errNum = clSetKernelArg(kernel_matproduct[dev], 0, sizeof(cl_mem), &A);
    errNum = clSetKernelArg(kernel_matproduct[dev], 1, sizeof(cl_mem), &B);
    errNum = clSetKernelArg(kernel_matproduct[dev], 2, sizeof(cl_mem), &C);
    errNum = clSetKernelArg(kernel_matproduct[dev], 3, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_matproduct[dev], 4, sizeof(cl_int), &M);

    printErr(errNum, "clSetKernelArg matproduct");

    size_t globalWorkSize[2] = {N, M};
    size_t localWorkSize[2] = {16, 16};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_matproduct[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel matproduct");
    devcnt[dev]++;
}

static void cl_matadd_sigmoid(cl_mem C, cl_mem A, cl_mem B, int N, int M, int dev)
{
    errNum = clSetKernelArg(kernel_matadd_sigmoid[dev], 0, sizeof(cl_mem), &C);
    errNum = clSetKernelArg(kernel_matadd_sigmoid[dev], 1, sizeof(cl_mem), &A);
    errNum = clSetKernelArg(kernel_matadd_sigmoid[dev], 2, sizeof(cl_mem), &B);
    errNum = clSetKernelArg(kernel_matadd_sigmoid[dev], 3, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_matadd_sigmoid[dev], 4, sizeof(cl_int), &M);

    printErr(errNum, "clSetKernelArg matadd sigmoid");

    size_t globalWorkSize[2] = {N, M};
    size_t localWorkSize[2] = {16, 16};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_matadd_sigmoid[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel matadd sigmoid");
    devcnt[dev]++;
}

static void cl_matadd_tanh(cl_mem C, cl_mem A, cl_mem B, int N, int M, int dev)
{
    errNum = clSetKernelArg(kernel_matadd_tanh[dev], 0, sizeof(cl_mem), &C);
    errNum = clSetKernelArg(kernel_matadd_tanh[dev], 1, sizeof(cl_mem), &A);
    errNum = clSetKernelArg(kernel_matadd_tanh[dev], 2, sizeof(cl_mem), &B);
    errNum = clSetKernelArg(kernel_matadd_tanh[dev], 3, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_matadd_tanh[dev], 4, sizeof(cl_int), &M);

    printErr(errNum, "clSetKernelArg matadd tanh");

    size_t globalWorkSize[2] = {N, M};
    size_t localWorkSize[2] = {16, 16};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_matadd_tanh[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel matadd tanh");
    devcnt[dev]++;
}

static void cl_calcH(cl_mem out, cl_mem z, cl_mem n, cl_mem h, int N, int M, int dev) 
{
    errNum = clSetKernelArg(kernel_calcH[dev], 0, sizeof(cl_mem), &out);
    errNum = clSetKernelArg(kernel_calcH[dev], 1, sizeof(cl_mem), &z);
    errNum = clSetKernelArg(kernel_calcH[dev], 2, sizeof(cl_mem), &n);
    errNum = clSetKernelArg(kernel_calcH[dev], 3, sizeof(cl_mem), &h);
    errNum = clSetKernelArg(kernel_calcH[dev], 4, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_calcH[dev], 5, sizeof(cl_int), &M);

    printErr(errNum, "clSetKernelArg calcH");

    size_t globalWorkSize[2] = {N, M};
    size_t localWorkSize[2] = {8, 8};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_calcH[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel calcH");

}

static void cl_transpose(cl_mem out, cl_mem in, int offset, int N, int M, int opt, int dev) 
{
    // opt 가 1 인 경우 transpose 하면서 저장한다
    startTimer("ktrans");
    errNum = clSetKernelArg(kernel_transpose[dev], 0, sizeof(cl_mem), &out);
    errNum = clSetKernelArg(kernel_transpose[dev], 1, sizeof(cl_mem), &in);
    errNum = clSetKernelArg(kernel_transpose[dev], 2, sizeof(cl_int), &offset);
    errNum = clSetKernelArg(kernel_transpose[dev], 3, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_transpose[dev], 4, sizeof(cl_int), &M);
    errNum = clSetKernelArg(kernel_transpose[dev], 5, sizeof(cl_int), &opt);

    printErr(errNum, "clSetKernelArg transpose");

    size_t globalWorkSize[2] = {N, M};
    size_t localWorkSize[2] = {8, 8};
    if ( N % 8 != 0 ) localWorkSize[0] = 1;
    if ( M % 8 != 0 ) localWorkSize[1] = 1;

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_transpose[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    stopTimer("ktrans");
    printErr(errNum, "clEnqueueNDRangeKernel transpose");
}

static void cl_sample(cl_mem A, int N, int M, int dev)
{
    errNum = clSetKernelArg(kernel_sample[dev], 0, sizeof(cl_mem), &A);
    errNum = clSetKernelArg(kernel_sample[dev], 1, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_sample[dev], 2, sizeof(cl_int), &M);

    printErr(errNum, "clSetKernelArg sample");

    size_t globalWorkSize[2] = {N, M};
    size_t localWorkSize[2] = {8, 8};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_sample[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel sample");
    devcnt[dev]++;
}

static void cl_sample2(cl_mem A, cl_mem output, cl_mem iter, cl_mem valid, int N, int M, int dev)
{
    errNum = clSetKernelArg(kernel_sample2[dev], 0, sizeof(cl_mem), &A);
    errNum = clSetKernelArg(kernel_sample2[dev], 1, sizeof(cl_mem), &output);
    errNum = clSetKernelArg(kernel_sample2[dev], 2, sizeof(cl_mem), &iter);
    errNum = clSetKernelArg(kernel_sample2[dev], 3, sizeof(cl_mem), &valid);
    errNum = clSetKernelArg(kernel_sample2[dev], 4, sizeof(cl_int), &N);
    errNum = clSetKernelArg(kernel_sample2[dev], 5, sizeof(cl_int), &M);

    printErr(errNum, "clSetKernelArg sample2");

    size_t globalWorkSize[2] = {N, M};
    size_t localWorkSize[2] = {8, 8};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_sample2[dev], 2, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel sample2");
    devcnt[dev]++;
}

static void cl_addone(cl_mem A, int N, int dev)
{
    errNum = clSetKernelArg(kernel_addone[dev], 0, sizeof(cl_mem), &A);

    printErr(errNum, "clSetKernelArg addone");

    size_t globalWorkSize[1] = {N};
    size_t localWorkSize[1] = {8};

    errNum = clEnqueueNDRangeKernel(commandQueue[dev], kernel_addone[dev], 1, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
    printErr(errNum, "clEnqueueNDRangeKernel addone");
    devcnt[dev]++;

}

void addJob(int dev, int jobIdx, int position){
    // fprintf(stderr, "addJob %d %d %d\n", dev, jobIdx, position);
    gpu[dev].jobIdx[position] = jobIdx;
    gpu[dev].iter[position] = 0;
    gpu[dev].valid[position] = 1;
    gpu[dev].occupancy++;
    jobcnt[dev]++;

    errNum = clEnqueueWriteBuffer(commandQueue[dev], batchIter[dev], CL_FALSE, position * sizeof(int), sizeof(int),
                        &zeroInt[dev], 0, NULL, NULL);
    printErr(errNum, "clEnqueueWriteBuffer11");
    clEnqueueWriteBuffer(commandQueue[dev], batchValid[dev], CL_FALSE, position * sizeof(int), sizeof(int),
                        &oneInt[dev], 0, NULL, NULL);
    printErr(errNum, "clEnqueueWriteBuffer12");
    clEnqueueWriteBuffer(commandQueue[dev], batchOut[dev], CL_FALSE, position * 88 * sizeof(float), 88 * sizeof(float),
                        jobs[jobIdx].input, 0, NULL, NULL);
    printErr(errNum, "clEnqueueWriteBuffer13");
    clEnqueueWriteBuffer(commandQueue[dev], batchO1[dev], CL_FALSE, position * 2048 * sizeof(float), 2048 * sizeof(float), 
                        zero, 0, NULL, NULL);
    printErr(errNum, "clEnqueueWriteBuffer14");
    clEnqueueWriteBuffer(commandQueue[dev], batchO2[dev], CL_FALSE, position * 2048 * sizeof(float), 2048 * sizeof(float), 
                        zero, 0, NULL, NULL);
    printErr(errNum, "clEnqueueWriteBuffer15");
    // TODO : change fillzero to kernel
}


void executeLayer(int dev) {
    // /////////////////////////////////// //
    // FC : in -> x                        //
    // /////////////////////////////////// //
    // 지난 단계 Out 이 여기서 x input 으로 쓰인다
    cl_fc(batchX[dev], batchOut[dev], encoderwT[dev], encoderb[dev], BATCH_SIZE, 2048, 88, dev);

    // /////////////////////////////////// //
    // GRU Block 0 : (x, h1) -> o1         //
    // /////////////////////////////////// //
    // 지난 단계 O1 이 여기서 H1 input 으로 쓰인다
    // R
    cl_linear(batchT1[dev], batchX[dev], wirT[0][dev], bir[0][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_linear(batchT2[dev], batchO1[dev], whrT[0][dev], bhr[0][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_matadd_sigmoid(batchR[dev], batchT1[dev], batchT2[dev], BATCH_SIZE, 2048, dev);
    // Z
    cl_linear(batchT1[dev], batchX[dev], wizT[0][dev], biz[0][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_linear(batchT2[dev], batchO1[dev], whzT[0][dev], bhz[0][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_matadd_sigmoid(batchZ[dev], batchT1[dev], batchT2[dev], BATCH_SIZE, 2048, dev);
    // N
    cl_linear(batchT1[dev], batchX[dev], winT[0][dev], bin[0][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_linear(batchT2[dev], batchO1[dev], whnT[0][dev], bhn[0][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_matproduct(batchT2[dev], batchR[dev], batchT2[dev], BATCH_SIZE, 2048, dev);
    cl_matadd_tanh(batchN[dev], batchT1[dev], batchT2[dev], BATCH_SIZE, 2048, dev);
    // H
    cl_calcH(batchO1[dev], batchZ[dev], batchN[dev], batchO1[dev], BATCH_SIZE, 2048, dev);

    // /////////////////////////////////// //
    // GRU Block 1 : (o1, h2) -> o2        //
    // /////////////////////////////////// //
    // 지난 단계 O2 가 여기서 H2 input 으로 쓰인다
    // R
    cl_linear(batchT1[dev], batchO1[dev], wirT[1][dev], bir[1][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_linear(batchT2[dev], batchO2[dev], whrT[1][dev], bhr[1][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_matadd_sigmoid(batchR[dev], batchT1[dev], batchT2[dev], BATCH_SIZE, 2048, dev);
    // Z
    cl_linear(batchT1[dev], batchO1[dev], wizT[1][dev], biz[1][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_linear(batchT2[dev], batchO2[dev], whzT[1][dev], bhz[1][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_matadd_sigmoid(batchZ[dev], batchT1[dev], batchT2[dev], BATCH_SIZE, 2048, dev);
    // N
    cl_linear(batchT1[dev], batchO1[dev], winT[1][dev], bin[1][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_linear(batchT2[dev], batchO2[dev], whnT[1][dev], bhn[1][dev], BATCH_SIZE, 2048, 2048, dev);
    cl_matproduct(batchT2[dev], batchR[dev], batchT2[dev], BATCH_SIZE, 2048, dev);
    cl_matadd_tanh(batchN[dev], batchT1[dev], batchT2[dev], BATCH_SIZE, 2048, dev);
    // H
    cl_calcH(batchO2[dev], batchZ[dev], batchN[dev], batchO2[dev], BATCH_SIZE, 2048, dev);

    // /////////////////////////////////// //
    // FC : o2 -> out                      //
    // /////////////////////////////////// //
    cl_fc(batchOut[dev], batchO2[dev], decoderwT[dev], decoderb[dev], BATCH_SIZE, 88, 2048, dev);

    // /////////////////////////////////// //
    // last sampling                       //
    // /////////////////////////////////// //
    // cl_sample(batchOut[dev], BATCH_SIZE, 88, dev);
    // clFinish(commandQueue[dev]);
    cl_sample2(batchOut[dev], batchOutput[dev], batchIter[dev], batchValid[dev], BATCH_SIZE, 88, dev);
    cl_addone(batchIter[dev], BATCH_SIZE, dev);
    
}

int bcnt = 0 ;
void popJobs(int dev) {
    // 다 끝난 작업들 Pop
    for (int i = 0; i < BATCH_SIZE; i++) {
        // fprintf(stderr, "i=%d\n", i);
        if( gpu[dev].valid[i] == 0 ) continue;
        int jobIdx = gpu[dev].jobIdx[i];

        // errNum = clEnqueueCopyBuffer(commandQueue[dev], batchOut[dev], batchOutput[dev], 
        //                     i * 88 * sizeof(float),   (i * 88 * 500 + gpu[dev].iter[i] * 88 ) * sizeof(float),
        //                     88 * sizeof(float), 0, NULL, NULL);
        // printErr(errNum, "clEnqueueCopyBuffer2");

        gpu[dev].iter[i]++;
        if( gpu[dev].iter[i] == jobs[jobIdx].len ){
            gpu[dev].valid[i] = 0;
            gpu[dev].occupancy--;

            errNum = clEnqueueWriteBuffer(commandQueue[dev], batchValid[dev], CL_FALSE, i * sizeof(int), sizeof(int),
                        &zeroInt[dev], 0, NULL, NULL);
            printErr(errNum, "clEnqueueWriteBuffer1");

            errNum = clEnqueueReadBuffer(commandQueue[dev], batchOutput[dev], CL_TRUE, (i * 88 * 500) * sizeof(float),
                                88 * jobs[jobIdx].len * sizeof(float), jobs[jobIdx].output, 0, NULL, NULL);
            printErr(errNum, "clEnqueueReadBuffer2");
        }
    }
} 

void music(int num_sample, int *sample_lens, float *network, float *inputs, float *outputs) {
    int ni = 88, nh = 2048, no = 88;

    if ( num_sample < BATCH_SIZE * 3 ) NDEV = 1;
    else if( num_sample < BATCH_SIZE * 7 ) NDEV = 2;
    else NDEV = 4;

    fprintf(stderr, "%d Data\n", num_sample);
    fprintf(stderr, "BATCH_SIZE = %d, %d GPUs\n", BATCH_SIZE, NDEV);

    for (int id = 0; id < NDEV; id++) {
        cl_network[id] = clCreateBuffer(context, CL_MEM_READ_ONLY, NETWORK_SIZE_IN_BYTES2, NULL, &errNum);
        clEnqueueWriteBuffer(commandQueue[id], cl_network[id], CL_FALSE, 0, NETWORK_SIZE_IN_BYTES2, network, 0, NULL, NULL);
        printErr(errNum, "clEnqueueMapBuffer");
    }

    for (int id = 0; id < NDEV; id++) {
        cl_transpose(encoderwT[id], cl_network[id], 0, 2048, 88, 1, id);
        cl_transpose(encoderb[id], cl_network[id], 2048 * 88, 1, 2048, 0, id);
        cl_transpose(bir[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 6 + 2048 * 1, 1, 2048, 0, id);
        cl_transpose(biz[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 6 + 2048 * 2, 1, 2048, 0, id);
        cl_transpose(bin[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 6 + 2048 * 3, 1, 2048, 0, id);
        cl_transpose(bhr[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 6 + 2048 * 4, 1, 2048, 0, id);
        cl_transpose(bhz[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 6 + 2048 * 5, 1, 2048, 0, id);
        cl_transpose(bhn[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 6 + 2048 * 6, 1, 2048, 0, id);
        cl_transpose(wirT[0][id], cl_network[id], 2048 * 88 + 2048, 2048, 2048, 1, id);
        cl_transpose(wizT[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 1 + 2048, 2048, 2048, 1, id);
        cl_transpose(winT[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 2 + 2048, 2048, 2048, 1, id);
        cl_transpose(whrT[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 3 + 2048, 2048, 2048, 1, id);
        cl_transpose(whzT[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 4 + 2048, 2048, 2048, 1, id);
        cl_transpose(whnT[0][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 5 + 2048, 2048, 2048, 1, id);
        cl_transpose(bir[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 12 + 2048 * 7, 1, 2048, 0, id);
        cl_transpose(biz[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 12 + 2048 * 8, 1, 2048, 0, id);
        cl_transpose(bin[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 12 + 2048 * 9, 1, 2048, 0, id);
        cl_transpose(bhr[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 12 + 2048 * 10, 1, 2048, 0, id);
        cl_transpose(bhz[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 12 + 2048 * 11, 1, 2048, 0, id);
        cl_transpose(bhn[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 12 + 2048 * 12, 1, 2048, 0, id);
        cl_transpose(wirT[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 6 + 2048 * 7, 2048, 2048, 1, id);
        cl_transpose(wizT[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 7 + 2048 * 7, 2048, 2048, 1, id);
        cl_transpose(winT[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 8 + 2048 * 7, 2048, 2048, 1, id);
        cl_transpose(whrT[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 9 + 2048 * 7, 2048, 2048, 1, id);
        cl_transpose(whzT[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 10 + 2048 * 7, 2048, 2048, 1, id);
        cl_transpose(whnT[1][id], cl_network[id], 2048 * 88 + 2048 * 2048 * 11 + 2048 * 7, 2048, 2048, 1, id);
        cl_transpose(decoderwT[id], cl_network[id], 2048 * 88 * 1 + 2048 * 2048 * 12 + 2048 * 13, 88, 2048, 1, id);
        cl_transpose(decoderb[id], cl_network[id],  2048 * 88 * 2 + 2048 * 2048 * 12 + 2048 * 13, 1, 88, 1, id);
    }

    /*
    cl_mem batchIn[MAXDEV], batchX[MAXDEV], batchH1[MAXDEV], batchH2[MAXDEV], batchO1[MAXDEV], batchO2[MAXDEV], batchOut[MAXDEV];
    cl_mem batchR[MAXDEV], batchZ[MAXDEV], batchN[MAXDEV], batchH[MAXDEV];
    */
    for(int id = 0; id < NDEV; id++) {
        batchIn[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * MAX_BATCH_SIZE * 88 * sizeof(float), NULL, &errNum);
        batchOutput[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * MAX_BATCH_SIZE * 500 * 88 * sizeof(float), NULL, &errNum);
        batchOut[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * MAX_BATCH_SIZE * 88 * sizeof(float), NULL, &errNum);
        batchX[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchH1[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchH2[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchO1[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchO2[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchR[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchZ[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchN[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchH[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchT1[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
        batchT2[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_BATCH_SIZE * 2048 * sizeof(float), NULL, &errNum);
    }

    ///////////////////////// PREPARE END //////////////////////////////

    float *tin = inputs, *tout = outputs;
    for (int n = 0; n < num_sample; n++) {
        jobs[n].len = sample_lens[n];
        jobs[n].input = tin;
        jobs[n].output = tout;
        tin += ni;
        tout += sample_lens[n] * no;
    }

    // 작업들을 길이순으로 내림차순 정렬. 
    // 일단 O(ND) 이고, 오버헤드가 크다 싶으면 counting sort O(N+D) 로 변경
    int p = 0;
    for (int d = 500; d >= 1; d--) {
        for (int n = 0; n < num_sample; n++) {
            if( jobs[n].len == d ) joblist[p++] = n;
        }
    }

    int *arr = (int*) malloc( sizeof(int) * 100100 );
    p = 0;
    for (int dev = 0; dev < NDEV; dev++) {
        for (int n = dev; n < num_sample; n += NDEV) {
            arr[p++] = joblist[n];
        }
    }

    for (int n = 0; n < num_sample; n++) joblist[n] = arr[n];

    // 이제 GPU별로 N/4 씩 적당히 나눠 가지자
    for (int i = 0; i < NDEV; i++) {
        low[i] = (i) * num_sample / NDEV;
        high[i] = (i+1) *num_sample / NDEV;
    }
    int done[MAXDEV] = {0, 0, 0, 0};
    
    // low[0] = 0; high[0] = 0;
    // low[1] = 0; high[0] = 3000;
    // low[2] = 3000; high[0] = 6000;
    // low[3] = 6000; high[0] = num_sample;
    // done[0] = 1;

    for (int i = 0; i < NDEV; i++) cursor[i] = low[i];

    // 아직 작업이 다 안끝난 GPU가 있을 때까지
    int iter2 = 0;
    while( 1 ) {
        int flag = 1;
        for (int i = 0; i < NDEV; i++) flag &= done[i];
        if( flag == 1 ) break;

        // insert job phase
        for (int dev = 0; dev < NDEV; dev++) {
            if( done[dev] ) continue;

            // GPU에 작업을 추가가능한만큼 추가
            while( gpu[dev].occupancy < BATCH_SIZE && cursor[dev] < high[dev] ) {
                // GPU에서 빈칸 찾기
                int position = -1;
                for (int i = 0; i < BATCH_SIZE; i++) {
                    if( gpu[dev].valid[i] == 0 ){
                        position = i;
                        break;
                    }
                }

                // fprintf(stderr, "(%d,%d)", dev, position); fflush(stderr);
                addJob(dev, joblist[cursor[dev]], position);
                cursor[dev]++;
            }
            executeLayer(dev);
        }


        // pop & cleanup
        for (int dev = 0; dev < NDEV; dev++) {
            if( done[dev] ) continue;
            popJobs(dev);
        
            // 이 GPU 한테 할당된 작업은 다 끝났다.
            if( gpu[dev].occupancy == 0 && cursor[dev] == high[dev] ){
                done[dev] = 1;
            }
        }

    }

    fprintf(stderr, "jobs done\n"); fflush(stderr);
    startTimer("wait");
    int sum[4] = {0, 0, 0, 0};
    // startTimer("wait");
    for (int i = 0; i < NDEV; i++) {
        for (int j = low[i]; j < high[i]; j++) sum[i] += jobs[joblist[j]].len;
        clFinish(commandQueue[i]);
        fprintf(stderr, "[GPU %d] enqueue count=%d, total length=%d, done job=%d\n", i, devcnt[i], sum[i], jobcnt[i]);
    }
    stopTimer("wait");

    printAllTimer();
}

void music_init() {
    // platform 정하기 (어차피 하나니까 첫째 플랫폼 사용)
    errNum = clGetPlatformIDs(1, &platformId, &numPlatforms);
    printErr(errNum, "clGetPlatformIDs");

    // openCL context 생성
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformId,
         0
    };

    context = clCreateContextFromType(contextProperties,
                                      CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    printErr(errNum, "clCreateContextFromType");

    // Device 개수 받아오기
    errNum = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    printErr(errNum, "clGetDeviceIDs 1");
    fprintf(stderr, "%d devices detected\n", numDevices);
    errNum = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numDevices, deviceIds, NULL);
    printErr(errNum, "clGetDeviceIDs 2");

    // Device 별로 Command Queue 만들기
    for (int id = 0; id < MAXDEV; id++) {
        commandQueue[id] = clCreateCommandQueue(context, deviceIds[id], CL_QUEUE_PROFILING_ENABLE, &errNum);
        printErr(errNum, "clCreateCommandQueue");
    }

    // 커널 코드 읽고 Program 만들기
    char *kernelCode;
    readEntireFile("kernel.cl", &kernelCode);
    for (int id = 0; id < MAXDEV; id++) {
        program[id] = clCreateProgramWithSource(context, 1, (const char**) &kernelCode, NULL, &errNum);
    }
    printErr(errNum, "clCreateProgramWithSource");

    // 위에서 만든 Program 을 device 별로 빌드하기
    for (int id = 0; id < MAXDEV; id++) {
        errNum = clBuildProgram(program[id], 1, &deviceIds[id], NULL, NULL, NULL);
        if( errNum != CL_SUCCESS ) {
            char buildLog[16384];
            clGetProgramBuildInfo(program[id], deviceIds[id], CL_PROGRAM_BUILD_LOG,
                                sizeof(buildLog), buildLog, NULL);
            fprintf(stderr, "Error while building Kernel\n");
            fprintf(stderr, "%s\n", buildLog);
            clReleaseProgram(program[id]);
            return;
        }
    }

    for (int id = 0; id < MAXDEV; id++) {
        kernel_linear[id] = clCreateKernel(program[id], "linear", &errNum);
        kernel_fc[id] = clCreateKernel(program[id], "fc", &errNum);
        kernel_matadd_sigmoid[id] = clCreateKernel(program[id], "mat_add_sigmoid", &errNum);
        kernel_matadd_tanh[id] = clCreateKernel(program[id], "mat_add_tanh", &errNum);
        kernel_sample[id] = clCreateKernel(program[id], "sample", &errNum);
        kernel_sample2[id] = clCreateKernel(program[id], "sample2", &errNum);
        kernel_matproduct[id] = clCreateKernel(program[id], "matproduct", &errNum);
        kernel_calcH[id] = clCreateKernel(program[id], "calcH", &errNum);
        kernel_transpose[id] = clCreateKernel(program[id], "transpose", &errNum);
        kernel_addone[id] = clCreateKernel(program[id], "addone", &errNum);
    }


    for (int id = 0; id < MAXDEV; id++) {
        batchIter[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * MAX_BATCH_SIZE * sizeof(int), NULL, &errNum);
        printErr(errNum, "buffer");
        batchValid[id] = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * MAX_BATCH_SIZE * sizeof(int), NULL, &errNum);
        printErr(errNum, "buffer");

        cl_network[id] = clCreateBuffer(context, CL_MEM_READ_ONLY, NETWORK_SIZE_IN_BYTES2, NULL, &errNum);
        printErr(errNum, "buffer");
        encoderwT[id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 88 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        encoderb[id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        
        bir[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        biz[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        bin[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        bhr[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        bhz[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        bhn[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        
        wirT[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        wizT[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        winT[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        whrT[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        whzT[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        whnT[0][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        
        bir[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        biz[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        bin[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        bhr[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        bhz[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        bhn[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        
        wirT[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        wizT[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        winT[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        whrT[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        whzT[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        whnT[1][id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 2048 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");

        decoderb[id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 88 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
        decoderwT[id] = clCreateBuffer(context, CL_MEM_READ_ONLY, 2048 * 88 * sizeof(float), NULL, &errNum);
        printErr(errNum, "buffer");
    }
}


