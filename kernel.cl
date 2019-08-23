
__kernel void fc(__global float *C, __global float *A,
                 __global float *B, __global float *bias,
                 int N, int M, int K) {
    int i = get_global_id(1);
    int j = get_global_id(0);
    int k, l;
    if (i >= N || j >= M) return;

    __local float localA[8][8], localB[8][8];

    float sum = bias[j];
    for (k = 0; k < K; k+=8) {
        localA[get_local_id(1)][get_local_id(0)] = A[i * K + k + get_local_id(0)];
        localB[get_local_id(1)][get_local_id(0)] = B[(k + get_local_id(1)) * M + j];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (l = 0; l < 8; l++) {
            sum += localA[get_local_id(1)][l] * localB[l][get_local_id(0)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[i * M + j] = sum;
}

/*
__kernel void linear(__global float *C, __global float *A,
                     __global float *B, __global float *bias,
                     int N, int M, int K)
{
    int i = get_global_id(1), j = get_global_id(0);
    if( i >= N || j >= M ) return;

    __local float localA[16][16], localB[16][16];

    float sum = bias[j];
    for (int k = 0; k < K; k+=16) {
        localA[get_local_id(1)][get_local_id(0)] = A[i * K + k + get_local_id(0)];
        localB[get_local_id(1)][get_local_id(0)] = B[(k + get_local_id(1)) * M + j];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int l = 0; l < 16; l++) {
            sum += localA[get_local_id(1)][l] * localB[l][get_local_id(0)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[i * M + j] = sum;
}
*/

__kernel void linear(__global float4 *C, __global float4 *A,
                     __global float4 *B, __global float4 *bias,
                     int N, int M, int K)
{
    int i = get_global_id(1) * 4, j = get_global_id(0);
    int li = get_local_id(1) * 4, lj = get_local_id(0);
    int N4 = N / 4, M4 = M / 4, K4 = K / 4;
    int k, l;
    if( i >= N || j * 4 >= M ) return;

    __local float4 localA[64][16], localB[64][16];
    float4 ans[4] = {0.0, 0.0, 0.0, 0.0};

    for (k = 0; k < K; k+= 64) {
        localA[li+0][lj] = A[(i+0)*K4 + k/4 + lj];
        localA[li+1][lj] = A[(i+1)*K4 + k/4 + lj];
        localA[li+2][lj] = A[(i+2)*K4 + k/4 + lj];
        localA[li+3][lj] = A[(i+3)*K4 + k/4 + lj];
        localB[li+0][lj] = B[(k+li+0)*M4 + j];
        localB[li+1][lj] = B[(k+li+1)*M4 + j];
        localB[li+2][lj] = B[(k+li+2)*M4 + j];
        localB[li+3][lj] = B[(k+li+3)*M4 + j];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (l = 0; l < 16; l++) {
            ans[0] += localA[li+0][l].x * localB[l*4+0][lj];
            ans[0] += localA[li+0][l].y * localB[l*4+1][lj];
            ans[0] += localA[li+0][l].z * localB[l*4+2][lj];
            ans[0] += localA[li+0][l].w * localB[l*4+3][lj];
            
            ans[1] += localA[li+1][l].x * localB[l*4+0][lj];
            ans[1] += localA[li+1][l].y * localB[l*4+1][lj];
            ans[1] += localA[li+1][l].z * localB[l*4+2][lj];
            ans[1] += localA[li+1][l].w * localB[l*4+3][lj];
            
            ans[2] += localA[li+2][l].x * localB[l*4+0][lj];
            ans[2] += localA[li+2][l].y * localB[l*4+1][lj];
            ans[2] += localA[li+2][l].z * localB[l*4+2][lj];
            ans[2] += localA[li+2][l].w * localB[l*4+3][lj];
            
            ans[3] += localA[li+3][l].x * localB[l*4+0][lj];
            ans[3] += localA[li+3][l].y * localB[l*4+1][lj];
            ans[3] += localA[li+3][l].z * localB[l*4+2][lj];
            ans[3] += localA[li+3][l].w * localB[l*4+3][lj];
            
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[(i+0)*M4 + j] = ans[0] + bias[j];
    C[(i+1)*M4 + j] = ans[1] + bias[j];
    C[(i+2)*M4 + j] = ans[2] + bias[j];
    C[(i+3)*M4 + j] = ans[3] + bias[j];
}

__kernel void mat_add_sigmoid(__global float *C, __global float *A, __global float *B,
                      int N, int M)
{
    int i = get_global_id(0), j = get_global_id(1);
    float t = A[i * M + j] + B[i * M + j];
    t = 1.0/(1.0+exp(-t));
    C[i * M + j] = t;
}

__kernel void mat_add_tanh(__global float *C, __global float *A, __global float *B,
                      int N, int M)
{
    int i = get_global_id(0), j = get_global_id(1);
    float t = A[i * M + j] + B[i * M + j];
    t = tanh(t);
    C[i * M + j] = t;
}

__kernel void sigmoid(__global float *inout, int N, int M)
{
    int i = get_global_id(0), j = get_global_id(1);
    inout[i * M + j] = 1.0/(1.0+exp(-inout[i * M + j]));
}

float sample_seq(float x) {
    float temperature = 0.7f, threshold = 0.5f;
    float v1 = x / temperature, v2 = (1 - x) / temperature;
    float p = exp(v2) / (exp(v1) + exp(v2));
    // if ( fabs(p - threshold) < 1e-5 ) printf("%.20f\n", p);
    // printf("%.1f \n",p);
    return (p > threshold) ? 1 : 0;
}

__kernel void sample(__global float *inout, int N, int M) {
    int i = get_global_id(0), j = get_global_id(1);
    inout[i * M + j] = sample_seq(inout[i * M + j]);
}

__kernel void sample2(__global float *inout, __global float *output, 
                      __global int *iter, __global int *valid, int N, int M) {
    int i = get_global_id(0), j = get_global_id(1);

    float tmp = sample_seq(inout[i * M + j]);
    inout[i * M + j] = tmp;

    if( valid[i] == 1 ){
        int tt = iter[i];
       
        output[ i * 88 * 500 + tt * 88 + j ] = tmp;
    }
}

__kernel void addone(__global int *inout){
    int i = get_global_id(0);
    inout[i] = inout[i] + 1;
}

__kernel void cl_tanh(__global float *inout, int N, int M) {
    int i = get_global_id(0), j = get_global_id(1);
    inout[i * M + j] = tanh(inout[i * M + j]);
}

__kernel void matproduct(__global float *A, __global float *B, __global float *C,
                         int N, int M)
{
    int i = get_global_id(0), j = get_global_id(1);
    C[i * M + j] = A[i * M + j] * B[i * M + j];
}

__kernel void calcH(__global float *out, __global float *z, __global float *n, __global float *h,
                    int N, int M)
{
    int i = get_global_id(0), j = get_global_id(1);
    float sum1 = 1.0 - z[i * M + j];
    float sum2 = z[i * M + j];
    sum1 *= n[i * M + j];
    sum2 *= h[i * M + j];
    out[i * M + j] = sum1 + sum2;
}

__kernel void setzero(__global float *inout)
{
    int i = get_global_id(0);
    inout[i] = 0.0;
}

__kernel void transpose(__global float *out, __global float *in, int offset, int N, int M, int trans) {
    int i = get_global_id(0), j = get_global_id(1);
    if ( trans == 1 ) {
        out[j * N + i] = in[i * M + j + offset];
    }
    else {
        out[i * M + j] = in[i * M + j + offset];
    }
}

__kernel void mat_add(__global float *a, __global float *b, __global float *c, int N, int M)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    a[i*M+j] = b[i*M+j] + c[i*M+j];
}
