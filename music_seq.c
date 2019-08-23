// TIMESTAMP: 1
#include "music.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static float _exp(float x) {
    return expf(x);
}

static float _sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

static float _tanh(float x) {
    return tanhf(x);
}

static void mat_init(float *A, int ROW, int COL) {
    int i;
    for (i = 0; i < ROW * COL; i++) {
        A[i] = 0.0f;
    }
}

static void mat_add(float *A, float *B, float *C,
                    int ROW, int COL) {
    int i;
    for (i = 0; i < ROW * COL; i++) {
        C[i] = A[i] + B[i];
    }
}

static void mat_mul(float *A, float *B, float *C,
                    int ROW_A, int COL_A, int COL_B) {
  int i, j, k;
  for (i = 0; i < ROW_A; i++) {
    for (j = 0; j < COL_B; j++) {
      C[i * COL_B + j] = 0.0f;
      for (k = 0; k < COL_A; k++) {
        C[i * COL_B + j] += A[i * COL_A + k] * B[k * COL_B + j];
      }
    }
  }
}

/*
 * linear transformation: y = Ax + b
 * in : (H_IN)
 * out : (H_OUT)
 * weight : (H_OUT, H_IN)
 * bias : (H_OUT)
 */
static void fc_layer(float *in, float *out,
                    float *weight, float *bias,
                    int H_IN, int H_OUT) {
    mat_mul(weight, in, out, H_OUT, H_IN, 1);
    for (int h_out = 0; h_out < H_OUT; ++h_out) {
        out[h_out] += bias[h_out];
    }
}

static void gru_layer(float *in, float *hidden, float *out, 
                    float *w_ir, float *w_iz, float *w_in,
                    float *w_hr, float *w_hz, float *w_hn,
                    float *b_ir, float *b_iz, float *b_in,
                    float *b_hr, float *b_hz, float *b_hn,
                    float *r, float *z, float *n, float *h, 
                    float *t1, float *t2,
                    int N, int H) {
    int i;

    // initailize r, z, n, h
    mat_init(r, H, 1);
    mat_init(z, H, 1);
    mat_init(n, H, 1);
    mat_init(h, H, 1);

    // compute r
    mat_mul(w_ir, in, t1, H, N, 1);
    mat_mul(w_hr, hidden, t2, H, H, 1);
    mat_add(r, t1, r, H, 1);
    mat_add(r, b_ir, r, H, 1);
    mat_add(r, t2, r, H, 1);
    mat_add(r, b_hr, r, H, 1);
    for (i = 0; i < H; i++) {
        r[i] = _sigmoid(r[i]);
    }

    // compute z
    mat_mul(w_iz, in, t1, H, N, 1);
    mat_mul(w_hz, hidden, t2, H, H, 1);
    mat_add(z, t1, z, H, 1);
    mat_add(z, b_iz, z, H, 1);
    mat_add(z, t2, z, H, 1);
    mat_add(z, b_hz, z, H, 1);
    for (i = 0; i < H; i++) {
        z[i] = _sigmoid(z[i]);
    }
    
    // compute n
    mat_mul(w_in, in, t1, H, N, 1);
    mat_mul(w_hn, hidden, t2, H, H, 1);
    mat_add(t2, b_hn, t2, H, 1);
    for (i = 0; i < H; i++) {
        t2[i] = r[i] * t2[i];
    }
    mat_add(n, t1, n, H, 1);
    mat_add(n, b_in, n, H, 1);
    mat_add(n, t2, n, H, 1);
    for (i = 0; i < H; i++) {
        n[i] = _tanh(n[i]);
    }

    // compute h
    for (i = 0; i < H; i++) {
        t1[i] = (1 - z[i]) * n[i];
        t2[i] = z[i] * hidden[i];
        h[i] = t1[i] + t2[i];
    }

    // compute out
    for (i = 0; i < H; i++) {
        out[i] = h[i];
    }
}

float sample_seq(float x) {
    float temperature = 0.7f, threshold = 0.5f;
    float v1 = x / temperature, v2 = (1 - x) / temperature;
    float p = _exp(v2) / (_exp(v1) + _exp(v2));
    // if ( fabs(p - threshold) < 1e-5 ) fprintf(stderr, "Sequential p %.20f\n", p);
    return (p > threshold) ? 1 : 0;
}

void music_init() {
    // nothing to do
}

void music(int num_sample, int *sample_lens, float *network, float *inputs, float *outputs) {
    int ni = 88, nh = 2048, no = 88;

    // Encoder layer
    float *encoder_w = network; network += ni * nh;
    float *encoder_b = network; network += nh;

    // GRU layer0
    float *w_ir0 = network; network += nh * nh;
    float *w_iz0 = network; network += nh * nh;
    float *w_in0 = network; network += nh * nh;
    float *w_hr0 = network; network += nh * nh;
    float *w_hz0 = network; network += nh * nh;
    float *w_hn0 = network; network += nh * nh;
    float *b_ir0 = network; network += nh;
    float *b_iz0 = network; network += nh;
    float *b_in0 = network; network += nh;
    float *b_hr0 = network; network += nh;
    float *b_hz0 = network; network += nh;
    float *b_hn0 = network; network += nh;

    // GRU layer1
    float *w_ir1 = network; network += nh * nh;
    float *w_iz1 = network; network += nh * nh;
    float *w_in1 = network; network += nh * nh;
    float *w_hr1 = network; network += nh * nh;
    float *w_hz1 = network; network += nh * nh;
    float *w_hn1 = network; network += nh * nh;
    float *b_ir1 = network; network += nh;
    float *b_iz1 = network; network += nh;
    float *b_in1 = network; network += nh;
    float *b_hr1 = network; network += nh;
    float *b_hz1 = network; network += nh;
    float *b_hn1 = network; network += nh;

    // Decoder layer
    float *decoder_w = network; network += nh * no;
    float *decoder_b = network; network += no;

    // intermediate buffer for hidden state
    float *in = (float*)malloc(ni * sizeof(float));
    float *x = (float*)malloc(nh * sizeof(float));
    float *h1 = (float*)malloc(nh * sizeof(float));
    float *h2 = (float*)malloc(nh * sizeof(float));
    float *o1 = (float*)malloc(nh * sizeof(float));
    float *o2 = (float*)malloc(nh * sizeof(float));
    float *out = (float*)malloc(no * sizeof(float));

    // intermediate buffer for GRU layer
    float *r = (float*)malloc(nh * sizeof(float));
    float *z = (float*)malloc(nh * sizeof(float));
    float *n = (float*)malloc(nh * sizeof(float));
    float *h = (float*)malloc(nh * sizeof(float));
    float *t1 = (float*)malloc(nh * sizeof(float));
    float *t2 = (float*)malloc(nh * sizeof(float));

    // run network for each sample
    float *input = inputs;
    float *output = outputs;
    for (int s = 0; s < num_sample; ++s) {
        int len = sample_lens[s];
        for (int i = 0; i < ni; i++) {
            in[i] = input[i];
        }
        for (int i = 0; i < nh; i++) {
            h1[i] = 0.0f;
            h2[i] = 0.0f;
        }        

        for (int l = 0; l < len; ++l) {
            // FC : in -> x
            fc_layer(in, x, encoder_w, encoder_b, ni, nh);

            // GRU : (x, h1) -> o1
            gru_layer(x, h1, o1,
                w_ir0, w_iz0, w_in0,
                w_hr0, w_hz0, w_hn0,
                b_ir0, b_iz0, b_in0,
                b_hr0, b_hz0, b_hn0,
                r, z, n, h, t1, t2, nh, nh);

            // GRU : (o1, h2) -> o2
            gru_layer(o1, h2, o2,
                w_ir1, w_iz1, w_in1,
                w_hr1, w_hz1, w_hn1,
                b_ir1, b_iz1, b_in1,
                b_hr1, b_hz1, b_hn1,
                r, z, n, h, t1, t2, nh, nh);

            // FC : o2 -> out
            fc_layer(o2, out, decoder_w, decoder_b, nh, no);

            // prepare for next state
            for (int i = 0; i < no; i++) {
                out[i] = sample_seq(out[i]);
                in[i] = out[i];
            }
            for (int j = 0; j < nh; j++) {
                h1[j] = o1[j];
                h2[j] = o2[j];
            }

            for (int k = 0; k < no; k++) {
                output[k] = out[k];
            }
            output += no;
        }

        input += ni;
    }

    // free resources
    free(in);
    free(x);
    free(h1);
    free(h2);
    free(o1);
    free(o2);
    free(out);
    free(r);
    free(z);
    free(n);
    free(h);
    free(t1);
    free(t2);
}
