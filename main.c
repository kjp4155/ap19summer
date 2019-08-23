#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "timer.h"
#include "music.h"

const int NETWORK_SIZE_IN_BYTES = 202875232;

// read network binary
float* read_network(char *fn) {
    FILE *fin = fopen(fn, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open '%s'.\n", fn);
        exit(EXIT_FAILURE);
    }
    printf("Reading '%s'...", fn); fflush(stdout);

    fseek(fin, 0, SEEK_END);
    long sz = ftell(fin);
    if (sz != NETWORK_SIZE_IN_BYTES) {
        fprintf(stderr, "Expected %dB, but actual size is %ldB.\n", NETWORK_SIZE_IN_BYTES, sz);
        exit(EXIT_FAILURE);
    }
    fseek(fin, 0, SEEK_SET);

    float *network = (float*)malloc(NETWORK_SIZE_IN_BYTES);
    fread(network, 1, NETWORK_SIZE_IN_BYTES, fin);
    fclose(fin);
    printf(" done!\n");
    return network;
}

// read input file
float *read_inputs(char *fn, int *num_sample, int **sample_lens) {
    FILE *fin = fopen(fn, "r");
    if (!fin) {
        fprintf(stderr, "Failed to open '%s'.\n", fn);
        exit(EXIT_FAILURE);
    }
    printf("Reading '%s'...", fn); fflush(stdout);
    
    fscanf(fin, "%d", num_sample);
    float *inputs = (float*)malloc(*num_sample * 88 * sizeof(float));
    *sample_lens = (int*)malloc(*num_sample * sizeof(int));
    for (int n = 0; n < *num_sample; ++n) {
        fscanf(fin, "%d", (*sample_lens) + n);

        char tmp[128];
        fscanf(fin, "%s", tmp);
        for (int i = 0; i < 88; ++i) {
            inputs[n * 88 + i] = (float)(tmp[i] - '0');
        }
    }
    fclose(fin);
    printf(" done!\n");
    return inputs;
}

// write output file
void write_outputs(char *fn, int num_sample, int *sample_lens, float *outputs) {
    printf("Writing '%s'...", fn); fflush(stdout);
    FILE *fout = fopen(fn, "w");
    fprintf(fout, "%d\n", num_sample);

    int offset = 0;
    for (int i = 0; i < num_sample; ++i) {
        fprintf(fout, "%d\n", sample_lens[i]);

        for (int j = 0; j < sample_lens[i]; ++j) {
            for (int k = 0; k < 88; ++k) {
                fprintf(fout, "%d", (int)outputs[offset + k]);
            }
            fprintf(fout, "\n");
            offset += 88;
        }
    }
    fclose(fout);
    printf(" done!\n");
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input data> <output data>\n", argv[0]);
        fprintf(stderr, " e.g., %s input1.in output1.in\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int num_sample;
    int *sample_lens = NULL;
    char *network_path = "./network.bin";
    float *network = read_network(network_path);
    float *inputs = read_inputs(argv[1], &num_sample, &sample_lens);
    
    int num_outputs = 0;
    for (int i = 0; i < num_sample; i++) {
        num_outputs += sample_lens[i] * 88;
    }

    float *outputs = (float*)malloc(num_outputs * sizeof(float));

    // initialize; does not count into elapsed time
    printf("Initializing..."); fflush(stdout);
    music_init();
    printf(" done!\n");

    // main calculation
    printf("Calculating..."); fflush(stdout);
    timer_start(0);
    music(num_sample, sample_lens, network, inputs, outputs);
    double elapsed = timer_stop(0);
    printf(" done!\n");
    printf("Elapsed time : %.6f sec\n", elapsed);

    write_outputs(argv[2], num_sample, sample_lens, outputs);

    // free resources
    free(sample_lens);
    free(network);
    free(inputs);
    free(outputs);

    return 0;
}
