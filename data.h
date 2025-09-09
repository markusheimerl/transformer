#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

void generate_data(float** X, float** y, int seq_len, int num_samples, int d_model, float range_min, float range_max);
void save_data(float* X, float* y, int seq_len, int num_samples, int d_model, const char* filename);

#endif