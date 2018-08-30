extern "C"
__global__ void rowfilter(
    float* dest, const float* src, const float *w, int N, int C, int H, int W,
    int Win, int Mlow, int Mhigh, int stride) {
    /* dest - output array. should be same shape as input
       src - input array
       w - input kernel. Should be a 1d array
       N, C, H, W - input tensor sizes
       Mlow - idx of most negative filter tap
       Mhigh - idx of most positive filter tap
       rev - used for calculating gradients - need to do correlation, and
        some funny things with the filter.
    */
    for (int i = stride * (blockIdx.x * blockDim.x + threadIdx.x);
         i < N*C*H*W; i += stride * (blockDim.x * gridDim.x)) {
        const int n = i / C / H / W;
        const int c = (i / H / W) % C;
        const int y = (i / W) % H;
        const int x = i % W;
        float value = 0;
        // Use convolution formula: y[n] = sum h[k]*x[n-k]
#pragma unroll
        for (int k = Mlow; k <= Mhigh; k++) {
            int x_in = x - k;

            // handle padding - the above complicated equation
            // simply makes sure that the correct index input is used
            // for symmetric padding. I.e. it should result in x_in going from:
            // -3 -2 -1 | 0  1  2  3  4  5  6 | 7  8  9
            //  to:
            //  2  1  0 | 0  1  2  3  4  5  6 | 6  5  4
            // It also allows padding by more than the input length.
            // The group variable will be:
            // 1  1  1  | 0  0  0  0  0  0  0 | 1  1  1  1  1  1 | 0  0  0 ...
            const int group = x_in >= 0 ? ((x_in / Win) % 2)
                                        : 1-(((-x_in-1)/Win) % 2);

            // This does modulo operation but allowing for negative numbers
            // i.e. we want  -2 % 5 = 3. In python this works but in C we it
            // gives -2.
            // On top of reflecting the signal, we also need to reflect the
            // filter around the boundary (unlike with the forward pass).
            const int res = (x_in % Win + Win) % Win;
            x_in = (group == 1) ? (Win-1) - res : res;

            const int offset = n*C*H*Win + c*H*Win + y*Win + x_in;
            value +=  w[k-Mlow] * src[offset];
        }
        dest[i/stride] = value;
    }
}

extern "C"
__global__ void rowfilter_bwd(
    float* dest, const float* src, const float *w, int N, int C, int H, int W,
    int Win, int Mlow, int Mhigh, int stride) {
    /* dest - output array. should be same shape as input
       src - input array
       w - input kernel. Should be a 1d array
       N, C, H, W - input tensor sizes
       Mlow - idx of most negative filter tap
       Mhigh - idx of most positive filter tap
       rev - used for calculating gradients - need to do correlation, and
        some funny things with the filter.
    */
    for (int i = stride * (blockIdx.x * blockDim.x + threadIdx.x);
         i < N*C*H*W; i += stride * (blockDim.x * gridDim.x)) {
        const int n = i / C / H / W;
        const int c = (i / H / W) % C;
        const int y = (i / W) % H;
        const int x = i % W;
        float value = 0;
        // Use correlation formula: y[n] = sum h[k]*x[n+k]
#pragma unroll
        for (int k = Mlow; k <= Mhigh; k++) {
            int x_in = x + k;
            int k_in = (x_in < 0 || x_in >= Win) ? -k : k;

            // handle padding - the above complicated equation
            // simply makes sure that the correct index input is used
            // for symmetric padding. I.e. it should result in x_in going from:
            // -3 -2 -1 | 0  1  2  3  4  5  6 | 7  8  9
            //  to:
            //  2  1  0 | 0  1  2  3  4  5  6 | 6  5  4
            // It also allows padding by more than the input length.
            // The group variable will be:
            // 1  1  1  | 0  0  0  0  0  0  0 | 1  1  1  1  1  1 | 0  0  0 ...
            const int group = x_in >= 0 ? ((x_in / Win) % 2)
                                        : 1-(((-x_in-1)/Win) % 2);

            // This does modulo operation but allowing for negative numbers
            // i.e. we want  -2 % 5 = 3. In python this works but in C we it
            // gives -2.
            // On top of reflecting the signal, we also need to reflect the
            // filter around the boundary (unlike with the forward pass).
            const int res = (x_in % Win + Win) % Win;
            x_in = (group == 1) ? (Win-1) - res : res;

            const int offset = n*C*H*Win + c*H*Win + y*Win + x_in;
            value += w[k_in - Mlow] * src[offset];
        }
        dest[i/stride] = value;
    }
}

extern "C"
__global__ void colfilter(
    float* dest, const float* src, const float *w, int N, int C, int H, int W,
    int Hin, int Mlow, int Mhigh, int stride) {
    /* dest - output array. should be same shape as input
       src - input array
       w - input kernel. Should be a 1d array
       N, C, H, W - input tensor sizes
       Mlow - idx of most negative filter tap
       Mhigh - idx of most positive filter tap
       rev - used for calculating gradients - need to do correlation, and
        some funny things with the filter.
    */
    for (int i = stride * (blockIdx.x * blockDim.x + threadIdx.x);
         i < N*C*H*W; i += stride * (blockDim.x * gridDim.x)) {
        const int n = i / C / H / W;
        const int c = (i / H / W) % C;
        const int y = (i / W) % H;
        const int x = i % W;
        float value = 0;
        // Use convolution formula: y[n] = sum h[k]*x[n-k]
#pragma unroll
        for (int k = Mlow; k <= Mhigh; k++) {
            int y_in = y - k;

            // handle padding - the above complicated equation
            // simply makes sure that the correct index input is used
            // for symmetric padding. I.e. it should result in x_in going from:
            // -3 -2 -1 | 0  1  2  3  4  5  6 | 7  8  9
            //  to:
            //  2  1  0 | 0  1  2  3  4  5  6 | 6  5  4
            // It also allows padding by more than the input length.
            // The group variable will be:
            // 1  1  1  | 0  0  0  0  0  0  0 | 1  1  1  1  1  1 | 0  0  0 ...
            const int group = y_in >= 0 ? ((y_in / Hin) % 2)
                                        : 1-(((-y_in-1)/Hin) % 2);

            // This does modulo operation but allowing for negative numbers
            // i.e. we want  -2 % 5 = 3. In python this works but in C we it
            // gives -2.
            // On top of reflecting the signal, we also need to reflect the
            // filter around the boundary (unlike with the forward pass).
            const int res = (y_in % Hin + Hin) % Hin;
            y_in = (group == 1) ? (Hin-1) - res : res;

            const int offset = n*C*Hin*W + c*Hin*W + y_in*W + x;
            value +=  w[k-Mlow] * src[offset];
        }
        dest[i/stride] = value;
    }
}

extern "C"
__global__ void colfilter_bwd(
    float* dest, const float* src, const float *w, int N, int C, int H, int W,
    int Hin, int Mlow, int Mhigh, int stride) {
    /* dest - output array. should be same shape as input
       src - input array
       w - input kernel. Should be a 1d array
       N, C, H, W - input tensor sizes
       Mlow - idx of most negative filter tap
       Mhigh - idx of most positive filter tap
       rev - used for calculating gradients - need to do correlation, and
        some funny things with the filter.
    */
    for (int i = stride * (blockIdx.x * blockDim.x + threadIdx.x);
         i < N*C*H*W; i += stride * (blockDim.x * gridDim.x)) {
        const int n = i / C / H / W;
        const int c = (i / H / W) % C;
        const int y = (i / W) % H;
        const int x = i % W;
        float value = 0;

        // Use correlation formula: y[n] = sum h[k]*x[n+k]
#pragma unroll
        for (int k = Mlow; k <= Mhigh; k++) {
            int y_in = y + k;
            int k_in = (y_in < 0 || y_in >= Hin) ? -k : k;

            // handle padding - the above complicated equation
            // simply makes sure that the correct index input is used
            // for symmetric padding. I.e. it should result in x_in going from:
            // -3 -2 -1 | 0  1  2  3  4  5  6 | 7  8  9
            //  to:
            //  2  1  0 | 0  1  2  3  4  5  6 | 6  5  4
            // It also allows padding by more than the input length.
            // The group variable will be:
            // 1  1  1  | 0  0  0  0  0  0  0 | 1  1  1  1  1  1 | 0  0  0 ...
            const int group = y_in >= 0 ? ((y_in / Hin) % 2)
                                        : 1-(((-y_in-1)/Hin) % 2);

            // This does modulo operation but allowing for negative numbers
            // i.e. we want  -2 % 5 = 3. In python this works but in C we it
            // gives -2.
            // On top of reflecting the signal, we also need to reflect the
            // filter around the boundary (unlike with the forward pass).
            const int res = (y_in % Hin + Hin) % Hin;
            y_in = (group == 1) ? (Hin-1) - res : res;

            const int offset = n*C*Hin*W + c*Hin*W + y_in*W + x;
            value += w[k_in - Mlow] * src[offset];
        }
        dest[i/stride] = value;
    }
}

