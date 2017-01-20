#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "activations.h"
#include "stddef.h"
#include "tree.h"

#ifdef __cplusplus
extern "C" {
#endif

struct network_state;

struct layer;
typedef struct layer layer;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    REORG,
    BLANK
} LAYER_TYPE;

typedef enum {
    SSE, MASKED, SMOOTH
} COST_TYPE;

struct layer {

    layer() :
    batch_normalize(0),
    shortcut(0),
    batch(0),
    forced(0),
    flipped(0),
    inputs(0),
    outputs(0),
    truths(0),
    h(0),
    w(0),
    c(0),
    out_h(0),
    out_w(0),
    out_c(0),
    n(0),
    max_boxes(0),
    groups(0),
    size(0),
    side(0),
    stride(0),
    reverse(0),
    pad(0),
    sqrt(0),
    flip(0),
    index(0),
    binary(0),
    xnor(0),
    steps(0),
    hidden(0),
    dot(0.f),
    angle(0.f),
    jitter(0.f),
    saturation(0.f),
    exposure(0.f),
    shift(0.f),
    ratio(0.f),
    softmax(0),
    classes(0),
    coords(0),
    background(0),
    rescore(0),
    objectness(0),
    does_cost(0),
    joint(0),
    noadjust(0),
    reorg(0),
    log(0),

    adam(0),
    B1(0.f),
    B2(0.f),
    eps(0.f),
    t(0),

    alpha(0.f),
    beta(0.f),
    kappa(0.f),

    coord_scale(0.f),
    object_scale(0.f),
    noobject_scale(0.f),
    class_scale(0.f),
    bias_match(0),
    random(0),
    thresh(0.f),
    classfix(0),
    absolute(0),

    dontload(0),
    dontloadscales(0),

    temperature(0.f),
    probability(0.f),
    scale(0.f)
    {
        setDefaults();
    }

    void setDefaults()
    {
        workspace_size = 0;
        cweights = NULL;
        indexes = NULL;
        input_layers = NULL;
        input_sizes = NULL;
        map = NULL;
        rand = NULL;
        cost = NULL;
        state = NULL;
        prev_state = NULL;
        forgot_state = NULL;
        forgot_delta = NULL;
        state_delta = NULL;

        concat = NULL;
        concat_delta = NULL;

        binary_weights = NULL;

        biases = NULL;
        bias_updates = NULL;

        scales = NULL;
        scale_updates = NULL;

        weights = NULL;
        weight_updates = NULL;

        col_image = NULL;
        delta = NULL;
        output = NULL;
        squared = NULL;
        norms = NULL;

        spatial_mean = NULL;
        mean = NULL;
        variance = NULL;

        mean_delta = NULL;
        variance_delta = NULL;

        rolling_mean = NULL;
        rolling_variance = NULL;

        x = NULL;
        x_norm = NULL;

        m = NULL;
        v = NULL;

        z_cpu = NULL;
        r_cpu = NULL;
        h_cpu = NULL;

        binary_input = NULL;

        input_layer = NULL;
        self_layer = NULL;
        output_layer = NULL;

        input_gate_layer = NULL;
        state_gate_layer = NULL;
        input_save_layer = NULL;
        state_save_layer = NULL;
        input_state_layer = NULL;
        state_state_layer = NULL;

        input_z_layer = NULL;
        state_z_layer = NULL;

        input_r_layer = NULL;
        state_r_layer = NULL;

        input_h_layer = NULL;
        state_h_layer = NULL;

        softmax_tree = NULL;

#ifdef GPU
        indexes_gpu = NULL;

        z_gpu = NULL;
        r_gpu = NULL;
        h_gpu = NULL;

        m_gpu = NULL;
        v_gpu = NULL;

         prev_state_gpu = NULL;
         forgot_state_gpu = NULL;
         forgot_delta_gpu = NULL;
         state_gpu = NULL;
         state_delta_gpu = NULL;
         gate_gpu = NULL;
         gate_delta_gpu = NULL;
         save_gpu = NULL;
         save_delta_gpu = NULL;
         concat_gpu = NULL;
         concat_delta_gpu = NULL;

        binary_input_gpu = NULL;
        binary_weights_gpu = NULL;

         mean_gpu = NULL;
         variance_gpu = NULL;

         rolling_mean_gpu = NULL;
         rolling_variance_gpu = NULL;

         variance_delta_gpu = NULL;
         mean_delta_gpu = NULL;

         col_image_gpu = NULL;

         x_gpu = NULL;
         x_norm_gpu = NULL;
         weights_gpu = NULL;
         weight_updates_gpu = NULL;

         biases_gpu = NULL;
         bias_updates_gpu = NULL;

         scales_gpu = NULL;
         scale_updates_gpu = NULL;

         output_gpu = NULL;
         delta_gpu = NULL;
         rand_gpu = NULL;
         squared_gpu = NULL;
         norms_gpu = NULL;
#endif
    }

    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;

    void (*forward)(struct layer, struct network_state);

    void (*backward)(struct layer, struct network_state);

    void (*update)(struct layer, int, float, float, float);

    void (*forward_gpu)(struct layer, struct network_state);

    void (*backward_gpu)(struct layer, struct network_state);

    void (*update_gpu)(struct layer, int, float, float, float);

    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int truths;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;
    int reorg;
    int log;

    int adam;
    float B1;
    float B2;
    float eps;
    int t;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float class_scale;
    int bias_match;
    int random;
    float thresh;
    int classfix;
    int absolute;

    int dontload;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    char *cweights;
    int *indexes;
    int *input_layers;
    int *input_sizes;
    int *map;
    float *rand;
    float *cost;
    float *state;
    float *prev_state;
    float *forgot_state;
    float *forgot_delta;
    float *state_delta;

    float *concat;
    float *concat_delta;

    float *binary_weights;

    float *biases;
    float *bias_updates;

    float *scales;
    float *scale_updates;

    float *weights;
    float *weight_updates;

    float *col_image;
    float *delta;
    float *output;
    float *squared;
    float *norms;

    float *spatial_mean;
    float *mean;
    float *variance;

    float *mean_delta;
    float *variance_delta;

    float *rolling_mean;
    float *rolling_variance;

    float *x;
    float *x_norm;

    float *m;
    float *v;

    float *z_cpu;
    float *r_cpu;
    float *h_cpu;

    float *binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    tree *softmax_tree;

    size_t workspace_size;

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *m_gpu;
    float *v_gpu;

    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float *binary_input_gpu;
    float *binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * col_image_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;

    float * output_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

#ifdef __cplusplus
}
#endif

#endif
