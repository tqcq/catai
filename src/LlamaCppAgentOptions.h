//
// Created by tqcq on 2023/5/28.
//

#ifndef CATAI_LLAMACPPAGENTOPTIONS_H
#define CATAI_LLAMACPPAGENTOPTIONS_H

#include <string>

/**
 * model_path: path to the model
 * n_ctx     : context size
 * seed      : random seed
 * f16_kv    : use float16 for KV cache
 * use_mmap  : use mmap to load model
 * use_mlock : use mlock to lock memory
 * n_gpu_layers: number of layers to distribute on GPUs
 */

struct LlamaCppAgentOptions {
    std::string model_path;
    int n_ctx;
    int seed;
    bool f16_kv;
    bool use_mmap;
    bool use_mlock;
    int n_gpu_layers;
};


#endif //CATAI_LLAMACPPAGENTOPTIONS_H
