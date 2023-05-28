//
// Created by tqcq on 2023/5/28.
//

#include "LlamaCppAgent.h"
#include <absl/base/macros.h>

std::shared_ptr<LlamaCppAgent> LlamaCppAgent::singleton(const LlamaCppAgentOptions& options) {
    return std::shared_ptr<LlamaCppAgent>(new LlamaCppAgent(options));
}

const char *LlamaCppAgent::token_to_str(llama_token token_id) const {
    ABSL_ASSERT(llama_ctx_ != nullptr);
    return llama_token_to_str(llama_ctx_.get(), token_id);
}

std::vector<llama_token> LlamaCppAgent::tokenize(const std::string &text, bool add_bos) const {
    ABSL_ASSERT(llama_ctx_ != nullptr);
    std::vector<llama_token> tokens(text.size() + (add_bos ? 1 : 0));
    llama_tokenize(llama_ctx_.get(), text.c_str(), tokens.data(), (int)tokens.size(), add_bos);
    return tokens;
}

LlamaCppAgent::LlamaCppAgent(const LlamaCppAgentOptions &options)
    : options(options)
{
}

