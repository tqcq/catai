//
// Created by tqcq on 2023/5/28.
//

#ifndef CATAI_LLAMACPPAGENT_H
#define CATAI_LLAMACPPAGENT_H

#include "llama.h"
#include "LlamaCppAgentOptions.h"
#include <memory>

class LlamaCppAgent : public std::enable_shared_from_this<LlamaCppAgent> {
public:
    static std::shared_ptr<LlamaCppAgent> singleton(const LlamaCppAgentOptions& options);

    const char* token_to_str(llama_token token_id) const;
    std::vector<llama_token> tokenize(const std::string &text, bool add_bos=false) const;
private:
    LlamaCppAgent(const LlamaCppAgentOptions& options);
    ~LlamaCppAgent() = default;

    LlamaCppAgent(const LlamaCppAgent&) = delete;
    LlamaCppAgent& operator=(const LlamaCppAgent&) = delete;

    static LlamaCppAgent singleton_;

    LlamaCppAgentOptions options;
    std::shared_ptr<llama_context> llama_ctx_;
};


#endif //CATAI_LLAMACPPAGENT_H
