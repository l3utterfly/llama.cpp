#pragma once

// Literal-string path ported from ik_llama.cpp server-context.cpp, revision
// 39ccfcb873901e0195a95d03589a04c1dde21edf. See licenses/LICENSE.ik_llama.cpp.
// Keep matching, buffer sizing, positional bans and retry policy in sync with
// upstream; the server owns the model-specific rewind/checkpoint operations.
#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

struct server_string_ban_params {
    std::vector<std::string> banned_strings;
    float banned_bias = -999.0f;
    int32_t banned_n = 1;
    int32_t banbuffer_size = 0;
    int32_t rewind_count_max = -1;
    bool saturate_predict = false;
};

inline std::string server_string_ban_lower(std::string s) {
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    }
    return s;
}

template <typename TokenOutput>
struct server_string_ban {
    server_string_ban_params params;
    size_t n_buffer = 0;
    int32_t rewind_count = 0;
    std::vector<std::string> ban_phrases;
    std::deque<TokenOutput> token_buffer;
    std::map<int32_t, std::set<int32_t>> positional_bans;

    void init(const server_string_ban_params & p) {
        *this = {};
        params = p;
        for (const auto & val : p.banned_strings) {
            if (!val.empty()) {
                std::string s = server_string_ban_lower(val);
                if (s.length() > n_buffer) n_buffer = s.length();
                ban_phrases.push_back(s);
            }
        }
        std::sort(ban_phrases.begin(), ban_phrases.end(), [](const std::string & a, const std::string & b) {
            return a.length() > b.length();
        });
        if (!ban_phrases.empty()) {
            n_buffer = p.banbuffer_size > 0 ? p.banbuffer_size : n_buffer + 1;
        }
    }

    // n_past is the number of cached tokens, excluding the just-sampled token.
    int32_t check_ban_phrase(int32_t n_past) const {
        if (token_buffer.empty()) return -1;

        std::string string_buffer;
        std::vector<size_t> token_offsets;
        for (const auto & it : token_buffer) {
            token_offsets.push_back(string_buffer.size());
            string_buffer += it.text_to_send;
        }

        size_t best_start = std::string::npos;
        bool found = false;
        std::string string_buffer_lower = server_string_ban_lower(string_buffer);
        for (const auto & phrase : ban_phrases) {
            size_t start = string_buffer_lower.find(phrase);
            if (start != std::string::npos && start < best_start) {
                best_start = start;
                found = true;
            }
        }
        if (found) {
            for (size_t i = 0; i < token_offsets.size(); ++i) {
                size_t len = (i == token_offsets.size() - 1)
                    ? string_buffer.size() - token_offsets[i]
                    : token_offsets[i + 1] - token_offsets[i];
                if (best_start >= token_offsets[i] && best_start < token_offsets[i] + len) {
                    return n_past - (int32_t) token_buffer.size() + 1 + (int32_t) i;
                }
            }
        }
        return -1;
    }

    bool allow_rewind() const {
        if (params.rewind_count_max == -1) {
            return !(rewind_count >= 20 && rewind_count > 2 * ban_phrases.size());
        }
        return params.rewind_count_max <= 0 || rewind_count < params.rewind_count_max;
    }

    // Returns how many entries survive. Every selected token is penalized at
    // ban_pos (not globally, and not at successive positions), as in upstream.
    size_t rewind(int32_t n_past, int32_t ban_pos) {
        rewind_count++;
        int32_t buffer_start_pos = n_past - (int32_t) token_buffer.size() + 1;
        int32_t n_keep_buffer = std::max(0, ban_pos - buffer_start_pos);
        if (params.banned_n != 0) {
            int32_t n = 0;
            for (auto it = token_buffer.begin() + n_keep_buffer; it != token_buffer.end(); ++it) {
                positional_bans[ban_pos].insert(it->tok);
                if (params.banned_n > 0 && ++n == params.banned_n) break;
            }
        }
        token_buffer.resize(n_keep_buffer);
        return n_keep_buffer;
    }
};
