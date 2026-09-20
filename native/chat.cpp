#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <string>

// Only trusted instructions and a data placeholder reach this renderer. A
// single user message works for templates that reject a separate system role.
extern "C" int32_t sd_render_chat(const char * tmpl, const char * user,
        const char * bos, const char * eos, char * out, int32_t capacity) noexcept {
    try {
        if (!tmpl || !*tmpl) return -1;
        jinja::lexer lexer;
        auto program = jinja::parse_from_tokens(lexer.tokenize(tmpl));
        jinja::context context(tmpl);
        const common_json variables = {
            {"messages", common_json::array({common_json{{"role", "user"}, {"content", user}}})},
            {"add_generation_prompt", true},
            {"enable_thinking", false},
            {"bos_token", bos},
            {"eos_token", eos},
        };
        jinja::global_from_json(context, variables, false);
        jinja::runtime runtime(context);
        const auto rendered = runtime.gather_string_parts(runtime.execute(program))->as_string().str();
        if (rendered.size() > INT_MAX) return -1;
        if (out && capacity > 0)
            std::memcpy(out, rendered.data(), std::min(rendered.size(), size_t(capacity)));
        return int32_t(rendered.size());
    } catch (...) { return -1; }
}
