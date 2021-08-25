// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/util/require.h>
#include <vespa/eval/eval/function.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/eval/eval/value_type.h>
#include <vespa/eval/eval/value.h>
#include <vespa/eval/eval/value_codec.h>
#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/lazy_params.h>
#include <vespa/eval/eval/interpreted_function.h>
#include <vespa/eval/eval/feature_name_extractor.h>
#include <vespa/eval/eval/tensor_function.h>
#include <vespa/eval/eval/make_tensor_function.h>
#include <vespa/eval/eval/optimize_tensor_function.h>
#include <vespa/eval/eval/compile_tensor_function.h>
#include <vespa/eval/eval/test/test_io.h>
#include <vespa/vespalib/util/stringfmt.h>

using vespalib::make_string_short::fmt;

using namespace vespalib::eval;
using namespace vespalib::eval::test;

using vespalib::Slime;
using vespalib::slime::JsonFormat;
using vespalib::slime::Inspector;
using vespalib::slime::Cursor;


const auto &factory = FastValueBuilderFactory::get();

int usage(const char *self) {
    //               -------------------------------------------------------------------------------
    fprintf(stderr, "usage: %s [--verbose] <expr> [expr ...]\n", self);
    fprintf(stderr, "  Evaluate a sequence of expressions. The first expression must be\n");
    fprintf(stderr, "  self-contained (no external values). Later expressions may use the\n");
    fprintf(stderr, "  results of earlier expressions. Expressions are automatically named\n");
    fprintf(stderr, "  using single letter symbols ('a' through 'z'). Quote expressions to\n");
    fprintf(stderr, "  make sure they become separate parameters. The --verbose option may\n");
    fprintf(stderr, "  be specified to get more detailed informaion about how the various\n");
    fprintf(stderr, "  expressions are optimized.\n\n");
    fprintf(stderr, "example: %s \"2+2\" \"a+2\" \"a+b\"\n", self);
    fprintf(stderr, "  (a=4, b=6, c=10)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "advanced usage: %s interactive\n", self);
    fprintf(stderr, "  This runs the progam in interactive mode. possible commands (line based):\n");
    fprintf(stderr, "      'exit' -> exit the program\n");
    fprintf(stderr, "      'verbose (true|false)' -> enable or disable verbose output\n");
    fprintf(stderr, "      'def <name> <expr>' -> evaluate expression, bind result to a name\n");
    fprintf(stderr, "      '<expr>' -> evaluate expression\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "advanced usage: %s json-repl\n", self);
    fprintf(stderr, "  This will put the program into a read-eval-print loop where it reads\n");
    fprintf(stderr, "  json objects from stdin and writes json objects to stdout.\n");
    fprintf(stderr, "  possible commands: (object based)\n");
    fprintf(stderr, "    {expr:<expr>, ?name:<name>, ?verbose:true}\n");
    fprintf(stderr, "    -> { result:<verbatim-expr> ?steps:[{class:string,symbol:string}] }\n");
    fprintf(stderr, "      Evaluate an expression and return the result. If a name is specified,\n");
    fprintf(stderr, "      the result will be bound to that name and will be available as a symbol\n");
    fprintf(stderr, "      when doing future evaluations. Verbose output must be enabled for each\n");
    fprintf(stderr, "      relevant command and will result in the 'steps' field being populated in\n");
    fprintf(stderr, "      the response.\n");
    fprintf(stderr, "  if any command fails, the response will be { error:string }\n");
    fprintf(stderr, "\n");
    //               -------------------------------------------------------------------------------
    return 1;
}


int overflow(int cnt, int max) {
    fprintf(stderr, "error: too many expressions: %d (max is %d)\n", cnt, max);
    return 2;
}

class Context
{
private:
    std::vector<vespalib::string> _param_names;
    std::vector<ValueType>        _param_types;
    std::vector<Value::UP>        _param_values;
    std::vector<Value::CREF>      _param_refs;
    bool                          _verbose;
    vespalib::string              _error;
    CTFMetaData                   _meta;

    void clear_state() {
        _error.clear();
        _meta = CTFMetaData();
    }

public:
    Context() : _param_names(), _param_types(), _param_values(), _param_refs(), _verbose(), _meta() {}
    ~Context();

    void verbose(bool value) { _verbose = value; }
    bool verbose() const { return _verbose; }

    Value::UP eval(const vespalib::string &expr) {
        clear_state();
        SimpleObjectParams params(_param_refs);
        auto fun = Function::parse(_param_names, expr, FeatureNameExtractor());
        if (fun->has_error()) {
            _error = fmt("expression parsing failed: %s", fun->get_error().c_str());
            return {};
        }
        NodeTypes types = NodeTypes(*fun, _param_types);
        ValueType res_type = types.get_type(fun->root());
        if (res_type.is_error() || !types.errors().empty()) {
            _error = fmt("type resolving failed for expression: '%s'", expr.c_str());
            for (const auto &issue: types.errors()) {
                _error.append(fmt("\n  type issue: %s", issue.c_str()));
            }
            return {};
        }
        vespalib::Stash stash;
        const TensorFunction &plain_fun = make_tensor_function(factory, fun->root(), types, stash);
        const TensorFunction &optimized = optimize_tensor_function(factory, plain_fun, stash);
        InterpretedFunction ifun(factory, optimized, _verbose ? &_meta : nullptr);
        InterpretedFunction::Context ctx(ifun);
        auto result = factory.copy(ifun.eval(ctx, params));
        REQUIRE_EQ(result->type(), res_type);
        return result;
    }

    const vespalib::string &error() const { return _error; }
    const CTFMetaData &meta() const { return _meta; }

    void save(const vespalib::string &name, Value::UP value) {
        REQUIRE(value);
        for (size_t i = 0; i < _param_names.size(); ++i) {
            if (_param_names[i] == name) {
                _param_types[i] = value->type();
                _param_values[i] = std::move(value);
                _param_refs[i] = *_param_values[i];
                return;
            }
        }
        _param_names.push_back(name);
        _param_types.push_back(value->type());
        _param_values.push_back(std::move(value));
        _param_refs.emplace_back(*_param_values.back());
    }
};
Context::~Context() = default;

void print_error(const vespalib::string &error) {
    fprintf(stderr, "error: %s\n", error.c_str());
}

void print_value(const Value &value, const vespalib::string &name, const CTFMetaData &meta) {
    bool with_name = !name.empty();
    bool with_meta = !meta.steps.empty();
    auto spec = spec_from_value(value);
    if (with_meta) {
        if (with_name) {
            fprintf(stderr, "meta-data(%s):\n", name.c_str());
        } else {
            fprintf(stderr, "meta-data:\n");
        }
        for (const auto &step: meta.steps) {
            fprintf(stderr, "  class: %s\n", step.class_name.c_str());
            fprintf(stderr, "    symbol: %s\n", step.symbol_name.c_str());
        }
    }
    if (with_name) {
        fprintf(stdout, "%s: ", name.c_str());
    }
    if (value.type().is_double()) {
        fprintf(stdout, "%.32g\n", spec.as_double());
    } else {
        fprintf(stdout, "%s\n", spec.to_string().c_str());
    }
}

void handle_message(Context &ctx, const Inspector &req, Cursor &reply) {
    vespalib::string expr = req["expr"].asString().make_string();
    vespalib::string name = req["name"].asString().make_string();
    ctx.verbose(req["verbose"].asBool());
    if (expr.empty()) {
        reply.setString("error", "missing expression (field name: 'expr')");
        return;
    }
    auto value = ctx.eval(expr);
    if (!value) {
        reply.setString("error", ctx.error());
        return;
    }
    reply.setString("result", spec_from_value(*value).to_expr());
    if (!name.empty()) {
        ctx.save(name, std::move(value));
    }
    if (!ctx.meta().steps.empty()) {
        auto &steps_out = reply.setArray("steps");
        for (const auto &step: ctx.meta().steps) {
            auto &step_out = steps_out.addObject();
            step_out.setString("class", step.class_name);
            step_out.setString("symbol", step.symbol_name);            
        }
    }
}

const vespalib::string exit_cmd("exit");
const vespalib::string verbose_cmd("verbose ");
const vespalib::string def_cmd("def ");

bool is_only_whitespace(const vespalib::string &str) {
    for (auto c: str) {
        if (!isspace(c)) {
            return false;
        }
    }
    return true;
}

int interactive_mode(Context &ctx) {
    StdIn std_in;
    LineReader input(std_in);
    vespalib::string line;
    while (input.read_line(line)) {
        if (is_only_whitespace(line)) {
            continue;
        }
        if (line.find(exit_cmd) == 0) {
            return 0;
        }
        if (line.find(verbose_cmd) == 0) {
            auto flag_str = line.substr(verbose_cmd.size());
            bool flag = (flag_str == "true");
            bool bad = (!flag && (flag_str != "false"));
            if (bad) {
                fprintf(stderr, "bad flag specifier: '%s', must be 'true' or 'false'\n", flag_str.c_str());                
            } else {
                ctx.verbose(flag);
                fprintf(stdout, "verbose set to %s\n", flag ? "true" : "false");
            }
            continue;
        }
        vespalib::string expr;
        vespalib::string name;
        if (line.find(def_cmd) == 0) {
            auto name_size = (line.find(" ", def_cmd.size()) - def_cmd.size());
            name = line.substr(def_cmd.size(), name_size);
            expr = line.substr(def_cmd.size() + name_size + 1);
        } else {
            expr = line;
        }
        if (ctx.verbose()) {
            fprintf(stderr, "eval '%s'", expr.c_str());
            if (name.empty()) {
                fprintf(stderr, "\n");
            } else {
                fprintf(stderr, " -> '%s'\n", name.c_str());
            }
        }
        if (auto value = ctx.eval(expr)) {
            print_value(*value, name, ctx.meta());
            if (!name.empty()) {
                ctx.save(name, std::move(value));
            }
        } else {
            print_error(ctx.error());
        }
    }
    return 0;
}

int json_repl_mode(Context &ctx) {
    StdIn std_in;
    StdOut std_out;
    for (;;) {
        if (look_for_eof(std_in)) {
            return 0;
        }
        Slime req;
        if (!JsonFormat::decode(std_in, req)) {
            return 3;
        }
        Slime reply;
        if (req.get().type().getId() == vespalib::slime::ARRAY::ID) {
            reply.setArray();
            for (size_t i = 0; i < req.get().entries(); ++i) {
                handle_message(ctx, req[i], reply.get().addObject());
            }
        } else {
            handle_message(ctx, req.get(), reply.setObject());
        }
        write_compact(reply, std_out);
    }
}

int main(int argc, char **argv) {
    bool verbose = ((argc > 1) && (vespalib::string(argv[1]) == "--verbose"));
    int expr_idx = verbose ? 2 : 1;
    int expr_cnt = (argc - expr_idx);
    int expr_max = ('z' - 'a') + 1;
    if (expr_cnt == 0) {
        return usage(argv[0]);
    }
    if (expr_cnt > expr_max) {
        return overflow(expr_cnt, expr_max);
    }
    Context ctx;
    if ((expr_cnt == 1) && (vespalib::string(argv[expr_idx]) == "interactive")) {
        return interactive_mode(ctx);
    }
    if ((expr_cnt == 1) && (vespalib::string(argv[expr_idx]) == "json-repl")) {
        return json_repl_mode(ctx);
    }
    ctx.verbose(verbose);
    vespalib::string name("a");
    for (int i = expr_idx; i < argc; ++i) {
        if (auto value = ctx.eval(argv[i])) {
            if (expr_cnt > 1) {
                print_value(*value, name, ctx.meta());
                ctx.save(name, std::move(value));
                ++name[0];
            } else {
                vespalib::string no_name;
                print_value(*value, no_name, ctx.meta());
            }
        } else {
            print_error(ctx.error());
            return 3;
        }
    }
    return 0;
}
