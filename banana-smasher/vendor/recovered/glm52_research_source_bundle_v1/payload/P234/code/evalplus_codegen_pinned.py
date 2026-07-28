#!${MACMINI_HOME}/venvs/evalplus/bin/python
import argparse
import json
import evalplus.codegen as codegen_module

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--model", default="deepseek-v4-flash-ud-iq4-xs")
p.add_argument("--base-url", default="http://${MODEL_HOST}:8356/v1")
p.add_argument("--id-range", nargs=2, type=int, metavar=("LOW", "HIGH"))
p.add_argument("--max-new-tokens", type=int, default=4096)
a = p.parse_args()

config = {
    "model": a.model,
    "dataset": "humaneval",
    "backend": "openai",
    "base_url": a.base_url,
    "greedy": True,
    "max_new_tokens": a.max_new_tokens,
    "id_range": a.id_range,
}
print("EFFECTIVE_CONFIG=" + json.dumps(config, sort_keys=True), flush=True)

# EvalPlus 26d6d00 accepts provider kwargs in run_codegen(), but its
# make_model(openai) branch silently drops **kwargs. This runtime shim keeps the
# exact upstream generation path and explicitly binds the requested cap on the
# returned decoder; without it, every advertised --max-new-tokens CLI value is
# actually the DecoderBase default of 768.
original_make_model = codegen_module.make_model

def make_model_with_openai_cap(*args, **kwargs):
    model = original_make_model(*args, **kwargs)
    if kwargs.get("backend") == "openai":
        model.max_new_tokens = int(kwargs["max_new_tokens"])
        print(f"EFFECTIVE_DECODER_MAX_NEW_TOKENS={model.max_new_tokens}", flush=True)
    return model

codegen_module.make_model = make_model_with_openai_cap
result = codegen_module.run_codegen(root=a.root, **config)
print(result)
