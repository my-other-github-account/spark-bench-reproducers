#!/bin/bash
# For long prompt benchmarks, keep the target model on the FlashQLA AR path
# while leaving DFlash active for shorter decode-heavy requests.
set -euo pipefail

VLLM_DIR="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages/vllm}"
RUNNER="${VLLM_DIR}/v1/worker/gpu_model_runner.py"
SCHED="${VLLM_DIR}/v1/core/sched/scheduler.py"
TF_BASE="${VLLM_DIR}/model_executor/models/transformers/base.py"

python3 - <<'PY'
import pathlib
import sys

vllm_dir = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
runner = vllm_dir / "v1/worker/gpu_model_runner.py"
sched = vllm_dir / "v1/core/sched/scheduler.py"
tf_base = vllm_dir / "model_executor/models/transformers/base.py"

src = runner.read_text()
if "Combined DFlash prompt-threshold AR path active" not in src:
    if "import os\n" not in src[:1000]:
        src = src.replace("import itertools\n", "import itertools\nimport os\n", 1)

    old = """            logits_indices, spec_decode_metadata = self._prepare_inputs(\n                scheduler_output,\n                num_scheduled_tokens_np,\n            )\n\n            cascade_attn_prefix_lens = None\n"""
    new = """            logits_indices, spec_decode_metadata = self._prepare_inputs(\n                scheduler_output,\n                num_scheduled_tokens_np,\n            )\n\n            cascade_attn_prefix_lens = None\n"""
    if old not in src:
        print("prompt-threshold: runner pre-prepare anchor not found")
        sys.exit(1)
    src = src.replace(old, new, 1)

    old = """            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0\n            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices\n"""
    new = """            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0\n            skip_spec_decode_once = False\n            if (\n                self.speculative_config is not None\n                and self.speculative_config.use_dflash()\n                and not use_spec_decode\n            ):\n                prompt_threshold_raw = os.environ.get(\"VLLM_DFLASH_AR_PROMPT_THRESHOLD\", \"0\")\n                try:\n                    prompt_threshold = int(prompt_threshold_raw)\n                except ValueError:\n                    prompt_threshold = 0\n                if prompt_threshold > 0 and max_num_scheduled_tokens >= prompt_threshold:\n                    skip_spec_decode_once = True\n                    logger.info_once(\n                        \"Combined DFlash prompt-threshold AR path active: threshold=%s max_num_scheduled_tokens=%s\",\n                        prompt_threshold,\n                        max_num_scheduled_tokens,\n                    )\n            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices\n"""
    if old not in src:
        print("prompt-threshold: runner use_spec anchor not found")
        sys.exit(1)
    src = src.replace(old, new, 1)

    old = """            if self.use_aux_hidden_state_outputs:\n                # True when EAGLE 3 is used.\n                hidden_states, aux_hidden_states = model_output\n            else:\n                # Common case.\n                hidden_states = model_output\n                aux_hidden_states = None\n"""
    new = """            if self.use_aux_hidden_state_outputs and not skip_spec_decode_once:\n                # True when EAGLE 3 / DFlash aux hidden states are used.\n                hidden_states, aux_hidden_states = model_output\n            else:\n                # Common case, plus long-prompt DFlash threshold AR path.\n                if skip_spec_decode_once and isinstance(model_output, tuple):\n                    hidden_states = model_output[0]\n                else:\n                    hidden_states = model_output\n                aux_hidden_states = None\n"""
    if old not in src:
        print("prompt-threshold: runner unpack anchor not found")
        sys.exit(1)
    src = src.replace(old, new, 1)

    old = """        if spec_config is not None:\n            # Decide whether to run the drafter or zero out draft tokens.\n"""
    new = """        dflash_prompt_threshold_skip_propose = False\n        if spec_config is not None and spec_config.use_dflash():\n            prompt_threshold_raw = os.environ.get(\"VLLM_DFLASH_AR_PROMPT_THRESHOLD\", \"0\")\n            try:\n                prompt_threshold = int(prompt_threshold_raw)\n            except ValueError:\n                prompt_threshold = 0\n            num_reqs_for_threshold = self.input_batch.num_reqs\n            max_prompt_tokens_for_threshold = 0\n            if prompt_threshold > 0 and num_reqs_for_threshold > 0:\n                max_prompt_tokens_for_threshold = int(\n                    self.input_batch.num_prompt_tokens[:num_reqs_for_threshold].max())\n            dflash_prompt_threshold_skip_propose = (\n                prompt_threshold > 0\n                and max_prompt_tokens_for_threshold >= prompt_threshold\n            )\n            if dflash_prompt_threshold_skip_propose:\n                self._draft_token_ids = torch.zeros(\n                    1, device=self.device, dtype=torch.int32\n                ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)\n                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)\n                logger.info_once(\n                    \"Combined DFlash prompt-threshold proposer skip active: threshold=%s max_prompt_tokens=%s\",\n                    prompt_threshold,\n                    max_prompt_tokens_for_threshold,\n                )\n        if spec_config is not None and aux_hidden_states is not None and not dflash_prompt_threshold_skip_propose:\n            # Decide whether to run the drafter or zero out draft tokens.\n"""
    if old not in src:
        print("prompt-threshold: runner spec gate anchor not found")
        sys.exit(1)
    src = src.replace(old, new, 1)

    old = """        defer_kv_connector_finalize = self.speculative_config is not None\n        with (\n            set_forward_context(\n"""
    new = """        defer_kv_connector_finalize = self.speculative_config is not None\n        aux_kwargs_restore = None\n        if skip_spec_decode_once and self.use_aux_hidden_state_outputs:\n            model_obj = self.model\n            aux_kwargs = getattr(model_obj, \"_output_aux_hidden_states_kwargs\", None)\n            if aux_kwargs:\n                aux_kwargs_restore = dict(aux_kwargs)\n                aux_kwargs.clear()\n                logger.info_once(\"Combined DFlash prompt-threshold runner aux capture disabled\")\n        try:\n            with (\n                set_forward_context(\n"""
    if old not in src:
        print("prompt-threshold: runner forward-enter anchor not found")
        sys.exit(1)
    src = src.replace(old, new, 1)

    old = """        ):\n            model_output = self._model_forward(\n                input_ids=input_ids,\n                positions=positions,\n                intermediate_tensors=intermediate_tensors,\n                inputs_embeds=inputs_embeds,\n                **model_kwargs,\n            )\n\n        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n"""
    new = """            ):\n                model_output = self._model_forward(\n                    input_ids=input_ids,\n                    positions=positions,\n                    intermediate_tensors=intermediate_tensors,\n                    inputs_embeds=inputs_embeds,\n                    **model_kwargs,\n                )\n        finally:\n            if aux_kwargs_restore is not None:\n                getattr(self.model, \"_output_aux_hidden_states_kwargs\").update(aux_kwargs_restore)\n\n        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n"""
    if old not in src:
        print("prompt-threshold: runner forward-exit anchor not found")
        sys.exit(1)
    src = src.replace(old, new, 1)

    old = """        with record_function_or_nullcontext(\"gpu_model_runner: sample\"):\n            sampler_output = self._sample(logits, spec_decode_metadata)\n\n        self._update_states_after_model_execute(\n            sampler_output.sampled_token_ids, scheduler_output\n        )\n"""
    new = """        with record_function_or_nullcontext(\"gpu_model_runner: sample\"):\n            sampler_output = self._sample(logits, spec_decode_metadata)\n\n        spec_config_for_threshold = self.speculative_config\n        if spec_config_for_threshold is not None and spec_config_for_threshold.use_dflash():\n            prompt_threshold_raw = os.environ.get(\"VLLM_DFLASH_AR_PROMPT_THRESHOLD\", \"0\")\n            try:\n                prompt_threshold = int(prompt_threshold_raw)\n            except ValueError:\n                prompt_threshold = 0\n            num_reqs_for_threshold = self.input_batch.num_reqs\n            max_prompt_tokens_for_threshold = 0\n            if prompt_threshold > 0 and num_reqs_for_threshold > 0:\n                max_prompt_tokens_for_threshold = int(\n                    self.input_batch.num_prompt_tokens[:num_reqs_for_threshold].max())\n            if (\n                prompt_threshold > 0\n                and max_prompt_tokens_for_threshold >= prompt_threshold\n                and sampler_output.sampled_token_ids.shape[-1] > 1\n            ):\n                sampler_output.sampled_token_ids = sampler_output.sampled_token_ids[:, :1]\n                scheduler_output.scheduled_spec_decode_tokens.clear()\n                logger.info_once(\n                    \"Combined DFlash prompt-threshold sampler trim active: threshold=%s max_prompt_tokens=%s\",\n                    prompt_threshold,\n                    max_prompt_tokens_for_threshold,\n                )\n\n        self._update_states_after_model_execute(\n            sampler_output.sampled_token_ids, scheduler_output\n        )\n"""
    if old not in src:
        print("prompt-threshold: runner sampler-trim anchor not found")
        sys.exit(1)
    src = src.replace(old, new, 1)
    runner.write_text(src)

s = sched.read_text()
if "Combined DFlash prompt-threshold scheduler draft clear active" not in s:
    if "import os\n" not in s[:1000]:
        s = s.replace("from collections.abc import Iterable\n",
                      "from collections.abc import Iterable\nimport os\n", 1)
    old = """            # Add newly generated spec token ids to the request.\n            if self.structured_output_manager.should_advance(request):\n"""
    new = """            threshold_raw = os.environ.get(\"VLLM_DFLASH_AR_PROMPT_THRESHOLD\", \"0\")\n            try:\n                threshold = int(threshold_raw)\n            except ValueError:\n                threshold = 0\n            if threshold > 0 and request.num_prompt_tokens >= threshold:\n                request.spec_token_ids = []\n                request.num_output_placeholders = 0\n                logger.info_once(\n                    \"Combined DFlash prompt-threshold scheduler draft clear active: threshold=%s prompt_tokens=%s\",\n                    threshold,\n                    request.num_prompt_tokens,\n                )\n                continue\n\n            # Add newly generated spec token ids to the request.\n            if self.structured_output_manager.should_advance(request):\n"""
    if old not in s:
        print("prompt-threshold: scheduler update anchor not found")
        sys.exit(1)
    s = s.replace(old, new, 1)

    old = """            placeholder_spec_tokens = sched_spec_tokens.get(req_id)\n            if not placeholder_spec_tokens:\n                continue\n\n            orig_num_spec_tokens = len(placeholder_spec_tokens)\n"""
    new = """            threshold_raw = os.environ.get(\"VLLM_DFLASH_AR_PROMPT_THRESHOLD\", \"0\")\n            try:\n                threshold = int(threshold_raw)\n            except ValueError:\n                threshold = 0\n            if threshold > 0 and request.num_prompt_tokens >= threshold:\n                sched_spec_tokens.pop(req_id, None)\n                request.spec_token_ids = []\n                request.num_output_placeholders = 0\n                logger.info_once(\n                    \"Combined DFlash prompt-threshold async draft clear active: threshold=%s prompt_tokens=%s\",\n                    threshold,\n                    request.num_prompt_tokens,\n                )\n                continue\n\n            placeholder_spec_tokens = sched_spec_tokens.get(req_id)\n            if not placeholder_spec_tokens:\n                continue\n\n            orig_num_spec_tokens = len(placeholder_spec_tokens)\n"""
    if old not in s:
        print("prompt-threshold: scheduler async-output anchor not found")
        sys.exit(1)
    s = s.replace(old, new, 1)
    sched.write_text(s)

t = tf_base.read_text()
if "Combined DFlash prompt-threshold aux capture disabled" not in t:
    if "import os\n" not in t[:1000]:
        t = t.replace("import contextlib\n", "import contextlib\nimport os\n", 1)
    old = """        outputs = self.model(\n            input_ids=input_ids,\n            inputs_embeds=inputs_embeds,\n            use_cache=False,\n            position_ids=positions,\n            attention_instances=self.attention_instances,\n            return_dict=False,\n            **self._output_aux_hidden_states_kwargs,\n            **kwargs,\n        )\n\n        # Remove batch dimension after exiting Transformers model\n        hidden_states = outputs[0][0, ...]\n        if self._output_aux_hidden_states_kwargs:\n            aux_hidden_states = [x[0][0, ...] for x in outputs[1:]]\n"""
    new = """        prompt_threshold_raw = os.environ.get(\"VLLM_DFLASH_AR_PROMPT_THRESHOLD\", \"0\")\n        try:\n            prompt_threshold = int(prompt_threshold_raw)\n        except ValueError:\n            prompt_threshold = 0\n        seq_len = int(positions.shape[-1]) if positions is not None else 0\n        output_aux_kwargs = self._output_aux_hidden_states_kwargs\n        if prompt_threshold > 0 and seq_len >= prompt_threshold:\n            output_aux_kwargs = {}\n            import logging as _logging\n            _logging.getLogger(\"vllm\").info_once(\n                \"Combined DFlash prompt-threshold aux capture disabled: threshold=%s seq_len=%s\",\n                prompt_threshold,\n                seq_len,\n            )\n\n        outputs = self.model(\n            input_ids=input_ids,\n            inputs_embeds=inputs_embeds,\n            use_cache=False,\n            position_ids=positions,\n            attention_instances=self.attention_instances,\n            return_dict=False,\n            **output_aux_kwargs,\n            **kwargs,\n        )\n\n        # Remove batch dimension after exiting Transformers model\n        hidden_states = outputs[0][0, ...]\n        aux_hidden_states = []\n        if output_aux_kwargs:\n            aux_hidden_states = [x[0][0, ...] for x in outputs[1:]]\n"""
    if old not in t:
        print("prompt-threshold: transformers forward anchor not found")
        sys.exit(1)
    t = t.replace(old, new, 1)
    tf_base.write_text(t)

print("prompt-threshold: applied")
PY
