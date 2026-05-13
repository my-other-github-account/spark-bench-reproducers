from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py")

payload = r'''

# --- pp32k DFlash rescue shifted suffix block-table proof patch ---
import os as _dflash_suffix_os

_DFLASH_SHIFTED_SUFFIX_ORIG_SET_INPUTS = DFlashProposer.set_inputs_first_pass
_DFLASH_SHIFTED_SUFFIX_CALLS = 0


def _dflash_shifted_suffix_set_inputs(
    self,
    target_token_ids,
    next_token_ids,
    target_positions,
    target_hidden_states,
    token_indices_to_sample,
    cad,
    num_rejected_tokens_gpu,
):
    global _DFLASH_SHIFTED_SUFFIX_CALLS
    _DFLASH_SHIFTED_SUFFIX_CALLS += 1
    result = _DFLASH_SHIFTED_SUFFIX_ORIG_SET_INPUTS(
        self,
        target_token_ids,
        next_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        cad,
        num_rejected_tokens_gpu,
    )
    num_query_total, token_indices_to_sample, new_cad = result

    try:
        if int(new_cad.num_reqs) != 1 or bool(new_cad.causal):
            return result
        block_size = int(getattr(self, "block_size", 0))
        if block_size <= 0:
            return result
        num_query_per_req = int(1 + self.num_speculative_tokens)
        seq_total = int(new_cad.seq_lens[0].item())
        context_len = max(0, seq_total - num_query_per_req)
        suffix_blocks = int(
            _dflash_suffix_os.environ.get("DFLASH_SUFFIX_CONTEXT_BLOCKS", "104")
        )
        suffix_tokens = max(block_size, suffix_blocks * block_size)
        if context_len <= suffix_tokens:
            return result
        suffix_token_start = ((context_len - suffix_tokens) // block_size) * block_size
        suffix_block_start = suffix_token_start // block_size
        suffix_intrablock_offset = suffix_token_start % block_size
        if suffix_intrablock_offset != 0:
            logger.info(
                "DFlash shifted suffix block-table skipped: suffix_intrablock_offset=%s context_len=%s suffix_tokens=%s block_size=%s",
                suffix_intrablock_offset,
                context_len,
                suffix_tokens,
                block_size,
            )
            return result

        shifted_block_table = new_cad.block_table_tensor[:, suffix_block_start:]
        shifted_seq_lens = new_cad.seq_lens - suffix_token_start
        shifted_upper = (
            new_cad.seq_lens_cpu_upper_bound - suffix_token_start
            if new_cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        shifted_max_seq_len = max(1, int(new_cad.max_seq_len) - suffix_token_start)
        shifted_cad = CommonAttentionMetadata(
            query_start_loc=new_cad.query_start_loc,
            query_start_loc_cpu=new_cad.query_start_loc_cpu,
            seq_lens=shifted_seq_lens,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=shifted_upper,
            num_reqs=new_cad.num_reqs,
            num_actual_tokens=new_cad.num_actual_tokens,
            max_query_len=new_cad.max_query_len,
            max_seq_len=shifted_max_seq_len,
            block_table_tensor=shifted_block_table,
            slot_mapping=new_cad.slot_mapping,
            causal=False,
        )
        if _DFLASH_SHIFTED_SUFFIX_CALLS <= 80 or _DFLASH_SHIFTED_SUFFIX_CALLS % 50 == 0:
            logger.info(
                "DFlash shifted suffix block-table path active: call=%s suffix_token_start=%s suffix_block_start=%s suffix_intrablock_offset=%s context_len_before=%s seq_total_before=%s seq_total_after=%s max_seq_before=%s max_seq_after=%s suffix_context_blocks=%s block_size=%s absolute_positions_preserved=True block_table_shifted=True seq_lens_adjusted=True slot_mapping_preserved=True causal_false_preserved=True",
                _DFLASH_SHIFTED_SUFFIX_CALLS,
                suffix_token_start,
                suffix_block_start,
                suffix_intrablock_offset,
                context_len,
                seq_total,
                int(shifted_seq_lens[0].item()),
                int(new_cad.max_seq_len),
                shifted_max_seq_len,
                suffix_blocks,
                block_size,
            )
        return num_query_total, token_indices_to_sample, shifted_cad
    except Exception as exc:
        logger.exception("DFlash shifted suffix block-table patch failed open: %s", exc)
        return result


DFlashProposer.set_inputs_first_pass = _dflash_shifted_suffix_set_inputs
# --- end pp32k DFlash rescue shifted suffix block-table proof patch ---
'''

text = path.read_text()
marker = "pp32k DFlash rescue shifted suffix block-table proof patch"
if marker not in text:
    path.with_suffix(path.suffix + ".bak_shifted_suffix_block_table").write_text(text)
    path.write_text(text + payload)
print("shifted-suffix-block-table-patched", path)
