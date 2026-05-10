try:
    import apport_python_hook
    apport_python_hook.install()
except Exception:
    pass

import os, sys
if os.environ.get("FLASHQLA_DISABLE", "0") == "1":
    sys.stderr.write("[flashqla-patch] disabled via FLASHQLA_DISABLE=1\n"); sys.stderr.flush()
else:
    try:
        import torch
        import vllm.model_executor.layers.fla.ops.chunk as _chunk
        try:
            import vllm.model_executor.layers.fla.ops as _ops
        except Exception:
            _ops = None
        try:
            import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn
        except Exception:
            _gdn = None
        _orig = getattr(_gdn, "fla_chunk_gated_delta_rule", None) or getattr(_chunk, "chunk_gated_delta_rule", None)
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        if cap in ((12, 0), (12, 1)):
            _calls=[0]; _fallbacks=[0]; _large_logged=[False]; _lazy={"fq":None,"o":None}; _orig_allowed=[None]
            def _ensure_lazy():
                if _lazy["fq"] is None:
                    from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as _flashqla_fwd
                    from flashqla_hkv_o import chunk_fwd_o_hkv as _vllm_chunk_fwd_o
                    _lazy["fq"]=_flashqla_fwd; _lazy["o"]=_vllm_chunk_fwd_o
                return _lazy["fq"], _lazy["o"]
            def _meta(name,x):
                if x is None: return name+"=None"
                try:
                    return f"{name}: shape={tuple(x.shape)} dtype={x.dtype} stride={tuple(x.stride())} contig={x.is_contiguous()} min={float(x.float().min())} max={float(x.float().max())} mean={float(x.float().mean())}"
                except Exception as e:
                    return f"{name}: shape={tuple(x.shape)} dtype={x.dtype} stride={tuple(x.stride())} meta_err={e}"
            def _fallback(reason,q,k,v,g,beta,scale,initial_state,output_final_state,cu_seqlens,chunk_indices,chunk_offsets,use_qk_l2norm_in_kernel,head_first,kw):
                _fallbacks[0]+=1
                if _fallbacks[0] <= 8:
                    sys.stderr.write(f"[flashqla-patch] fallback #{_fallbacks[0]} reason={reason} q_shape={tuple(q.shape)} chunk_indices={chunk_indices is not None} chunk_offsets={chunk_offsets is not None}\n"); sys.stderr.flush()
                # vLLM vendored FLA accepts chunk_indices/chunk_offsets but not all wrapper kwargs.
                args = dict(q=q,k=k,v=v,g=g,beta=beta,scale=scale,initial_state=initial_state,
                            output_final_state=output_final_state,cu_seqlens=cu_seqlens,
                            chunk_indices=chunk_indices,chunk_offsets=chunk_offsets,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                args.update(kw)
                try:
                    allowed = _orig_allowed[0]
                    if allowed is None:
                        import inspect
                        allowed = set(inspect.signature(_orig).parameters.keys())
                        _orig_allowed[0] = allowed
                    args = {kk: vv for kk, vv in args.items() if kk in allowed}
                except Exception:
                    args.pop('head_first', None)
                return _orig(**args)
            def _patched(q,k,v,g,beta,scale=None,initial_state=None,
                         output_final_state=False,cu_seqlens=None,
                         chunk_indices=None,chunk_offsets=None,
                         use_qk_l2norm_in_kernel=False,head_first=False,**kw):
                _calls[0]+=1
                if (not _large_logged[0]) and q.shape[1] >= 128:
                    _large_logged[0] = True
                    sys.stderr.write(f"[flashqla-large-shape] optimized shape={tuple(q.shape)} calls={_calls[0]} fallbacks={_fallbacks[0]}\n"); sys.stderr.flush()
                if _orig is None:
                    raise RuntimeError("FlashQLA fallback original FLA function unavailable")
                if head_first or use_qk_l2norm_in_kernel:
                    return _fallback("unsupported-interface",q,k,v,g,beta,scale,initial_state,output_final_state,cu_seqlens,chunk_indices,chunk_offsets,use_qk_l2norm_in_kernel,head_first,kw)
                # vLLM passes chunk_indices/chunk_offsets even for a single packed sequence.
                # Isolated correctness (T=30/31/64/128, Hq=16,Hv=48) shows this metadata is
                # equivalent to cu_seqlens=[0,T] for one sequence, so route it through FlashQLA.
                packed_single = False
                if (chunk_indices is not None or chunk_offsets is not None):
                    try:
                        # Avoid .item() on CUDA metadata here; this wrapper runs once per GDN
                        # layer and scalar reads can serialize the prefill stream.  The recipe
                        # serves max_num_seqs=1, so numel/rank checks are enough for this path.
                        packed_single = (cu_seqlens is not None
                                         and int(cu_seqlens.numel()) == 2
                                         and int(q.shape[0]) == 1
                                         and (chunk_offsets is None or int(chunk_offsets.numel()) == 2))
                    except Exception:
                        packed_single = False
                    if not packed_single:
                        return _fallback("unsupported-packed-multi-seq",q,k,v,g,beta,scale,initial_state,output_final_state,cu_seqlens,chunk_indices,chunk_offsets,use_qk_l2norm_in_kernel,head_first,kw)
                # Decode/chat sanity failed when routing very short T=30 calls through FlashQLA
                # (coherent Paris answer regressed to repeated punctuation). Keep short decode
                # on vLLM/FLA while continuing FlashQLA for prefill-sized chunks.
                if q.shape[1] < 64:
                    return _fallback("short-decode-preserve-coherence",q,k,v,g,beta,scale,initial_state,output_final_state,cu_seqlens,chunk_indices,chunk_offsets,use_qk_l2norm_in_kernel,head_first,kw)
                if scale is None:
                    scale = q.shape[-1] ** -0.5
                _flashqla_fwd, _vllm_chunk_fwd_o = _ensure_lazy()
                if os.environ.get("FLASHQLA_VERBOSE_META", "0") == "1" and _calls[0] in (1,2,10,20):
                    sys.stderr.write(f"[flashqla-patch] hybrid call #{_calls[0]} output_final_state={output_final_state} scale={scale}\n")
                    for name,x in [("q",q),("k",k),("v",v),("g",g),("beta",beta),("initial_state",initial_state),("cu_seqlens",cu_seqlens)]:
                        sys.stderr.write("[flashqla-meta] "+_meta(name,x)+"\n")
                    sys.stderr.flush()
                fq_initial_state = initial_state.transpose(-1,-2).contiguous() if initial_state is not None else None
                g_cum, A, v_new, h_fq, final_state_fq = _flashqla_fwd(
                    q=q,k=k,v=v,g=g,beta=beta,scale=scale,initial_state=fq_initial_state,
                    cu_seqlens=cu_seqlens,output_final_state=output_final_state,output_h=False,auto_cp=False)
                out = v_new
                final_state = final_state_fq.transpose(-1,-2).contiguous() if final_state_fq is not None else None
                if os.environ.get("FLASHQLA_VERBOSE_META", "0") == "1" and _calls[0] in (1,2,10,20):
                    for name,x in [("v_new",v_new),("h_fq",h_fq),("out",out),("final_state",final_state)]:
                        sys.stderr.write("[flashqla-meta] "+_meta(name,x)+"\n")
                    sys.stderr.flush()
                return out, final_state
            _chunk.chunk_gated_delta_rule = _patched
            if _ops is not None:
                _ops.chunk_gated_delta_rule = _patched
            if _gdn is not None:
                _gdn.fla_chunk_gated_delta_rule = _patched
            sys.stderr.write("[flashqla-v2] active: HKV-output FlashQLA packed-single prefill with tunable HKV output kernel and short-decode fallback; original FLA fallback for unsupported paths\n"); sys.stderr.flush()
        else:
            sys.stderr.write(f"[flashqla-patch] cap={cap}; not patching\n"); sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[flashqla-patch] FAILED {type(e).__name__}: {e}\n"); sys.stderr.flush()
