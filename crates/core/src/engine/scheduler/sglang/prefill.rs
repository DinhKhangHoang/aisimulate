// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use super::super::AdmissionEvent;
use super::config::{SglangConfig, ceil_to_block};
use super::request::{SglangRequest, WaitingQueue};
use crate::engine::CacheTierAttribution;
use crate::engine::kv_manager::SglangKvManager;
use crate::engine::kv_manager::sglang_backend::{
    AdmittedPrefix, HostLoad, LoadBackGate, RequestAdmission,
};

/// Per-request prefill data needed for FPM snapshot construction.
pub(super) struct PrefillFpmItem {
    pub(super) prompt_len: usize,
    pub(super) tokens_computed: usize,
    /// KV context already present when this chunk runs (cached prefix or the request's own
    /// earlier chunks); drives the forward-pass model.
    pub(super) prefix_tokens: usize,
    /// Radix-cache reuse credited to this pass: the admission match on a request's first chunk,
    /// zero for continuations, which only extend their own tokens.
    pub(super) cache_reused_tokens: usize,
}

#[derive(Default)]
pub(super) struct AdmitResult {
    pub(super) can_run: Vec<SglangRequest>,
    pub(super) admissions: Vec<AdmissionEvent>,
    pub(super) total_isl: usize,
    pub(super) total_prefix: usize,
    /// Per-request prefill info for building FPM snapshots.
    pub(super) prefill_fpm: Vec<PrefillFpmItem>,
    /// SGLang HiCache load-backs started for this batch. The scheduler times
    /// their H2D copy and finishes them once the batch's pass completes.
    pub(super) host_loads: Vec<HostLoad>,
}

/// Behavioral model of SGLang's prefill admission (`Scheduler.get_new_batch_prefill`,
/// `PrefillAdder.add_one_req` / `add_chunked_req`, `Req.init_next_round_input`) as of
/// sgl-project/sglang v0.5.6.post2 (`5c8bd8b5`, `python/sglang/srt/managers/scheduler.py`,
/// `schedule_batch.py`, `schedule_policy.py`). Re-implemented; no SGLang source is copied.
pub(super) fn get_new_batch_prefill(
    waiting: &mut WaitingQueue,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    new_token_ratio: f64,
    running: &[SglangRequest],
) -> AdmitResult {
    let cache = kv_manager.cache();
    let reserved_decode_output: f64 = running
        .iter()
        .map(|req| {
            let remaining_output = req
                .remaining_output_tokens()
                .min(config.clip_max_new_tokens);
            remaining_output as f64 * new_token_ratio
        })
        .sum();
    // PagePool already removes the full physical footprint of every active
    // partial page from available capacity, so page slack must not be charged
    // a second time here.
    let mut rem_total_tokens =
        (cache.available_tokens() + cache.evictable_size()) as f64 - reserved_decode_output;
    let mut rem_input_tokens = config.max_prefill_tokens as f64;
    let mut rem_chunk_tokens = config.chunked_prefill_size as f64;

    let mut can_run = Vec::new();
    let mut admissions = Vec::new();
    let mut prefill_fpm = Vec::new();
    let mut rejected = VecDeque::new();
    let mut total_isl = 0usize;
    let mut total_prefix = 0usize;
    let mut host_loads = Vec::new();

    let available_running_slots = config.max_running_requests.saturating_sub(running.len());
    while can_run.len() < available_running_slots
        && let Some(mut req) = waiting.pop_front()
    {
        // SGLang `Req.init_next_round_input(tree_cache)`: a waiting request matches its whole
        // prompt against the radix tree once, capped at `input_len - 1` so a request that must
        // produce a token always computes at least one. Chunked continuations are re-initialised
        // without the tree cache and never re-match; the request simply continues from the tokens
        // it already owns. Zero-output (prefix-only) replay requests keep the full match: they
        // have no first token to sample, so a fully cached prompt is no forward-pass work.
        let cached_prefix = if req.materialized_tokens == 0 && config.enable_prefix_caching {
            req.kv_lease
                .ensure_page_hashes(&req.sequence_tokens, config.block_size);
            let match_len = if req.max_output_tokens == 0 {
                req.current_sequence_len()
            } else {
                req.current_sequence_len().saturating_sub(1)
            };
            let page_hashes = req.kv_lease.page_hashes();
            let pages = (match_len / config.block_size).min(page_hashes.len());
            if kv_manager.is_write_back() {
                // HiCache: the host hit that admission loads back is prefix
                // too, so the chunk and budgets below cover only the tokens
                // computed after it (`PrefillAdder.add_one_req` recomputes
                // `extend_input_len` after `init_load_back`).
                kv_manager.reusable_prefix_with_host(&page_hashes[..pages])
            } else {
                kv_manager
                    .cache()
                    .prefix_match_hashes_len(&page_hashes[..pages])
            }
        } else {
            0
        };
        let start = req.materialized_tokens.max(cached_prefix);
        // SGLang `extend_input_len`: tokens left to compute beyond the matched/owned prefix.
        let extend_input = req.current_sequence_len().saturating_sub(start);
        if extend_input == 0 && req.materialized_tokens > 0 {
            rejected.push_back(req);
            break;
        }

        let Some(chunk_tokens) = chunk_for(extend_input, rem_chunk_tokens, config.block_size)
        else {
            rejected.push_back(req);
            break;
        };

        let (charged_input_tokens, output_reserve) =
            prefill_charge(&req, chunk_tokens, extend_input, config);
        if charged_input_tokens + output_reserve as f64 >= rem_total_tokens {
            rejected.push_back(req);
            break;
        }
        if charged_input_tokens > rem_input_tokens || charged_input_tokens > rem_chunk_tokens {
            rejected.push_back(req);
            break;
        }

        let mut chunk_end = start + chunk_tokens;
        let first_admission = req.materialized_tokens == 0;
        let mut lease = std::mem::take(&mut req.kv_lease);
        let alloc_tokens = req.sequence_prefix(chunk_end);

        let admitted = if req.materialized_tokens > 0 {
            if !lease.is_active() {
                panic!(
                    "prefill: request {} has materialized_tokens={} but no active KV lease",
                    req.uuid, req.materialized_tokens
                );
            }
            // Continuation: extend the request's own pages; no re-match.
            kv_manager
                .extend_allocation(alloc_tokens, &mut lease)
                .map_or(RequestAdmission::NoRoom { load: None }, |prefix_len| {
                    RequestAdmission::Admitted(AdmittedPrefix {
                        prefix_len,
                        load: None,
                    })
                })
        } else {
            // Locks the path the match above saw, matching no further than that cap so
            // physical ownership and `cached_tokens` also stop at `input_len - 1`: the last token
            // of a fully cached prompt is computed and gets its own KV slot.
            //
            // Under HiCache write_back a host hit is loaded back here. Aggregated prefill never
            // gates on a free load slot: SGLang starts the loads of one prefill batch together
            // and the forward pass waits for them layer by layer, so the in-flight cap
            // (`LayerDoneCounter`) only bounds batches, which prefill already serializes.
            //
            // If the load is refused after all (another in-flight load pins part of the hit, or
            // no device room), `prefix_len` falls short of `start`. As SGLang 0.5.18
            // `add_one_req` recomputes `input_tokens` after `init_load_back`, the chunk is sized
            // again from the admitted prefix: the whole remaining input if it fits the chunk
            // budget, else truncated to it. SGLang re-checks nothing else here with chunked
            // prefill on, so the request is never rejected after a load: the earlier chunk check
            // left at least one page of chunk budget. The budgets below charge this chunk.
            let sequence_len = req.current_sequence_len();
            let fit = |prefix_len: usize| {
                let extend = sequence_len - prefix_len;
                let chunk = chunk_for(extend, rem_chunk_tokens, config.block_size)
                    .expect("the chunk check above left a page of chunk budget");
                chunk_end = prefix_len + chunk;
                chunk_end
            };
            kv_manager.admit_request_lease(
                alloc_tokens,
                &mut lease,
                start,
                LoadBackGate::Allow,
                fit,
            )
        };

        let (prefix_len, loaded_tokens) = match admitted {
            RequestAdmission::Admitted(AdmittedPrefix { prefix_len, load }) => {
                let loaded_tokens = load.as_ref().map_or(0, |load| load.tokens);
                if let Some(load) = load {
                    host_loads.push(*load);
                }
                (prefix_len, loaded_tokens)
            }
            // `NoRoom` is the old allocation failure; a load it started still runs, so the
            // scheduler times it with this batch. `WaitForLoadSlot` keeps the request waiting the
            // same way; it cannot occur under `LoadBackGate::Allow`.
            RequestAdmission::NoRoom { load } => {
                if let Some(load) = load {
                    host_loads.push(*load);
                }
                req.kv_lease = lease;
                rejected.push_back(req);
                break;
            }
            RequestAdmission::WaitForLoadSlot => {
                req.kv_lease = lease;
                rejected.push_back(req);
                break;
            }
        };
        let tokens_computed = chunk_end.saturating_sub(prefix_len);
        // Charge what admission settled on: a refused load-back computes more than estimated.
        let (charged_input_tokens, output_reserve) = if first_admission && prefix_len != start {
            prefill_charge(
                &req,
                tokens_computed,
                req.current_sequence_len() - prefix_len,
                config,
            )
        } else {
            (charged_input_tokens, output_reserve)
        };
        // Cache reuse is what the radix tree supplied at admission; a continuation's `prefix_len`
        // is the request's own earlier chunks and must not be reported as a cache hit.
        let admission_reused_tokens = lease.admission_reused_tokens();
        let cache_reused_tokens = if first_admission { prefix_len } else { 0 };

        req.kv_lease = lease;
        req.materialized_tokens = chunk_end;
        req.allocated_tokens = ceil_to_block(chunk_end, config.block_size);
        req.debug_assert_invariants(config.block_size);

        // One admission per request: a continuation chunk is the same admission still running,
        // not a readmission.
        if first_admission {
            // Cache-tier provenance of `reused_input_tokens`, under HiCache write_back only, as
            // vLLM's native host tier reports it (`VllmHostOffloadAdapter::on_admitted`):
            // - `g1` is the prefix already on device at admission and `host` the prefix loaded
            //   back from host by this admission; they sum to `reused_input_tokens`, the cached
            //   prompt tokens (capped at `input_len - 1`), loaded tokens included.
            // - The denominator is therefore the request's reuse, not its prompt length: a
            //   host-hit rate is `sum(host) / sum(prompt tokens)` over requests, or
            //   `sum(host) / sum(g1 + host)` for the host share of hits.
            // - Only a request's first admission reports it; a readmission after retraction
            //   reports `None`. SGLang 0.5.18 likewise computes `cached_tokens_device` /
            //   `cached_tokens_host` (`meta_info.cached_tokens_details`) once, on the first
            //   chunk, from `prefix_indices` and `host_hit_length`, and skips retracted requests.
            // - Without write_back there is no host tier and the attribution stays `None`.
            let cache_tier_attribution = (kv_manager.is_write_back()
                && !std::mem::replace(&mut req.first_admission_recorded, true))
            .then(|| CacheTierAttribution {
                g1_reused_input_tokens: admission_reused_tokens - loaded_tokens,
                host_reused_input_tokens: loaded_tokens,
            });
            admissions.push(AdmissionEvent {
                uuid: req.uuid,
                reused_input_tokens: admission_reused_tokens,
                cache_tier_attribution,
            });
        }
        prefill_fpm.push(PrefillFpmItem {
            prompt_len: req.prompt_len(),
            tokens_computed,
            prefix_tokens: prefix_len,
            cache_reused_tokens,
        });

        total_isl += chunk_end;
        total_prefix += prefix_len;
        rem_total_tokens -= charged_input_tokens + output_reserve as f64;
        rem_input_tokens -= charged_input_tokens;
        rem_chunk_tokens -= charged_input_tokens;
        can_run.push(req);

        if rem_chunk_tokens <= 0.0 {
            break;
        }
    }

    while let Some(req) = rejected.pop_back() {
        waiting.push_front(req);
    }

    AdmitResult {
        can_run,
        admissions,
        total_isl,
        total_prefix,
        prefill_fpm,
        host_loads,
    }
}

/// SGLang `add_one_req` chunk sizing: the whole `extend_input` if its paged size fits the chunk
/// budget, else the budget rounded down to whole pages. `None` if not even a page fits.
fn chunk_for(extend_input: usize, rem_chunk_tokens: f64, block_size: usize) -> Option<usize> {
    if ceil_to_block(extend_input, block_size) as f64 <= rem_chunk_tokens {
        return Some(extend_input);
    }
    let chunk = (rem_chunk_tokens as usize / block_size) * block_size;
    (chunk > 0).then(|| chunk.min(extend_input))
}

/// Budget charge of a chunk of `chunk_tokens` out of `extend_input` tokens left to compute:
/// the paged input tokens and, for a final chunk, the clipped output reserve. Budgets are charged
/// for computed tokens only (`PrefillAdder._update_prefill_budget`), not for the cached prefix.
fn prefill_charge(
    req: &SglangRequest,
    chunk_tokens: usize,
    extend_input: usize,
    config: &SglangConfig,
) -> (f64, usize) {
    let charged_input_tokens = ceil_to_block(chunk_tokens, config.block_size) as f64;
    let output_reserve = if chunk_tokens < extend_input {
        0
    } else {
        req.remaining_output_tokens()
            .min(config.clip_max_new_tokens)
    };
    (charged_input_tokens, output_reserve)
}
