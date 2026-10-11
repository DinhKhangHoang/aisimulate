// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-DP-rank timing of SGLang HiCache write_back host transfers.
//!
//! SGLang copies KV between device and host on two independent streams:
//!
//! - **D2H at eviction is synchronous.** Under write_back, `evict` waits in
//!   `writing_check(write_back=True)` until the copy of the demoted pages is
//!   done, so the scheduler step that evicted cannot start its forward pass
//!   before then. The clock records the copy's completion as a
//!   compute-not-before time for the rank.
//! - **H2D at load-back overlaps the prefill.** The loads of one prefill batch
//!   are started together before the forward pass, which waits for them layer
//!   by layer.
//!
//! Each direction is one FIFO lane per rank. A transfer becomes ready
//! `latency_to_first_byte_ms` after it is issued and then queues behind the
//! lane's earlier transfers, as in the vLLM host tier.

use crate::engine::NativeHostOffloadConfig;
use crate::engine::offload_transfer::FifoLane;

pub(super) struct HostTransferClock {
    d2h: FifoLane,
    h2d: FifoLane,
    /// `kv_cache_bytes_per_token x page_size`: one SGLang page is one
    /// AISimulate block.
    page_bytes: usize,
    latency_to_first_byte_ms: f64,
    /// Completion of the latest synchronous D2H copy. The next forward pass
    /// of this rank cannot start earlier.
    compute_not_before_ms: f64,
    /// Pages copied in each direction since construction.
    d2h_pages: usize,
    h2d_pages: usize,
}

impl HostTransferClock {
    pub(super) fn new(
        config: &NativeHostOffloadConfig,
        page_size: usize,
        kv_cache_bytes_per_token: usize,
    ) -> Self {
        Self {
            d2h: FifoLane::new(config.d2h_bandwidth_gbps),
            h2d: FifoLane::new(config.h2d_bandwidth_gbps),
            page_bytes: page_size.saturating_mul(kv_cache_bytes_per_token),
            latency_to_first_byte_ms: config.latency_to_first_byte_ms,
            compute_not_before_ms: 0.0,
            d2h_pages: 0,
            h2d_pages: 0,
        }
    }

    /// Charge the synchronous write_back copy of `pages` demoted at `now_ms`.
    pub(super) fn write_back(&mut self, now_ms: f64, pages: usize) {
        if pages == 0 {
            return;
        }
        self.d2h_pages += pages;
        let done_ms = self.d2h.submit(
            now_ms + self.latency_to_first_byte_ms,
            pages.saturating_mul(self.page_bytes),
        );
        self.compute_not_before_ms = self.compute_not_before_ms.max(done_ms);
    }

    /// The earliest start of a forward pass at `now_ms`: after every D2H copy
    /// charged so far.
    pub(super) fn compute_start(&self, now_ms: f64) -> f64 {
        now_ms.max(self.compute_not_before_ms)
    }

    /// Start the H2D copy of a prefill batch's `pages` at `not_before_ms`;
    /// returns its completion time.
    pub(super) fn load(&mut self, not_before_ms: f64, pages: usize) -> f64 {
        self.h2d_pages += pages;
        self.h2d.submit(
            not_before_ms + self.latency_to_first_byte_ms,
            pages.saturating_mul(self.page_bytes),
        )
    }

    /// (D2H, H2D) pages copied so far.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn pages(&self) -> (usize, usize) {
        (self.d2h_pages, self.h2d_pages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(d2h: f64, h2d: f64, ttfb: f64) -> HostTransferClock {
        let mut config = NativeHostOffloadConfig::new(8);
        config.d2h_bandwidth_gbps = d2h;
        config.h2d_bandwidth_gbps = h2d;
        config.latency_to_first_byte_ms = ttfb;
        // 1 MB pages.
        HostTransferClock::new(&config, 4, 250_000)
    }

    #[test]
    fn write_back_sets_compute_not_before_and_queues_fifo() {
        let mut clock = clock(1.0, 1.0, 0.5);
        assert_eq!(clock.compute_start(3.0), 3.0);
        clock.write_back(10.0, 0);
        assert_eq!(clock.compute_start(3.0), 3.0);
        // Ready at 10.5, 2 MB at 1 GB/s = 2 ms.
        clock.write_back(10.0, 2);
        assert_eq!(clock.compute_start(10.0), 12.5);
        // Ready at 11.5 but queued behind the first copy.
        clock.write_back(11.0, 1);
        assert_eq!(clock.compute_start(10.0), 13.5);
        assert_eq!(clock.compute_start(20.0), 20.0);
        assert_eq!(clock.pages(), (3, 0));
    }

    /// TTFB overlaps queueing: a transfer is ready `ttfb` after it is issued
    /// and then waits for the lane, so behind a busy lane its TTFB is hidden.
    /// It does not occupy the lane per transfer, which would serialize it.
    #[test]
    fn ttfb_overlaps_a_busy_lane_instead_of_occupying_it() {
        let mut clock = clock(1.0, 1.0, 1.0);
        // Ready at 1, 4 MB at 1 GB/s = 4 ms.
        assert_eq!(clock.load(0.0, 4), 5.0);
        // Ready at 1 but the lane is busy until 5: 6, not 7.
        assert_eq!(clock.load(0.0, 1), 6.0);
        // Issued at 3 and ready at 4, still behind the lane: 7, not 8.
        assert_eq!(clock.load(3.0, 1), 7.0);
        // An idle lane pays the full TTFB again.
        assert_eq!(clock.load(10.0, 1), 12.0);
        // The D2H lane follows the same rule.
        clock.write_back(0.0, 4);
        clock.write_back(0.0, 1);
        assert_eq!(clock.compute_start(0.0), 6.0);
        assert_eq!(clock.pages(), (5, 7));
    }

    #[test]
    fn directions_are_independent_and_zero_bandwidth_is_unlimited() {
        let mut clock = clock(1.0, 0.0, 0.0);
        clock.write_back(0.0, 4);
        assert_eq!(clock.load(0.0, 4), 0.0);
        assert_eq!(clock.compute_start(0.0), 4.0);
        assert_eq!(clock.pages(), (4, 4));
    }
}
