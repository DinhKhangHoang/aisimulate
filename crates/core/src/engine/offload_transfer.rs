// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Equal-share byte service shared by offload tiers.
//!
//! Each tier owns its own instance and budgets. Callers resolve bandwidth,
//! first-byte readiness and job IDs; this module only advances bytes through
//! virtual time. A moving job's rate in its direction is
//! `min(client_rate / active_client_jobs, shared_rate / active_jobs)`. Jobs
//! waiting for their first byte consume no bandwidth, idle shares are not
//! redistributed, and admission, completion or cancellation reflows service.

use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Read,
    Write,
}

/// Decimal GB/s to bytes per millisecond. Zero removes the limit.
pub(crate) fn bytes_per_ms(gbps: f64) -> f64 {
    if gbps == 0.0 {
        f64::INFINITY
    } else {
        gbps * 1e6
    }
}

/// One directional transfer lane that serves transfers in submission order at a
/// fixed rate. A transfer starts at the later of its ready time and the
/// previous transfer's completion.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FifoLane {
    bytes_per_ms: f64,
    tail_ms: f64,
}

impl FifoLane {
    /// A lane at `gbps` decimal GB/s. Zero removes the limit.
    pub(crate) fn new(gbps: f64) -> Self {
        Self {
            bytes_per_ms: bytes_per_ms(gbps),
            tail_ms: 0.0,
        }
    }

    /// Queue `bytes` that become ready at `not_before_ms`; returns the completion time.
    pub(crate) fn submit(&mut self, not_before_ms: f64, bytes: usize) -> f64 {
        let starts_at_ms = self.tail_ms.max(not_before_ms);
        let duration_ms = if self.bytes_per_ms.is_infinite() {
            0.0
        } else {
            bytes as f64 / self.bytes_per_ms
        };
        self.tail_ms = starts_at_ms + duration_ms;
        self.tail_ms
    }
}

pub(crate) struct Job {
    pub(crate) id: u64,
    /// Budget group: jobs of one client share `client_rate`.
    pub(crate) client: u64,
    pub(crate) direction: Direction,
    /// First-byte time; the job moves no bytes before it.
    pub(crate) ready: f64,
    pub(crate) bytes: f64,
    pub(crate) client_rate: f64,
    pub(crate) shared_rate: f64,
}

/// Ready-job counts for one queue/time snapshot; never retained across events.
#[derive(Default)]
struct ReadyCounts {
    total: [usize; 2],
    clients: FxHashMap<u64, [usize; 2]>,
}

impl ReadyCounts {
    fn new<'a>(jobs: impl Iterator<Item = &'a Job>, now: f64) -> Self {
        let mut counts = Self::default();
        for job in jobs.filter(|job| job.ready <= now) {
            let direction = job.direction as usize;
            counts.total[direction] += 1;
            counts.clients.entry(job.client).or_default()[direction] += 1;
        }
        counts
    }

    fn rate(&self, job: &Job) -> f64 {
        let direction = job.direction as usize;
        let local = self.clients.get(&job.client).map_or(0, |n| n[direction]);
        let share = |rate: f64, n: usize| rate / n.max(1) as f64;
        share(job.client_rate, local).min(share(job.shared_rate, self.total[direction]))
    }
}

#[derive(Default)]
pub(crate) struct FairTransfers {
    now: f64,
    /// Submission order; completions in one step are reported in this order.
    jobs: Vec<Job>,
}

impl FairTransfers {
    pub(crate) fn now(&self) -> f64 {
        self.now
    }

    pub(crate) fn has_client_jobs(&self, client: u64) -> bool {
        self.jobs.iter().any(|job| job.client == client)
    }

    /// Caller must first advance to the submission time.
    pub(crate) fn submit(&mut self, job: Job) {
        debug_assert!(job.ready >= self.now);
        self.jobs.push(job);
    }

    pub(crate) fn cancel(&mut self, id: u64) -> bool {
        let before = self.jobs.len();
        self.jobs.retain(|job| job.id != id);
        self.jobs.len() != before
    }

    fn event_time(&self, job: &Job, counts: &ReadyCounts) -> f64 {
        if job.ready > self.now {
            job.ready
        } else {
            self.now + job.bytes / counts.rate(job)
        }
    }

    /// Earliest first-byte or completion event under the current job set.
    pub(crate) fn next_event(&self) -> Option<f64> {
        let counts = ReadyCounts::new(self.jobs.iter(), self.now);
        self.jobs
            .iter()
            .map(|job| self.event_time(job, &counts))
            .min_by(f64::total_cmp)
    }

    /// Move service to `target`, returning `(job id, completion time)` in
    /// event order and, within one event, submission order.
    pub(crate) fn advance(&mut self, target: f64) -> Vec<(u64, f64)> {
        let target = target.max(self.now);
        let mut completed = Vec::new();
        loop {
            let counts = ReadyCounts::new(self.jobs.iter(), self.now);
            let event = self
                .jobs
                .iter()
                .map(|job| self.event_time(job, &counts))
                .min_by(f64::total_cmp);
            let next = event.unwrap_or(target).min(target);
            let elapsed = next - self.now;
            let rates = self
                .jobs
                .iter()
                .map(|job| {
                    if job.ready <= self.now {
                        (counts.rate(job), self.event_time(job, &counts) <= next)
                    } else {
                        (0.0, false)
                    }
                })
                .collect::<Vec<_>>();
            for (job, (rate, completes)) in self.jobs.iter_mut().zip(rates) {
                // The selected completion boundary is authoritative. Repeated
                // byte subtraction at a large timestamp can leave a residual
                // whose duration is below one clock ULP and would never drain.
                job.bytes = if completes {
                    0.0
                } else {
                    (job.bytes - elapsed * rate).max(0.0)
                };
            }
            self.now = next;
            let before = completed.len();
            let now = self.now;
            self.jobs.retain(|job| {
                let done = job.ready <= now && job.bytes == 0.0;
                if done {
                    completed.push((job.id, now));
                }
                !done
            });
            if event.is_none_or(|time| time > target) {
                break;
            }
            if self.now == target && completed.len() == before {
                // Advancing time can make first-byte waiters eligible at this boundary.
                let counts = ReadyCounts::new(self.jobs.iter(), self.now);
                if self
                    .jobs
                    .iter()
                    .all(|job| self.event_time(job, &counts) > target)
                {
                    break;
                }
            }
        }
        completed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: f64 = 1_000_000.0;

    fn job(id: u64, client: u64, direction: Direction, ready: f64, rates: (f64, f64)) -> Job {
        Job {
            id,
            client,
            direction,
            ready,
            bytes: MB,
            client_rate: bytes_per_ms(rates.0),
            shared_rate: bytes_per_ms(rates.1),
        }
    }

    #[test]
    fn fifo_lane_serializes_transfers_after_their_ready_time() {
        let mut lane = FifoLane::new(1.0);
        // 1 MB at 1 GB/s takes 1 ms; an idle lane starts at the ready time.
        assert_eq!(lane.submit(2.0, 1_000_000), 3.0);
        // A transfer ready before the tail queues behind it.
        assert_eq!(lane.submit(0.0, 500_000), 3.5);
        // A transfer ready after the tail leaves the lane idle until then.
        assert_eq!(lane.submit(10.0, 1_000_000), 11.0);
    }

    #[test]
    fn unlimited_fifo_lane_completes_at_the_ready_time() {
        let mut lane = FifoLane::new(0.0);
        assert_eq!(lane.submit(4.0, usize::MAX), 4.0);
        assert_eq!(lane.submit(1.0, 1), 4.0);
    }

    fn run(jobs: Vec<Job>) -> Vec<(u64, f64)> {
        let mut transfers = FairTransfers::default();
        jobs.into_iter().for_each(|job| transfers.submit(job));
        transfers.advance(f64::MAX)
    }

    #[test]
    fn directions_have_independent_client_and_shared_budgets() {
        // G3 tier tests cover client and shared caps within one direction.
        let rates = (1.0, 1.0);
        assert_eq!(
            run(vec![
                job(0, 0, Direction::Write, 0.0, rates),
                job(1, 0, Direction::Read, 0.0, rates),
            ]),
            [(0, 1.0), (1, 1.0)]
        );
    }

    #[test]
    fn same_time_completions_follow_event_steps() {
        // Equal timestamps follow event steps: the moving job finishes in the
        // step that makes the instantaneous waiter ready. Tiers own any other
        // same-time order.
        assert_eq!(
            run(vec![
                job(0, 0, Direction::Write, 1.0, (0.0, 0.0)),
                job(1, 1, Direction::Write, 0.0, (1.0, 0.0)),
            ]),
            [(1, 1.0), (0, 1.0)]
        );
    }

    #[test]
    fn cancellation_reflows_the_remaining_bytes() {
        let mut transfers = FairTransfers::default();
        for id in 0..2 {
            transfers.submit(job(id, 0, Direction::Read, 0.0, (1.0, 0.0)));
        }
        assert_eq!(transfers.next_event(), Some(2.0));
        assert!(transfers.advance(1.0).is_empty());
        assert!(transfers.cancel(0) && !transfers.cancel(0));
        assert!(!transfers.has_client_jobs(1) && transfers.has_client_jobs(0));
        assert_eq!(transfers.next_event(), Some(1.5));
        assert_eq!(transfers.advance(10.0), [(1, 1.5)]);
        assert!(transfers.next_event().is_none() && transfers.now() == 10.0);
    }

    #[test]
    fn ready_counts_are_one_snapshot_pass() {
        use std::cell::Cell;
        let jobs = (0..64)
            .map(|id| {
                let direction = if id % 2 == 0 {
                    Direction::Read
                } else {
                    Direction::Write
                };
                job(
                    id,
                    id % 4,
                    direction,
                    if id < 32 { 0.0 } else { 1.0 },
                    (1.0, 0.0),
                )
            })
            .collect::<Vec<_>>();
        let visits = Cell::new(0);
        let counts = ReadyCounts::new(jobs.iter().inspect(|_| visits.set(visits.get() + 1)), 0.0);
        assert_eq!((visits.get(), counts.total), (64, [16, 16]));
        assert_eq!(counts.rate(&jobs[0]), MB / 8.0);
        assert_eq!(ReadyCounts::new(jobs.iter(), 1.0).rate(&jobs[0]), MB / 16.0);
    }
}
