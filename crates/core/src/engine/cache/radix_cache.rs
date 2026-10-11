// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang radix-cache adapter for engine simulation.
//!
//! Tree structure, matching, locking, and eviction are delegated to the
//! `sglang-radix-tree` crate. AISimulate continues to own physical page
//! allocation, request leases, event publication, and token accounting.

use std::borrow::{Borrow, Cow};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use crate::engine::belady::BeladyOracle;
use crate::engine::common::hashing::{SequenceHash, compute_next_seq_hash};

use sglang_radix_tree::{
    CacheAction, CacheInitParams, ChildKeyType, ComponentSet, DecLockRefParams, FULL, InsertParams,
    KeyNamespaceRef, MatchPrefixParams, PageValue, UnifiedTreeCore,
};

use crate::engine::common::hashing::LocalBlockHash;
#[cfg(test)]
use crate::engine::common::hashing::compute_block_hash_for_seq;

pub use sglang_radix_tree::NodeId;

/// Physical page identifier in the simulated SGLang KV pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KvPageId(usize);

impl KvPageId {
    pub(crate) fn index(self) -> usize {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn from_token_index(index: usize, page_size: usize) -> Self {
        Self(index / page_size)
    }

    #[cfg(test)]
    pub(crate) fn first_token_index(self, page_size: usize) -> usize {
        self.0 * page_size
    }
}

/// Manages free / allocated pages for the simulated SGLang KV cache.
///
/// SGLang's paged allocator owns and frees whole pages in production.
#[derive(Clone)]
pub struct PagePool {
    next_fresh: usize,
    free: Vec<KvPageId>,
    total_pages: usize,
    page_size: usize,
}

impl PagePool {
    pub fn new(total_tokens: usize, page_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        Self {
            next_fresh: 0,
            free: Vec::new(),
            total_pages: total_tokens / page_size,
            page_size,
        }
    }

    pub fn allocate_pages(&mut self, count: usize) -> Option<Vec<KvPageId>> {
        if self.available_pages() < count {
            return None;
        }

        let recycled = count.min(self.free.len());
        let fresh = count - recycled;
        let mut pages = Vec::with_capacity(count);
        pages.extend(self.free.drain(self.free.len() - recycled..));
        pages.extend((self.next_fresh..self.next_fresh + fresh).map(KvPageId));
        self.next_fresh += fresh;
        Some(pages)
    }

    #[cfg(test)]
    pub fn allocate(&mut self, token_count: usize) -> Option<Vec<usize>> {
        let mut indices = Vec::new();
        self.allocate_indices_into(token_count, &mut indices)
            .then_some(indices)
    }

    /// Append flattened indices for `new_tokens`, allocating whole pages only
    /// when the request's current final page has no remaining slots.
    #[cfg(test)]
    pub fn allocate_indices_into(&mut self, new_tokens: usize, indices: &mut Vec<usize>) -> bool {
        if new_tokens == 0 {
            return true;
        }

        let available_in_last_page = indices
            .last()
            .map_or(0, |last| self.page_size - 1 - (last % self.page_size));
        let tokens_requiring_pages = new_tokens.saturating_sub(available_in_last_page);
        let required_pages = tokens_requiring_pages.div_ceil(self.page_size);
        if self.available_pages() < required_pages {
            return false;
        }

        indices.reserve(new_tokens);
        let from_existing = new_tokens.min(available_in_last_page);
        if from_existing > 0 {
            let start = indices.last().copied().expect("last page must exist") + 1;
            indices.extend(start..start + from_existing);
        }

        let remaining = new_tokens - from_existing;
        let Some(pages) = self.allocate_pages(required_pages) else {
            return false;
        };
        for (page_idx, page) in pages.into_iter().enumerate() {
            let take = remaining
                .saturating_sub(page_idx * self.page_size)
                .min(self.page_size);
            let start = page.first_token_index(self.page_size);
            indices.extend(start..start + take);
        }
        true
    }

    pub fn free_pages(&mut self, pages: &[KvPageId]) {
        self.free.extend_from_slice(pages);
    }

    /// Free every distinct page represented by a contiguous request-index list.
    #[cfg(test)]
    pub fn free_indices(&mut self, indices: &[usize]) -> Vec<KvPageId> {
        let mut pages = Vec::with_capacity(indices.len().div_ceil(self.page_size));
        for &index in indices {
            let page = KvPageId::from_token_index(index, self.page_size);
            if pages.last().copied() != Some(page) {
                pages.push(page);
            }
        }
        self.free_pages(&pages);
        pages
    }

    #[cfg(test)]
    pub fn free(&mut self, indices: &[usize]) {
        self.free_indices(indices);
    }

    pub fn available_pages(&self) -> usize {
        self.free.len() + self.total_pages - self.next_fresh
    }

    pub fn available(&self) -> usize {
        self.available_pages() * self.page_size
    }

    pub fn total(&self) -> usize {
        self.total_pages * self.page_size
    }
}

/// Host page identifier in the simulated SGLang HiCache host pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HostPageId(usize);

impl HostPageId {
    pub(crate) fn index(self) -> usize {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn from_index(index: usize) -> Self {
        Self(index)
    }
}

/// Manages free / allocated pages for the simulated SGLang HiCache host pool.
///
/// The radix core stores these indices as a node's host value but never
/// allocates them; SGLang's host pool does, as this pool does here. Host
/// pages are whole device pages, so the pool counts pages, not tokens.
#[derive(Clone)]
pub struct HostPagePool {
    next_fresh: usize,
    free: Vec<HostPageId>,
    total_pages: usize,
}

impl HostPagePool {
    pub fn new(total_pages: usize) -> Self {
        Self {
            next_fresh: 0,
            free: Vec::new(),
            total_pages,
        }
    }

    pub fn allocate_pages(&mut self, count: usize) -> Option<Vec<HostPageId>> {
        if self.available_pages() < count {
            return None;
        }

        let recycled = count.min(self.free.len());
        let fresh = count - recycled;
        let mut pages = Vec::with_capacity(count);
        pages.extend(self.free.drain(self.free.len() - recycled..));
        pages.extend((self.next_fresh..self.next_fresh + fresh).map(HostPageId));
        self.next_fresh += fresh;
        Some(pages)
    }

    pub fn free_pages(&mut self, pages: &[HostPageId]) {
        for page in pages {
            assert!(
                page.0 < self.next_fresh,
                "host page {page:?} was never allocated"
            );
        }
        self.free.extend_from_slice(pages);
        assert!(
            self.free.len() <= self.next_fresh,
            "more host pages freed than allocated"
        );
    }

    pub fn available_pages(&self) -> usize {
        self.free.len() + self.total_pages - self.next_fresh
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }
}

/// One host-pool change made during eviction, in the order it happened.
///
/// The order matters: one `evict()` call can free a host page and then reuse
/// it for the next store, or store a page and then free it again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostPageEvent {
    /// The KV of `device` was copied to `host`. The device page is still
    /// allocated at that point, so its identity can be copied to the host
    /// page; it is freed later in the same call and listed in
    /// [`EvictionOutcome::device_pages`].
    Stored { host: HostPageId, device: KvPageId },
    /// The host page went back to the host pool.
    Freed(HostPageId),
}

/// What one `RadixCache::evict` call released.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EvictionOutcome {
    /// Tokens released from the device pool (`device_pages.len() * page_size`).
    pub evicted_tokens: usize,
    /// Device pages returned to the page pool, demoted pages included.
    pub device_pages: Vec<KvPageId>,
    /// Host-pool changes in order; always empty without write_back.
    pub host_events: Vec<HostPageEvent>,
}

impl EvictionOutcome {
    /// Host pages stored by write_back demotes, with their source device page.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stored_host(&self) -> impl Iterator<Item = (HostPageId, KvPageId)> + '_ {
        self.host_events.iter().filter_map(|event| match *event {
            HostPageEvent::Stored { host, device } => Some((host, device)),
            HostPageEvent::Freed(_) => None,
        })
    }

    /// Host pages returned to the host pool.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn host_frees(&self) -> impl Iterator<Item = HostPageId> + '_ {
        self.host_events.iter().filter_map(|event| match *event {
            HostPageEvent::Freed(host) => Some(host),
            HostPageEvent::Stored { .. } => None,
        })
    }
}

/// Smallest Full-KV load-back SGLang performs, in tokens: shorter host hits
/// are recomputed instead (`UnifiedRadixCache.load_back_threshold`, URC:515,
/// applied as `max(1, threshold)` in `_load_back_transfers`).
pub(crate) const LOAD_BACK_THRESHOLD_TOKENS: usize = 10;

/// Load-backs SGLang keeps in flight at once. `has_free_load_back_slot`
/// (URC:3545-3549) compares the load ack queue with
/// `LayerDoneCounter.num_counters`, fixed at 3 ("extra producer and consumer
/// counters for overlap mode", `LayerDoneCounter.__init__` in
/// managers/cache_controller.py). A fourth load waits for a slot rather than
/// falling back to recompute.
///
/// Not enforced yet: prefill admits with `LoadBackGate::Allow`, every load
/// finishes at its pass's engine boundary, and one prefill batch may start
/// more than three loads. Kept, with `LoadBackGate::NoFreeSlot`, for a future
/// gate (for example on disaggregated decode, which does not load back today).
#[allow(dead_code)]
pub(crate) const MAX_LOAD_BACK_IN_FLIGHT: usize = 3;

/// A prefix match that also reports the host-only continuation of the
/// device prefix (SGLang `MatchResult` under HiCache, URC:552-568).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefixMatch {
    /// Device-resident prefix, in tokens.
    pub(crate) device_tokens: usize,
    /// Deepest device node on the prefix. It carries the request lock.
    pub(crate) last_node: NodeId,
    /// The host-only prefix after `device_tokens`, if any.
    pub(crate) host_hit: Option<HostHit>,
}

impl PrefixMatch {
    /// Host-only prefix after the device prefix, in tokens.
    pub(crate) fn host_tokens(&self) -> usize {
        self.host_hit.map_or(0, |hit| hit.tokens)
    }
}

/// Opaque load-back anchor from [`RadixCache::match_prefix_with_host_and_lock`]:
/// SGLang's `best_match_node` plus the device node the request locked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostHit {
    tokens: usize,
    best_node: NodeId,
    request_node: NodeId,
}

impl HostHit {
    /// Host-only prefix tokens (SGLang `host_hit_length` times page size).
    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Why [`RadixCache::load_back`] declined; the request recomputes the host hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoadBackRefusal {
    /// The anchor is gone or already back on device; match again.
    Stale,
    /// Fewer than [`LOAD_BACK_THRESHOLD_TOKENS`] tokens to load.
    BelowThreshold,
    /// The load plus the newly protected prefix exceeds the caller's quota.
    OverQuota,
    /// Eviction could not free enough device pages.
    NoDeviceRoom,
}

/// Result of [`RadixCache::load_back`].
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum LoadBackResult {
    /// Boxed: the lock receipts make it much larger than a refusal.
    Loaded(Box<LoadedBack>),
    Refused {
        reason: LoadBackRefusal,
        /// Write_back eviction done before the refusal. Its pages have moved
        /// even though nothing was loaded, so the caller must publish it.
        eviction: EvictionOutcome,
    },
}

/// A committed H->D load-back.
#[derive(Debug)]
pub(crate) struct LoadedBack {
    /// Each loaded device page with the host page it is copied from, in
    /// prefix order (ancestors first).
    pub(crate) pages: Vec<(HostPageId, KvPageId)>,
    /// Loaded tokens (`pages.len() * page_size`).
    pub(crate) tokens: usize,
    /// The loaded node. The request lock has moved here from the matched
    /// device node; the request prefix is now `device_tokens + tokens` long.
    pub(crate) last_node: NodeId,
    /// Write_back eviction done to make device room; may be empty.
    pub(crate) eviction: EvictionOutcome,
    /// Pass to [`RadixCache::finish_load_back`] when the H2D copy completes.
    pub(crate) handle: LoadBackHandle,
}

/// An in-flight load-back (SGLang `_OngoingLoadBack`, URC:1838-1842): the
/// device and host lock receipts on the anchor, released by
/// [`RadixCache::finish_load_back`].
#[must_use = "an unfinished load-back keeps its node locked and pinned"]
pub(crate) struct LoadBackHandle {
    anchor: NodeId,
    lock: DecLockRefParams,
    host_lock: DecLockRefParams,
}

impl LoadBackHandle {
    /// The loaded node this load-back is anchored on.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn node(&self) -> NodeId {
        self.anchor
    }
}

impl std::fmt::Debug for LoadBackHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadBackHandle")
            .field("anchor", &self.anchor)
            .finish_non_exhaustive()
    }
}

pub(crate) struct InsertPageResult {
    pub(crate) last_node: NodeId,
    pub(crate) canonical_suffix: Vec<KvPageId>,
    pub(crate) unretained_pages: Vec<KvPageId>,
    /// Host pages the insert released, already returned to the host pool. A
    /// Full-only tree never produces them; see `collect_unretained_pages`.
    pub(crate) unretained_host_pages: Vec<HostPageId>,
}

/// Shared immutable keys keep admission checkpoints proportional to node metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct PageKey(Arc<[LocalBlockHash]>);

impl From<Vec<LocalBlockHash>> for PageKey {
    fn from(value: Vec<LocalBlockHash>) -> Self {
        Self(value.into())
    }
}
impl AsRef<[LocalBlockHash]> for PageKey {
    fn as_ref(&self) -> &[LocalBlockHash] {
        &self.0
    }
}
impl Borrow<[LocalBlockHash]> for PageKey {
    fn borrow(&self) -> &[LocalBlockHash] {
        &self.0
    }
}
impl ChildKeyType for PageKey {
    type Atom = LocalBlockHash;
    const IS_BIGRAM: bool = false;
    fn key_from(ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self> {
        Cow::Owned(
            ids.iter()
                .map(|&value| LocalBlockHash(value as u64))
                .collect::<Vec<_>>()
                .into(),
        )
    }
    fn hash_words(atom: &LocalBlockHash) -> impl Iterator<Item = u32> {
        [atom.0 as u32, (atom.0 >> 32) as u32].into_iter()
    }
    fn raw_token_ids(atoms: &[LocalBlockHash]) -> Cow<'_, [i64]> {
        Cow::Owned(atoms.iter().map(|atom| atom.0 as i64).collect())
    }
}

impl From<KvPageId> for i64 {
    fn from(page: KvPageId) -> Self {
        i64::try_from(page.0).expect("page ID fits in i64")
    }
}

type SglangTree = UnifiedTreeCore<PageKey, PageValue<KvPageId>>;

/// The core stores host and device indices in the same value type.
fn host_value(pages: &[HostPageId]) -> PageValue<KvPageId> {
    PageValue::from_vec(pages.iter().map(|page| KvPageId(page.0)).collect())
}

fn host_pages_of(value: &PageValue<KvPageId>) -> impl Iterator<Item = HostPageId> + '_ {
    value.as_slice().iter().map(|page| HostPageId(page.0))
}

#[derive(Clone)]
struct BeladyPages {
    oracle: BeladyOracle,
    page_hashes: Vec<Option<SequenceHash>>,
}

/// Thin adapter from AISimulate's page-native cache contract to SGLang's tree.
pub struct RadixCache {
    tree: SglangTree,
    belady: Option<BeladyPages>,
    pub page_pool: PagePool,
    /// Present only in SGLang HiCache write_back mode.
    host_pool: Option<HostPagePool>,
    page_size: usize,
}

impl RadixCache {
    pub(crate) fn admission_checkpoint(&self) -> Self {
        Self {
            tree: self.tree.snapshot_full_device(),
            belady: self.belady.clone(),
            page_pool: self.page_pool.clone(),
            host_pool: self.host_pool.clone(),
            page_size: self.page_size,
        }
    }

    pub(crate) fn set_belady_oracle(&mut self, oracle: BeladyOracle) {
        assert!(
            self.host_pool.is_none(),
            "Belady eviction is not modelled with SGLang HiCache write_back"
        );
        assert_eq!(
            self.tree.evictable_size() + self.tree.protected_size(),
            0,
            "attach Belady before populating the cache"
        );
        self.belady = Some(BeladyPages {
            oracle,
            page_hashes: vec![None; self.page_pool.total_pages],
        });
    }

    fn record_belady_pages(
        &mut self,
        prefix_node: NodeId,
        keys: &[LocalBlockHash],
        pages: &[KvPageId],
    ) {
        let Some(belady) = self.belady.as_mut() else {
            return;
        };
        let prefix = self
            .tree
            .get_component_device_value(prefix_node, FULL)
            .expect("live prefix");
        let mut previous = prefix
            .as_ref()
            .and_then(|values| values.as_slice().last())
            .map(|page| belady.page_hashes[page.index()].expect("cached page has prefix identity"));
        for (&key, &page) in keys.iter().zip(pages) {
            let hash = previous.map_or(key.0, |parent| compute_next_seq_hash(parent, key));
            belady.page_hashes[page.index()] = Some(hash);
            previous = Some(hash);
        }
    }

    fn evict_belady(&mut self, target_pages: usize) -> Vec<KvPageId> {
        let belady = self.belady.as_ref().expect("Belady configured");
        let priority =
            |candidate: sglang_radix_tree::FullDeviceEvictionCandidate<'_, PageValue<KvPageId>>| {
                let page = candidate.value.as_slice().last().expect("nonempty leaf");
                let hash =
                    belady.page_hashes[page.index()].expect("cached page has prefix identity");
                Reverse((
                    Reverse(belady.oracle.next_use(hash)),
                    candidate.last_access_counter,
                    candidate.node_id,
                ))
            };
        // Start from the core's current unlocked leaves. Global forecast changes
        // are observed at each eviction; they never create or unlock local KV.
        let mut candidates: BinaryHeap<_> = self
            .tree
            .full_device_eviction_candidates()
            .map(priority)
            .collect();
        let mut pages = Vec::with_capacity(target_pages.min(self.tree.evictable_size()));
        while pages.len() < target_pages {
            let Some(Reverse((_, _, node_id))) = candidates.pop() else {
                break;
            };
            let (parent, step) = self
                .tree
                .evict_full_device_suffix(node_id, 1)
                .expect("live eviction candidate");
            for chunk in step.device_frees.get(&FULL).into_iter().flatten() {
                pages.extend_from_slice(chunk.as_slice());
            }
            if let Some(parent) = parent.and_then(|id| self.tree.full_device_eviction_candidate(id))
            {
                candidates.push(priority(parent));
            }
        }
        let belady = self.belady.as_mut().expect("Belady configured");
        for page in &pages {
            belady.page_hashes[page.index()] = None;
        }
        pages
    }
    pub fn new(total_tokens: usize, page_size: usize) -> Self {
        Self::build(total_tokens, page_size, None)
    }

    /// Build a cache that offloads evicted device KV to a host pool of
    /// `host_pages` pages, following SGLang HiCache's write_back policy:
    /// a device leaf is copied to host when it is evicted, then demoted.
    pub fn new_write_back(total_tokens: usize, page_size: usize, host_pages: usize) -> Self {
        Self::build(total_tokens, page_size, Some(HostPagePool::new(host_pages)))
    }

    fn build(total_tokens: usize, page_size: usize, host_pool: Option<HostPagePool>) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        let write_back = host_pool.is_some();
        let tree = SglangTree::new(
            CacheInitParams {
                // A request key already contains one hash per physical KV page.
                page_size: 1,
                eviction_policy: "lru".to_string(),
                // write_back never fires write-through backups, so the
                // threshold is unused. Core events stay off: the KV manager
                // publishes events in AISimulate's own block-hash space.
                is_write_back: write_back,
                enable_hicache: write_back,
                enable_kv_cache_events: false,
                ..Default::default()
            },
            vec![FULL],
        );
        Self {
            tree,
            belady: None,
            page_pool: PagePool::new(total_tokens, page_size),
            host_pool,
            page_size,
        }
    }

    pub(crate) fn host_pool(&self) -> Option<&HostPagePool> {
        self.host_pool.as_ref()
    }

    pub fn root(&self) -> NodeId {
        self.tree.root_node_handle(None)
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    #[cfg(test)]
    pub fn num_nodes(&self) -> usize {
        self.tree.inspect_get_all_node_ids().len()
    }

    #[cfg(test)]
    pub(crate) fn node(&self, id: NodeId) -> TreeNodeSnapshot {
        TreeNodeSnapshot {
            parent: self.tree.inspect_get_parent_node_id(id).unwrap(),
            children: self
                .tree
                .inspect_get_child_node_ids(id)
                .unwrap()
                .into_iter()
                .map(|child| {
                    let first = self.tree.inspect_get_node_token_ids(child).unwrap()[0];
                    (LocalBlockHash(first as u64), child)
                })
                .collect(),
            lock_ref: self
                .tree
                .inspect_get_component_device_lock_ref(id, FULL)
                .unwrap(),
            last_access_counter: self.tree.inspect_get_node_access_counter(id).unwrap(),
            key: self
                .tree
                .inspect_get_node_token_ids(id)
                .unwrap()
                .into_iter()
                .map(|value| LocalBlockHash(value as u64))
                .collect(),
            value: self
                .tree
                .get_component_device_value(id, FULL)
                .unwrap()
                .map_or_else(Vec::new, |value| value.as_slice().to_vec()),
        }
    }

    #[cfg(test)]
    pub(crate) fn page_hashes(&self, tokens: &[u32]) -> Vec<LocalBlockHash> {
        compute_block_hash_for_seq(tokens, self.page_size)
    }

    #[cfg(test)]
    pub fn match_prefix(&mut self, key: &[u32]) -> (usize, NodeId) {
        let page_keys = self.page_hashes(key);
        self.match_prefix_hashes(&page_keys)
    }

    #[cfg(test)]
    pub(crate) fn match_prefix_hashes(&mut self, page_keys: &[LocalBlockHash]) -> (usize, NodeId) {
        let key: PageKey = page_keys.to_vec().into();
        let result = self.tree.match_prefix(&MatchPrefixParams {
            key: &key,
            namespace: KeyNamespaceRef::default(),
        });
        assert!(
            result.cache_actions.is_empty(),
            "Full-only device match unexpectedly produced cache actions"
        );
        (
            result.device_indices.as_slice().len() * self.page_size,
            result.last_device_node_id,
        )
    }

    /// Match and protect a prefix using page hashes already owned by the request.
    ///
    /// Device-only: a host-only continuation is ignored. See
    /// [`Self::match_prefix_with_host_and_lock`] for the host hit.
    pub(crate) fn match_prefix_hashes_and_lock(
        &mut self,
        page_keys: &[LocalBlockHash],
    ) -> (usize, NodeId) {
        let matched = self.match_prefix_with_host_and_lock(page_keys);
        (matched.device_tokens, matched.last_node)
    }

    /// Match a prefix, lock its device part for the request, and report the
    /// host-only continuation that [`Self::load_back`] can bring back.
    ///
    /// The request lock sits on the deepest device node, like SGLang's
    /// `req.last_node` before `init_load_back`. Without write_back, or when
    /// nothing past the device prefix is on host, `host_hit` is `None`.
    pub(crate) fn match_prefix_with_host_and_lock(
        &mut self,
        page_keys: &[LocalBlockHash],
    ) -> PrefixMatch {
        let key: PageKey = page_keys.to_vec().into();
        let result = self.tree.match_prefix(&MatchPrefixParams {
            key: &key,
            namespace: KeyNamespaceRef::default(),
        });
        // A Full-only match can only emit the split fix-up for a pending
        // write-through ack, and write_back acks each backup inside the same
        // `evict()` call.
        assert!(
            result.cache_actions.is_empty(),
            "Full-only device match unexpectedly produced cache actions"
        );
        let last_node = result.last_device_node_id;
        let device_tokens = result.device_indices.as_slice().len() * self.page_size;
        self.tree
            .inc_lock_ref(last_node, ComponentSet::EMPTY)
            .expect("live prefix");
        let host_hit = (result.host_hit_length > 0).then(|| HostHit {
            tokens: result.host_hit_length * self.page_size,
            best_node: result.best_match_node_id,
            request_node: last_node,
        });
        PrefixMatch {
            device_tokens,
            last_node,
            host_hit,
        }
    }

    /// Start an H->D load-back of a host hit, following SGLang
    /// `init_load_back` -> `load_back` -> `_load_back_transfers`
    /// (URC:3412-3457, URC:1730-1842).
    ///
    /// Call it right after the match that produced `hit`, while the request
    /// still holds that match's lock. `mem_quota` is in tokens; `None` means
    /// no quota, as in SGLang.
    ///
    /// Locks, as in URC:
    /// 1. A host lock and a device lock on the anchor (`best_match_node`). The
    ///    device lock covers the evicted segment and its device ancestors, so
    ///    the eviction below cannot take them; the host lock keeps host
    ///    eviction off the anchor's host copy.
    /// 2. On refusal both are released and nothing else changes; the request
    ///    keeps its lock on the matched device node.
    /// 3. On success the device lock is released just before
    ///    `commit_load_back`, and a new device lock on the now device-resident
    ///    anchor is taken for the in-flight load. That lock and the host lock
    ///    go into the [`LoadBackHandle`]; `commit_load_back` also pins the
    ///    loaded nodes against demote and coexist reclaim until
    ///    [`Self::finish_load_back`].
    /// 4. The request lock then moves to the anchor: lock the anchor, then
    ///    release the matched device node. SGLang does the same in
    ///    `PrefillAdder.add_one_req`: `init_load_back` returns the anchor as
    ///    the new `req.last_node`, `_req_inc_lock_ref` locks it, and the
    ///    temporary `_lock_node` on the old `last_node` is released. The
    ///    request's prefix pages are then `collect_path_pages_through(
    ///    last_node, device_tokens + tokens)`, exactly as for a device hit.
    pub(crate) fn load_back(&mut self, hit: HostHit, mem_quota: Option<usize>) -> LoadBackResult {
        assert!(
            self.host_pool.is_some(),
            "load-back needs SGLang write_back"
        );
        let refused = |reason| LoadBackResult::Refused {
            reason,
            eviction: EvictionOutcome::default(),
        };
        let best = hit.best_node;
        if !matches!(self.tree.is_full_device_evicted(best), Ok(true)) {
            return refused(LoadBackRefusal::Stale);
        }

        let host_lock = self
            .tree
            .inc_host_lock_ref(best)
            .expect("live load-back anchor")
            .to_dec_params();
        let lock = self
            .tree
            .inc_lock_ref(best, ComponentSet::EMPTY)
            .expect("live load-back anchor");
        let newly_protected = lock.delta.unwrap_or(0) * self.page_size;
        let lock = lock.to_dec_params();

        let (kv_xfer, comp_xfers) = self
            .tree
            .build_load_back_spec(best, None)
            .expect("live load-back anchor");
        assert!(
            comp_xfers.is_empty(),
            "Full-only tree has no auxiliary load-back transfers"
        );
        // Empty when another in-flight load-back pins one of the nodes.
        let host_pages: Vec<HostPageId> = kv_xfer
            .host_indices
            .as_ref()
            .map(|value| host_pages_of(value).collect())
            .unwrap_or_default();
        let tokens = host_pages.len() * self.page_size;
        let refusal = if tokens < LOAD_BACK_THRESHOLD_TOKENS.max(1) {
            Some(LoadBackRefusal::BelowThreshold)
        } else if mem_quota.is_some_and(|quota| tokens + newly_protected > quota) {
            Some(LoadBackRefusal::OverQuota)
        } else {
            None
        };
        if let Some(reason) = refusal {
            self.release_load_back_locks(best, &lock, &host_lock);
            return refused(reason);
        }

        let mut eviction = EvictionOutcome::default();
        let available = self.page_pool.available();
        if available < tokens {
            eviction = self.evict(tokens - available);
            if self.page_pool.available() < tokens {
                self.release_load_back_locks(best, &lock, &host_lock);
                return LoadBackResult::Refused {
                    reason: LoadBackRefusal::NoDeviceRoom,
                    eviction,
                };
            }
        }
        let device_pages = self
            .page_pool
            .allocate_pages(host_pages.len())
            .expect("device room was just checked");

        self.tree
            .dec_lock_ref(best, &lock, false)
            .expect("live load-back anchor");
        let actions = self
            .tree
            .commit_load_back(
                best,
                PageValue::from_vec(device_pages.clone()),
                kv_xfer,
                comp_xfers,
            )
            .expect("live load-back anchor");
        assert!(
            actions.is_empty(),
            "Full-only load-back unexpectedly produced cache actions"
        );
        let lock = self
            .tree
            .inc_lock_ref(best, ComponentSet::EMPTY)
            .expect("loaded anchor")
            .to_dec_params();

        // Move the request lock to the loaded node.
        self.inc_lock_ref(best);
        self.dec_lock_ref(hit.request_node);

        LoadBackResult::Loaded(Box::new(LoadedBack {
            pages: host_pages.into_iter().zip(device_pages).collect(),
            tokens,
            last_node: best,
            eviction,
            handle: LoadBackHandle {
                anchor: best,
                lock,
                host_lock,
            },
        }))
    }

    /// Complete a load-back once its H2D copy is done, as SGLang
    /// `loading_check` does (URC:3356-3408): release the in-flight device
    /// lock and the host lock, then unpin the loaded nodes. The host copy
    /// stays as a coexisting duplicate that host eviction reclaims first.
    /// The request lock taken by [`Self::load_back`] is not touched.
    pub(crate) fn finish_load_back(&mut self, handle: LoadBackHandle) {
        let LoadBackHandle {
            anchor,
            lock,
            host_lock,
        } = handle;
        self.release_load_back_locks(anchor, &lock, &host_lock);
        self.tree
            .finish_load_back(anchor)
            .expect("live load-back anchor");
    }

    /// Host-lock a host hit's anchor without loading it, leaving it
    /// host-only. Production host locks come only from [`Self::load_back`],
    /// which device-locks the same anchor; this builds the states the
    /// manager cannot reach on its own.
    #[cfg(test)]
    pub(crate) fn host_lock_hit_for_test(&mut self, hit: HostHit) {
        self.tree
            .inc_host_lock_ref(hit.best_node)
            .expect("live host anchor");
    }

    fn release_load_back_locks(
        &mut self,
        anchor: NodeId,
        lock: &DecLockRefParams,
        host_lock: &DecLockRefParams,
    ) {
        self.tree
            .dec_lock_ref(anchor, lock, false)
            .expect("live load-back anchor");
        self.tree
            .dec_host_lock_ref(anchor, host_lock)
            .expect("live load-back anchor");
    }

    #[cfg(test)]
    pub fn prefix_match_len(&self, key: &[u32]) -> usize {
        let page_keys = self.page_hashes(key);
        self.prefix_match_hashes_len(&page_keys)
    }

    /// Read-only prefix match using page hashes already owned by the request.
    pub(crate) fn prefix_match_hashes_len(&self, page_keys: &[LocalBlockHash]) -> usize {
        self.tree
            .full_kv_prefix_len(page_keys, KeyNamespaceRef::default())
            * self.page_size
    }

    #[cfg(test)]
    pub fn insert(&mut self, key: &[u32], value: &[usize]) -> NodeId {
        let aligned_len = key.len() / self.page_size * self.page_size;
        assert!(
            value.len() >= aligned_len,
            "not enough token indices: need {aligned_len}, got {}",
            value.len()
        );
        let page_keys = self.page_hashes(&key[..aligned_len]);
        let pages = self.page_ids(&value[..aligned_len], page_keys.len());
        self.insert_page_hashes_from_node(self.root(), 0, &page_keys, &pages, false)
            .last_node
    }

    /// Insert page identities already materialized by the request lease.
    pub(crate) fn insert_page_hashes_from_node(
        &mut self,
        prefix_node: NodeId,
        prefix_len: usize,
        page_keys: &[LocalBlockHash],
        pages: &[KvPageId],
        chunked: bool,
    ) -> InsertPageResult {
        assert_eq!(
            prefix_len % self.page_size,
            0,
            "prefix length must be page-aligned"
        );
        assert!(
            pages.len() >= page_keys.len(),
            "not enough KV pages: need {}, got {}",
            page_keys.len(),
            pages.len()
        );
        let prefix_pages = prefix_len / self.page_size;
        assert!(
            prefix_pages <= page_keys.len(),
            "prefix pages {prefix_pages} exceed hashed pages {}",
            page_keys.len()
        );

        self.record_belady_pages(
            prefix_node,
            &page_keys[prefix_pages..],
            &pages[prefix_pages..page_keys.len()],
        );
        let key: PageKey = page_keys.to_vec().into();
        let suffix_value = PageValue::from_vec(pages[prefix_pages..page_keys.len()].to_vec());
        let result = self.tree.insert_suffix_from_node(
            prefix_node,
            prefix_pages,
            &InsertParams {
                rotation_base: None,
                session_id: None,
                swa_branching_seqlen: None,
                key: &key,
                namespace: KeyNamespaceRef::default(),
                value: suffix_value,
                prev_prefix_len: prefix_pages,
                swa_evicted_seqlen: 0,
                mamba_value: None,
                chunked,
                priority: 0,
                track_adopted_ranges: false,
            },
        );
        let last_node = result
            .last_device_node_id
            .expect("non-empty Full insert must end on a device node");
        let canonical_suffix = self
            .tree
            .collect_full_device_indices(last_node, prefix_node)
            .expect("live continuation path")
            .as_slice()
            .to_vec();
        assert_eq!(
            canonical_suffix.len(),
            page_keys.len() - prefix_pages,
            "SGLang core returned an incomplete canonical suffix"
        );

        let (unretained_pages, unretained_host_pages) =
            self.collect_unretained_pages(result.cache_actions);
        InsertPageResult {
            last_node,
            canonical_suffix,
            unretained_pages,
            unretained_host_pages,
        }
    }

    pub(crate) fn collect_path_pages(&self, last_node: NodeId) -> Vec<KvPageId> {
        self.tree
            .collect_full_device_indices(last_node, self.root())
            .expect("live prefix path")
            .as_slice()
            .to_vec()
    }

    pub(crate) fn collect_path_pages_through(
        &self,
        last_node: NodeId,
        prefix_len: usize,
    ) -> Vec<KvPageId> {
        assert_eq!(
            prefix_len % self.page_size,
            0,
            "matched SGLang prefix must be page-aligned"
        );
        let expected_pages = prefix_len / self.page_size;
        let mut pages = self.collect_path_pages(last_node);
        assert!(
            pages.len() >= expected_pages,
            "SGLang radix path returned {} pages for a {expected_pages}-page prefix",
            pages.len()
        );
        pages.truncate(expected_pages);
        pages
    }

    pub fn inc_lock_ref(&mut self, node_id: NodeId) {
        self.tree
            .inc_lock_ref(node_id, ComponentSet::EMPTY)
            .expect("live lock anchor");
    }

    pub fn dec_lock_ref(&mut self, node_id: NodeId) {
        // Full-only trees have no component UUIDs; the anchor identifies the receipt.
        self.tree
            .dec_lock_ref(
                node_id,
                &DecLockRefParams {
                    node_id: Some(node_id),
                    ..Default::default()
                },
                false,
            )
            .expect("live lock anchor");
    }

    /// Evict cache pages in SGLang LRU order.
    ///
    /// SGLang evicts complete compressed leaves, so this can release more than
    /// the requested number of tokens. Under write_back, each victim is first
    /// copied to the host pool and then demoted; see [`EvictionOutcome`].
    pub fn evict(&mut self, num_tokens: usize) -> EvictionOutcome {
        let target_pages = num_tokens.div_ceil(self.page_size);
        if target_pages == 0 {
            return EvictionOutcome::default();
        }
        if self.belady.is_some() {
            let pages = self.evict_belady(target_pages);
            self.page_pool.free_pages(&pages);
            return EvictionOutcome {
                evicted_tokens: pages.len() * self.page_size,
                device_pages: pages,
                host_events: Vec::new(),
            };
        }

        let write_back = self.host_pool.is_some();
        let mut tracker = HashMap::from([(FULL, 0)]);
        let mut outcome = EvictionOutcome {
            device_pages: Vec::with_capacity(target_pages.min(self.tree.evictable_size())),
            ..Default::default()
        };
        self.tree.evict_device_start(FULL, target_pages);
        // The core pops each candidate from its walk heap once, so a victim
        // that stays device-resident is not offered again and the walk ends.
        loop {
            let (candidate, step) = self.tree.evict_device_next_node(FULL, &tracker);
            self.absorb_eviction_step(Some(&mut tracker), &mut outcome, step);
            let Some(candidate) = candidate else {
                break;
            };

            let (backup, step) = self
                .tree
                .evict_device_leaf(candidate, write_back)
                .expect("live eviction candidate");
            self.absorb_eviction_step(Some(&mut tracker), &mut outcome, step);
            if let Some(backup) = backup {
                assert_eq!(
                    backup.node_ids,
                    [candidate],
                    "write_back backs up the victim alone"
                );
                self.write_back_victim(candidate, &mut tracker, &mut outcome);
            }
            if tracker.get(&FULL).copied().unwrap_or(0) >= target_pages {
                break;
            }
        }
        self.tree.evict_device_end(FULL);

        self.page_pool.free_pages(&outcome.device_pages);
        outcome.evicted_tokens = outcome.device_pages.len() * self.page_size;
        outcome
    }

    /// SGLang `_evict_components`: demote after a successful D->H backup,
    /// otherwise drop the unbacked subtree. A declined drop (host-locked
    /// subtree) leaves the victim device-resident until host space frees.
    fn write_back_victim(
        &mut self,
        node: NodeId,
        tracker: &mut HashMap<sglang_radix_tree::ComponentType, usize>,
        outcome: &mut EvictionOutcome,
    ) {
        let step = if self.back_up_to_host(node, outcome) {
            self.tree.demote(node).expect("backed-up D-leaf demotes")
        } else {
            let (_dropped, step) = self
                .tree
                .drop_subtree_no_host(node)
                .expect("live eviction candidate");
            step
        };
        // Demote and drop both count the device pages they free; a declined
        // drop returns an empty step.
        self.absorb_eviction_step(Some(tracker), outcome, step);
    }

    /// SGLang `_execute_and_commit_kv_backup(write_back=True)` followed by the
    /// blocking `writing_check(write_back=True)`: the copy is acknowledged in
    /// the same call, so no write-through stays pending.
    fn back_up_to_host(&mut self, node: NodeId, outcome: &mut EvictionOutcome) -> bool {
        let (device_value, comp_xfers) = self
            .tree
            .build_backup_spec(node)
            .expect("live eviction candidate");
        assert!(
            comp_xfers.is_empty(),
            "Full-only tree has no auxiliary backup transfers"
        );
        let device_pages = device_value.as_slice();
        assert!(
            !device_pages.is_empty(),
            "write_back victim is already backed up"
        );
        let needed = device_pages.len();
        let available = self.host_pool().expect("write_back").available_pages();
        if available < needed {
            let step = self.tree.drive_host_eviction(FULL, needed - available);
            self.absorb_eviction_step(None, outcome, step);
        }
        let Some(host_pages) = self
            .host_pool
            .as_mut()
            .expect("write_back")
            .allocate_pages(needed)
        else {
            return false;
        };

        self.tree
            .commit_backup(node, host_value(&host_pages), HashMap::new())
            .expect("live eviction candidate");
        let acked = self
            .tree
            .mark_write_through_pending(vec![node], node)
            .expect("live eviction candidate");
        self.tree
            .finish_write_through(acked, node)
            .expect("live eviction candidate");
        outcome.host_events.extend(
            host_pages
                .into_iter()
                .zip(device_pages.iter().copied())
                .map(|(host, device)| HostPageEvent::Stored { host, device }),
        );
        true
    }

    /// Fold one core step into the eviction outcome. Device frees are returned
    /// to the page pool by the caller; host frees go back to the host pool now,
    /// so a later backup in the same call can reuse them.
    fn absorb_eviction_step(
        &mut self,
        tracker: Option<&mut HashMap<sglang_radix_tree::ComponentType, usize>>,
        outcome: &mut EvictionOutcome,
        step: sglang_radix_tree::EvictionStepResult<PageValue<KvPageId>>,
    ) {
        if let Some(tracker) = tracker {
            for (component, count) in step.tracker {
                *tracker.entry(component).or_default() += count;
            }
        }
        for (component, chunks) in step.device_frees {
            assert_eq!(component, FULL, "Full-only tree evicted an auxiliary pool");
            for chunk in chunks {
                outcome.device_pages.extend_from_slice(chunk.as_slice());
            }
        }
        if step.host_frees.is_empty() {
            return;
        }
        let host_pool = self
            .host_pool
            .as_mut()
            .expect("device-only simulation tree unexpectedly freed host pages");
        for (component, chunks) in step.host_frees {
            assert_eq!(component, FULL, "Full-only tree evicted an auxiliary pool");
            for chunk in chunks {
                let pages: Vec<_> = host_pages_of(&chunk).collect();
                host_pool.free_pages(&pages);
                outcome
                    .host_events
                    .extend(pages.into_iter().map(HostPageEvent::Freed));
            }
        }
    }

    /// Split insert actions into device and host pages the tree did not keep.
    ///
    /// On a Full-only tree the core emits only duplicate-page frees:
    /// write-through backups never fire under write_back or without HiCache,
    /// the split fix-up needs a write-through still pending between calls, and
    /// host-slot frees come from SWA and Mamba components.
    fn collect_unretained_pages(
        &mut self,
        actions: Vec<CacheAction<PageValue<KvPageId>>>,
    ) -> (Vec<KvPageId>, Vec<HostPageId>) {
        let mut pages = Vec::new();
        let mut host_pages = Vec::new();
        for action in actions {
            match action {
                CacheAction::FreeDeviceKV(chunks) | CacheAction::FreeDeviceKVFullOnly(chunks) => {
                    for chunk in chunks {
                        pages.extend_from_slice(chunk.as_slice());
                    }
                }
                CacheAction::FreeComponentHostSlot {
                    component_type,
                    host_indices,
                } if component_type == FULL => {
                    let host_pool = self
                        .host_pool
                        .as_mut()
                        .expect("device-only simulation tree freed host pages");
                    for chunk in host_indices {
                        let freed: Vec<_> = host_pages_of(&chunk).collect();
                        host_pool.free_pages(&freed);
                        host_pages.extend(freed);
                    }
                }
                // write_back copies KV to host at eviction, never at insert.
                CacheAction::BackupKV(_) if self.host_pool.is_some() => {}
                CacheAction::BackupKV(_)
                | CacheAction::ReplaceWriteThroughOnNodeSplit { .. }
                | CacheAction::MambaEvictExcessPathStates { .. }
                | CacheAction::FreeComponentDeviceSlot { .. }
                | CacheAction::FreeComponentHostSlot { .. }
                | CacheAction::RebuildFullToSwaMapping { .. }
                | CacheAction::RecoverSwaWithLockedFull { .. }
                | CacheAction::SwaRebuild { .. } => {
                    panic!("Full-only simulation insert produced an unsupported cache action")
                }
            }
        }
        (pages, host_pages)
    }

    pub fn evictable_size(&self) -> usize {
        self.tree.evictable_size() * self.page_size
    }

    pub fn protected_size(&self) -> usize {
        self.tree.protected_size() * self.page_size
    }

    pub fn available_tokens(&self) -> usize {
        self.page_pool.available()
    }

    pub fn total_tokens(&self) -> usize {
        self.page_pool.total()
    }

    #[cfg(test)]
    fn page_ids(&self, indices: &[usize], page_count: usize) -> Vec<KvPageId> {
        assert!(
            indices.len() >= page_count * self.page_size,
            "not enough token indices for {page_count} complete pages"
        );
        indices
            .chunks_exact(self.page_size)
            .take(page_count)
            .map(|chunk| {
                let page = KvPageId::from_token_index(chunk[0], self.page_size);
                let start = page.first_token_index(self.page_size);
                assert!(
                    chunk.iter().copied().eq(start..start + self.page_size),
                    "SGLang cached pages must contain contiguous page-aligned indices"
                );
                page
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) struct TreeNodeSnapshot {
    pub(crate) parent: Option<NodeId>,
    pub(crate) children: rustc_hash::FxHashMap<LocalBlockHash, NodeId>,
    pub(crate) lock_ref: u32,
    pub(crate) last_access_counter: i64,
    pub(crate) key: Vec<LocalBlockHash>,
    pub(crate) value: Vec<KvPageId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::common::hashing::compute_seq_hash_for_block;
    use uuid::Uuid;

    fn oracle_for_prompts(prompts: &[&[u32]]) -> BeladyOracle {
        BeladyOracle::new(
            prompts
                .iter()
                .enumerate()
                .map(|(index, tokens)| {
                    (
                        Uuid::from_u128(index as u128 + 1),
                        compute_seq_hash_for_block(&compute_block_hash_for_seq(tokens, 1)),
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    #[rstest::rstest]
    fn belady_reranks_compressed_tail_before_evicting_its_prefix(
        #[values(false, true)] shared: bool,
    ) {
        let mut cache = RadixCache::new(3, 1);
        cache.set_belady_oracle(oracle_for_prompts(&[&[1], &[3], &[1, 2]]));
        let ab = cache.page_pool.allocate(2).unwrap();
        cache.insert(&[1, 2], &ab);
        let c = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[3], &c);
        let snapshot = shared.then(|| cache.admission_checkpoint());

        let outcome = cache.evict(2);
        assert_eq!(
            (outcome.evicted_tokens, outcome.device_pages),
            (2, vec![KvPageId(1), KvPageId(2)])
        );
        assert_eq!(cache.prefix_match_len(&[1, 2]), 1);
        assert_eq!(cache.prefix_match_len(&[3]), 0);
        assert_eq!(cache.available_tokens(), 2);
        if let Some(mut saved) = snapshot {
            assert_eq!(saved.prefix_match_len(&[1, 2]), 2);
            assert_eq!(saved.prefix_match_len(&[3]), 1);
            assert_eq!(saved.available_tokens(), 0);
            assert_eq!(saved.evict(2).device_pages, vec![KvPageId(1), KvPageId(2)]);
        }
    }

    #[test]
    fn belady_observes_global_retirement_before_local_victim_selection() {
        let oracle = oracle_for_prompts(&[&[1], &[2]]);
        let mut cache = RadixCache::new(2, 1);
        cache.set_belady_oracle(oracle.clone());
        let a = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[1], &a);
        let b = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[2], &b);

        // Another worker commits the first request; this worker must refresh its hidden key.
        oracle.retire_requests([Uuid::from_u128(1)]);
        assert_eq!(cache.evict(1).device_pages, vec![KvPageId(0)]);
        assert_eq!(cache.prefix_match_len(&[2]), 1);
    }

    #[test]
    fn belady_preserves_locked_prefixes_and_handles_branching_and_extension() {
        let mut cache = RadixCache::new(8, 1);
        cache.set_belady_oracle(oracle_for_prompts(&[&[1, 2], &[1, 2, 4], &[9], &[1, 2, 3]]));
        let mut pages = cache.page_pool.allocate(2).unwrap();
        let parent = cache.insert(&[1, 2], &pages);
        cache.inc_lock_ref(parent);
        pages.extend(cache.page_pool.allocate(1).unwrap());
        cache.insert_page_hashes_from_node(
            parent,
            2,
            &cache.page_hashes(&[1, 2, 3]),
            &pages.iter().copied().map(KvPageId).collect::<Vec<_>>(),
            true,
        );
        cache.dec_lock_ref(parent);
        let mut branch = pages[..2].to_vec();
        branch.extend(cache.page_pool.allocate(1).unwrap());
        cache.insert(&[1, 2, 4], &branch);
        let other = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[9], &other);

        assert_eq!(cache.evict(2).device_pages, vec![KvPageId(2), KvPageId(4)]);
        assert_eq!(cache.prefix_match_len(&[1, 2, 4]), 3);
        let (_, locked) = cache.match_prefix(&[1, 2]);
        cache.inc_lock_ref(locked);
        assert_eq!(cache.evict(8).device_pages, vec![KvPageId(3)]);
        assert_eq!(cache.protected_size(), 2);
        cache.dec_lock_ref(locked);
        assert_eq!(cache.evict(2).device_pages, vec![KvPageId(1), KvPageId(0)]);
        assert_eq!(cache.available_tokens(), 8);
    }

    #[test]
    fn belady_recycled_page_has_the_new_prefix_identity() {
        let oracle = oracle_for_prompts(&[&[2], &[3]]);
        let mut cache = RadixCache::new(2, 1);
        cache.set_belady_oracle(oracle);
        let first = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[1], &first);
        assert_eq!(cache.evict(1).device_pages, vec![KvPageId(0)]);
        let reused = cache.page_pool.allocate(1).unwrap();
        assert_eq!(reused, first);
        cache.insert(&[2], &reused);
        let second = cache.page_pool.allocate(1).unwrap();
        cache.insert(&[3], &second);
        assert_eq!(cache.evict(1).device_pages, vec![KvPageId(1)]);
        assert_eq!(cache.prefix_match_len(&[2]), 1);
    }

    #[test]
    fn belady_distinguishes_equal_local_pages_below_different_prefixes() {
        let mut cache = RadixCache::new(4, 1);
        cache.set_belady_oracle(oracle_for_prompts(&[&[1, 7], &[2]]));
        let first = cache.page_pool.allocate(2).unwrap();
        cache.insert(&[1, 7], &first);
        let second = cache.page_pool.allocate(2).unwrap();
        cache.insert(&[2, 7], &second);

        assert_eq!(cache.evict(2).device_pages, vec![KvPageId(3), KvPageId(2)]);
        assert_eq!(cache.prefix_match_len(&[1, 7]), 2);
        assert_eq!(cache.prefix_match_len(&[2, 7]), 0);
    }

    #[test]
    fn page_pool_allocate_extend_and_free() {
        let mut pool = PagePool::new(12, 4);
        assert_eq!(pool.available(), 12);
        assert!(pool.allocate(usize::MAX).is_none());
        let first = pool.allocate(3).unwrap();
        assert_eq!(pool.available(), 8);
        let mut extended = first.clone();
        assert!(pool.allocate_indices_into(1, &mut extended));
        assert_eq!(extended, vec![0, 1, 2, 3]);
        let second = pool.allocate(5).unwrap();
        assert_eq!(pool.available(), 0);
        assert!(pool.allocate(1).is_none());
        pool.free(&first);
        pool.free(&second);
        assert_eq!(pool.available(), 12);
    }

    #[test]
    fn allocation_failure_is_atomic() {
        let mut pool = PagePool::new(8, 4);
        let mut destination = pool.allocate(4).unwrap();
        let _other = pool.allocate(4).unwrap();
        let available_before = pool.available();
        let destination_before = destination.clone();

        assert!(!pool.allocate_indices_into(1, &mut destination));
        assert_eq!(destination, destination_before);
        assert_eq!(pool.available(), available_before);
    }

    #[test]
    fn match_insert_and_read_only_score_use_sglang_core() {
        let mut cache = RadixCache::new(64, 4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        cache.insert(&[1, 2, 3, 4, 9, 10, 11, 12], &[0, 1, 2, 3, 8, 9, 10, 11]);

        assert_eq!(cache.prefix_match_len(&[1, 2, 3, 4, 13, 14, 15, 16]), 4);
        let nodes_before = cache.num_nodes();
        assert_eq!(cache.prefix_match_len(&[1, 2, 3, 4, 13, 14, 15, 16]), 4);
        assert_eq!(cache.num_nodes(), nodes_before);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);
    }

    #[test]
    fn continuation_returns_canonical_suffix_and_duplicate_pages() {
        let mut cache = RadixCache::new(64, 4);
        let existing = cache.page_pool.allocate_pages(2).unwrap();
        let key = cache.page_hashes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let _ = cache.insert_page_hashes_from_node(cache.root(), 0, &key, &existing, false);

        let (prefix_len, prefix_node) = cache.match_prefix_hashes_and_lock(&key[..1]);
        assert_eq!(prefix_len, 4);
        let incoming_suffix = cache.page_pool.allocate_pages(1).unwrap();
        let incoming = [existing[0], incoming_suffix[0]];
        let result =
            cache.insert_page_hashes_from_node(prefix_node, prefix_len, &key, &incoming, true);

        assert_eq!(result.canonical_suffix, vec![existing[1]]);
        assert_eq!(result.unretained_pages, incoming_suffix);
        assert_eq!(cache.collect_path_pages(result.last_node), existing);
        cache.dec_lock_ref(prefix_node);
    }

    #[test]
    fn lock_accounting_and_lru_eviction_follow_sglang() {
        let mut cache = RadixCache::new(64, 4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        cache.insert(&[9, 10, 11, 12], &[8, 9, 10, 11]);

        let first_key = cache.page_hashes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let (_, locked) = cache.match_prefix_hashes_and_lock(&first_key);
        assert_eq!(cache.protected_size(), 8);
        assert_eq!(cache.evictable_size(), 4);

        let outcome = cache.evict(4);
        assert_eq!(outcome.evicted_tokens, 4);
        assert_eq!(outcome.device_pages, vec![KvPageId(2)]);
        assert!(outcome.host_events.is_empty());
        assert_eq!(cache.match_prefix(&[9, 10, 11, 12]).0, 0);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);

        cache.dec_lock_ref(locked);
        assert_eq!(cache.protected_size(), 0);
        assert_eq!(cache.evictable_size(), 8);
    }

    /// Insert like the KV manager: allocate request pages, insert from the root,
    /// and return the pages the tree did not retain.
    fn insert_pages(cache: &mut RadixCache, keys: &[LocalBlockHash]) -> Option<NodeId> {
        let available = cache.page_pool.available_pages();
        if available < keys.len() {
            cache.evict((keys.len() - available) * cache.page_size());
        }
        let pages = cache.page_pool.allocate_pages(keys.len())?;
        let result = cache.insert_page_hashes_from_node(cache.root(), 0, keys, &pages, false);
        cache.page_pool.free_pages(&result.unretained_pages);
        assert!(result.unretained_host_pages.is_empty());
        Some(result.last_node)
    }

    fn keys(values: &[u64]) -> Vec<LocalBlockHash> {
        values.iter().copied().map(LocalBlockHash).collect()
    }

    fn host_pages(cache: &RadixCache, node: NodeId) -> Option<Vec<HostPageId>> {
        cache
            .tree
            .inspect_get_component_host_value(node, FULL)
            .unwrap()
            .map(|value| host_pages_of(&value).collect())
    }

    /// Every page is either free or held by exactly one tree node, on both tiers.
    fn assert_pool_accounting(cache: &RadixCache) {
        assert_pool_accounting_with_loads(cache, &[]);
    }

    /// As [`assert_pool_accounting`], with the anchors of in-flight load-backs.
    fn assert_pool_accounting_with_loads(cache: &RadixCache, pending: &[&LoadBackHandle]) {
        let ongoing: Vec<_> = pending
            .iter()
            .map(|handle| (handle.node() as i64, handle.node()))
            .collect();
        cache.tree.sanity_check(&[], &ongoing);
        let mut device = Vec::new();
        let mut host = Vec::new();
        for node in cache.tree.inspect_get_all_node_ids() {
            if let Some(value) = cache.tree.get_component_device_value(node, FULL).unwrap() {
                device.extend_from_slice(value.as_slice());
            }
            if let Some(value) = cache
                .tree
                .inspect_get_component_host_value(node, FULL)
                .unwrap()
            {
                host.extend(host_pages_of(&value));
            }
        }
        assert_eq!(
            device.len(),
            cache.tree.evictable_size() + cache.tree.protected_size()
        );
        assert_eq!(
            cache.page_pool.available_pages() + device.len(),
            cache.page_pool.total_pages
        );
        device.sort_unstable();
        device.dedup();
        assert_eq!(
            device.len(),
            cache.tree.evictable_size() + cache.tree.protected_size()
        );
        let pool = cache.host_pool.as_ref().expect("write_back cache");
        assert_eq!(pool.available_pages() + host.len(), pool.total_pages());
        let held = host.len();
        host.sort_unstable();
        host.dedup();
        assert_eq!(host.len(), held, "host page held twice");
    }

    #[test]
    fn write_back_eviction_with_host_room_demotes_to_host() {
        let mut cache = RadixCache::new_write_back(8, 2, 4);
        let node = insert_pages(&mut cache, &keys(&[1, 2])).unwrap();
        let device = cache.collect_path_pages(node);

        let outcome = cache.evict(4);

        assert_eq!(outcome.evicted_tokens, 4);
        assert_eq!(outcome.device_pages, device);
        let stored: Vec<_> = outcome.stored_host().collect();
        assert_eq!(stored.len(), 2);
        assert_eq!(
            stored.iter().map(|&(_, page)| page).collect::<Vec<_>>(),
            device
        );
        assert_eq!(outcome.host_frees().count(), 0);
        assert_eq!(cache.available_tokens(), 8);
        assert_eq!(cache.host_pool().unwrap().available_pages(), 2);
        // The node stays in the tree as host-only.
        assert_eq!(
            host_pages(&cache, node),
            Some(stored.iter().map(|&(host, _)| host).collect())
        );
        assert!(
            cache
                .tree
                .get_component_device_value(node, FULL)
                .unwrap()
                .is_none()
        );
        assert!(cache.tree.inspect_is_host_evictable_leaf(node));
        assert_pool_accounting(&cache);
    }

    #[test]
    fn write_back_frees_host_lru_leaf_before_storing_a_new_victim() {
        let mut cache = RadixCache::new_write_back(8, 1, 1);
        let old = insert_pages(&mut cache, &keys(&[1])).unwrap();
        let new = insert_pages(&mut cache, &keys(&[2])).unwrap();
        let outcome = cache.evict(1);
        let old_host = host_pages(&cache, old).unwrap();
        assert_eq!(outcome.stored_host().count(), 1);
        assert_eq!(cache.host_pool().unwrap().available_pages(), 0);

        let outcome = cache.evict(1);

        assert_eq!(outcome.device_pages.len(), 1);
        assert!(!cache.tree.inspect_contains_node(old));
        assert_eq!(
            outcome.host_events,
            vec![
                HostPageEvent::Freed(old_host[0]),
                HostPageEvent::Stored {
                    host: old_host[0],
                    device: outcome.device_pages[0],
                },
            ]
        );
        assert_eq!(host_pages(&cache, new), Some(old_host));
        assert_pool_accounting(&cache);
    }

    #[test]
    fn write_back_drops_unbacked_leaf_when_host_room_cannot_be_made() {
        let mut cache = RadixCache::new_write_back(8, 1, 1);
        let pinned = insert_pages(&mut cache, &keys(&[1])).unwrap();
        cache.evict(1);
        cache.tree.inc_host_lock_ref(pinned).unwrap();
        let victim = insert_pages(&mut cache, &keys(&[2])).unwrap();
        let victim_page = cache.collect_path_pages(victim);

        let outcome = cache.evict(1);

        assert_eq!(outcome.device_pages, victim_page);
        assert!(outcome.host_events.is_empty());
        assert!(!cache.tree.inspect_contains_node(victim));
        assert!(host_pages(&cache, pinned).is_some());
        assert_eq!(cache.host_pool().unwrap().available_pages(), 0);
        assert_pool_accounting(&cache);
    }

    #[test]
    fn write_back_keeps_host_locked_subtree_device_resident() {
        let mut cache = RadixCache::new_write_back(8, 1, 1);
        let parent = insert_pages(&mut cache, &keys(&[1])).unwrap();
        let child = insert_pages(&mut cache, &keys(&[1, 2])).unwrap();
        assert_ne!(parent, child);
        // The child demotes into the only host page.
        assert_eq!(cache.evict(1).stored_host().count(), 1);
        cache.tree.inc_host_lock_ref(child).unwrap();

        // The parent can neither be backed up (the host copy is locked) nor
        // dropped (its subtree is host-locked), and eviction still terminates.
        let outcome = cache.evict(8);

        assert_eq!(outcome, EvictionOutcome::default());
        assert_eq!(cache.collect_path_pages(parent).len(), 1);
        assert!(host_pages(&cache, child).is_some());
        assert_pool_accounting(&cache);
    }

    /// The state above cannot arise without a test-only host lock. Every
    /// production host lock comes from `load_back`, which device-locks the
    /// same anchor once it is loaded, and a device lock reaches the root. So
    /// a device page above a host-locked node is never an eviction candidate
    /// nor counted in `evictable_size`, and eviction frees at least the
    /// smaller of the target and `evictable_size`, as the capacity checks of
    /// admission and prefill assume.
    #[test]
    fn host_locked_subtree_keeps_its_device_ancestors_out_of_evictable_size() {
        let mut cache = RadixCache::new_write_back(20, 4, 3);
        let parent = insert_pages(&mut cache, &keys(&[1])).unwrap();
        let child_key = keys(&[1, 2, 3, 4]);
        let child = insert_pages(&mut cache, &child_key).unwrap();
        insert_pages(&mut cache, &keys(&[9])).unwrap();
        // The child demotes into all three host pages.
        assert_eq!(cache.evict(12).stored_host().count(), 3);
        let matched = cache.match_prefix_with_host_and_lock(&child_key);
        assert_eq!((matched.device_tokens, matched.last_node), (4, parent));
        let loaded = expect_loaded(cache.load_back(matched.host_hit.unwrap(), None));
        assert_eq!(loaded.last_node, child);
        cache.dec_lock_ref(loaded.last_node);

        // The host is full and its only copy is locked by the in-flight
        // load, but the locked node is device-resident and device-locked, so
        // its parent is protected too: only the unrelated leaf is evictable.
        assert_eq!(cache.host_pool().unwrap().available_pages(), 0);
        assert!(host_pages(&cache, child).is_some());
        assert_eq!(cache.collect_path_pages(child).len(), 4);
        assert!(device_lock_ref(&cache, parent) > 0);
        assert_eq!(cache.evictable_size(), 4);
        let outcome = cache.evict(cache.total_tokens());
        assert_eq!(outcome.evicted_tokens, 4, "the leaf is dropped");
        assert!(outcome.host_events.is_empty());
        assert_eq!(cache.evictable_size(), 0);

        // Once the load finishes, all of it is evictable again and is freed.
        cache.finish_load_back(loaded.handle);
        let evictable = cache.evictable_size();
        assert_eq!(evictable, 16);
        assert_eq!(cache.evict(cache.total_tokens()).evicted_tokens, evictable);
        assert_eq!(cache.available_tokens(), cache.total_tokens());
        assert_pool_accounting(&cache);
    }

    #[test]
    fn match_after_demote_reports_no_device_hit() {
        let mut cache = RadixCache::new_write_back(8, 2, 4);
        let key = keys(&[1, 2]);
        insert_pages(&mut cache, &key).unwrap();
        cache.evict(4);

        assert_eq!(cache.match_prefix_hashes(&key), (0, cache.root()));
        let (matched, locked) = cache.match_prefix_hashes_and_lock(&key);
        assert_eq!((matched, locked), (0, cache.root()));
        cache.dec_lock_ref(locked);
        assert_eq!(cache.prefix_match_hashes_len(&key), 0);
        // The core already reports the host hit that load-back will consume.
        let result = cache.tree.match_prefix(&MatchPrefixParams {
            key: &PageKey::from(key.clone()),
            namespace: KeyNamespaceRef::default(),
        });
        assert_eq!(result.host_hit_length, 2);
        assert!(result.cache_actions.is_empty());
        assert_pool_accounting(&cache);
    }

    /// U1: under write_back, host eviction first reclaims host copies whose node
    /// is device-resident again, so a host free need not remove device KV.
    #[test]
    fn host_eviction_reclaims_coexisting_copy_and_keeps_device_value() {
        let mut cache = RadixCache::new_write_back(8, 1, 1);
        let key = keys(&[1]);
        let node = insert_pages(&mut cache, &key).unwrap();
        cache.evict(1);
        let host = host_pages(&cache, node).unwrap();
        // Re-inserting the prefix restores device KV next to the host copy.
        assert_eq!(insert_pages(&mut cache, &key), Some(node));
        let device = cache.collect_path_pages(node);
        insert_pages(&mut cache, &keys(&[1, 2])).unwrap();

        // Storing the [2] leaf needs the only host page, held by the node above it.
        let outcome = cache.evict(1);

        assert_eq!(outcome.host_events[0], HostPageEvent::Freed(host[0]));
        assert!(matches!(
            outcome.host_events[1],
            HostPageEvent::Stored { host: page, .. } if page == host[0]
        ));
        assert_eq!(host_pages(&cache, node), None);
        assert_eq!(cache.collect_path_pages(node), device);
        assert_pool_accounting(&cache);
    }

    /// U2: `backuped()` follows the host value written by `commit_backup`; the
    /// write-through ack pair only clears the pending mark it sets.
    #[test]
    fn commit_backup_alone_marks_node_backed_up() {
        let mut cache = RadixCache::new_write_back(8, 1, 4);
        let node = insert_pages(&mut cache, &keys(&[1])).unwrap();
        let host = cache.host_pool.as_mut().unwrap().allocate_pages(1).unwrap();

        cache
            .tree
            .commit_backup(node, host_value(&host), HashMap::new())
            .unwrap();

        assert!(cache.tree.node_backuped(node).unwrap());
        assert_eq!(
            cache
                .tree
                .inspect_get_write_through_pending_id(node)
                .unwrap(),
            None
        );
        let ack = cache
            .tree
            .mark_write_through_pending(vec![node], node)
            .unwrap();
        assert_eq!(
            cache
                .tree
                .inspect_get_write_through_pending_id(node)
                .unwrap(),
            Some(node)
        );
        cache.tree.finish_write_through(ack, node).unwrap();
        assert_eq!(
            cache
                .tree
                .inspect_get_write_through_pending_id(node)
                .unwrap(),
            None
        );
        assert!(cache.tree.node_backuped(node).unwrap());
    }

    #[test]
    fn write_back_random_ops_keep_both_pools_balanced() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0x5eed);
        // Four-token pages: a host hit of three pages clears the load-back threshold.
        let mut cache = RadixCache::new_write_back(48, 4, 16);
        let mut locked: Vec<NodeId> = Vec::new();
        let mut pending: Vec<LoadBackHandle> = Vec::new();
        let prompts: Vec<Vec<LocalBlockHash>> = (0..6)
            .map(|_| {
                (0..8)
                    .map(|page| LocalBlockHash(rng.random_range(1..=if page < 2 { 2 } else { 6 })))
                    .collect()
            })
            .collect();
        let mut inserted: Vec<Vec<LocalBlockHash>> = Vec::new();
        let (mut demotes, mut host_frees) = (0, 0);
        let (mut loads, mut refusals, mut finishes) = (0, 0, 0);
        let mut absorb = |outcome: &EvictionOutcome, page_size: usize| {
            assert_eq!(
                outcome.evicted_tokens,
                outcome.device_pages.len() * page_size
            );
            for (_, device) in outcome.stored_host() {
                assert!(outcome.device_pages.contains(&device));
            }
            demotes += outcome.stored_host().count();
            host_frees += outcome.host_frees().count();
        };
        for _ in 0..1000 {
            match rng.random_range(0..12) {
                0..4 => {
                    // Prefixes of a few long prompts that share their first
                    // pages, so demoted leaves are long enough to load back.
                    let prompt = &prompts[rng.random_range(0..prompts.len())];
                    let key = prompt[..rng.random_range(1..=prompt.len())].to_vec();
                    if insert_pages(&mut cache, &key).is_some() {
                        inserted.push(key);
                    }
                }
                4..6 if !inserted.is_empty() => {
                    let key = &inserted[rng.random_range(0..inserted.len())];
                    let prefix = if rng.random_bool(0.7) {
                        key.len()
                    } else {
                        rng.random_range(1..=key.len())
                    };
                    let matched = cache.match_prefix_with_host_and_lock(&key[..prefix]);
                    let Some(hit) = matched.host_hit else {
                        locked.push(matched.last_node);
                        continue;
                    };
                    let quota = rng.random_bool(0.2).then(|| rng.random_range(0..=24));
                    match cache.load_back(hit, quota) {
                        LoadBackResult::Loaded(loaded) => {
                            loads += 1;
                            absorb(&loaded.eviction, cache.page_size());
                            assert_eq!(loaded.tokens, loaded.pages.len() * cache.page_size());
                            assert_eq!(loaded.tokens, hit.tokens());
                            let path = cache.collect_path_pages_through(
                                loaded.last_node,
                                matched.device_tokens + loaded.tokens,
                            );
                            assert!(path.ends_with(
                                &loaded.pages.iter().map(|&(_, d)| d).collect::<Vec<_>>()
                            ));
                            locked.push(loaded.last_node);
                            pending.push(loaded.handle);
                        }
                        LoadBackResult::Refused { eviction, .. } => {
                            refusals += 1;
                            absorb(&eviction, cache.page_size());
                            locked.push(matched.last_node);
                        }
                    }
                }
                6..8 if !locked.is_empty() => {
                    let node = locked.swap_remove(rng.random_range(0..locked.len()));
                    cache.dec_lock_ref(node);
                }
                8..10 if !pending.is_empty() => {
                    let handle = pending.swap_remove(rng.random_range(0..pending.len()));
                    cache.finish_load_back(handle);
                    finishes += 1;
                }
                _ => {
                    // Admission sizes its capacity check with `evictable_size`:
                    // eviction must free at least the smaller of the two.
                    let target = rng.random_range(1..=24);
                    let evictable = cache.evictable_size();
                    let outcome = cache.evict(target);
                    absorb(&outcome, cache.page_size());
                    assert!(
                        outcome.evicted_tokens
                            >= (target.div_ceil(cache.page_size()) * cache.page_size())
                                .min(evictable),
                        "evicted {} of {target} with {evictable} evictable",
                        outcome.evicted_tokens
                    );
                }
            }
            assert_pool_accounting_with_loads(&cache, &pending.iter().collect::<Vec<_>>());
        }
        assert!(
            demotes > 0 && host_frees > 0 && loads > 0 && refusals > 0 && finishes > 0,
            "{demotes} demotes, {host_frees} host frees, {loads} loads, \
             {refusals} refusals, {finishes} finishes"
        );

        for handle in pending.drain(..) {
            cache.finish_load_back(handle);
        }
        for node in locked.drain(..) {
            cache.dec_lock_ref(node);
        }
        assert_eq!(cache.protected_size(), 0);
        cache.evict(cache.total_tokens());
        assert_eq!(cache.available_tokens(), cache.total_tokens());
        assert_pool_accounting(&cache);
    }

    /// Free host pages the way SGLang `evict_host` does (URC:1368-1380).
    fn evict_host(cache: &mut RadixCache, pages: usize) -> EvictionOutcome {
        let mut outcome = EvictionOutcome::default();
        let step = cache.tree.drive_host_eviction(FULL, pages);
        cache.absorb_eviction_step(None, &mut outcome, step);
        assert!(outcome.device_pages.is_empty());
        outcome
    }

    fn device_lock_ref(cache: &RadixCache, node: NodeId) -> u32 {
        cache
            .tree
            .inspect_get_component_device_lock_ref(node, FULL)
            .unwrap()
    }

    /// Insert `key`, demote it to host, and return its node and host pages.
    fn demoted(cache: &mut RadixCache, key: &[LocalBlockHash]) -> (NodeId, Vec<HostPageId>) {
        let node = insert_pages(cache, key).unwrap();
        let tokens = key.len() * cache.page_size();
        let outcome = cache.evict(tokens);
        assert_eq!(outcome.stored_host().count(), key.len());
        (node, host_pages(cache, node).unwrap())
    }

    fn expect_loaded(result: LoadBackResult) -> LoadedBack {
        match result {
            LoadBackResult::Loaded(loaded) => *loaded,
            LoadBackResult::Refused { reason, .. } => panic!("load-back refused: {reason:?}"),
        }
    }

    #[test]
    fn host_aware_match_reports_host_hit_after_demote() {
        let mut cache = RadixCache::new_write_back(16, 4, 4);
        let key = keys(&[1, 2, 3]);
        let (node, _) = demoted(&mut cache, &key);

        let matched = cache.match_prefix_with_host_and_lock(&key);

        assert_eq!(matched.device_tokens, 0);
        assert_eq!(matched.last_node, cache.root());
        assert_eq!(matched.host_tokens(), 12);
        assert_eq!(matched.host_hit.unwrap().best_node, node);
        // The device-only callers and the read-only score are unchanged.
        cache.dec_lock_ref(matched.last_node);
        assert_eq!(cache.match_prefix_hashes_and_lock(&key), (0, cache.root()));
        cache.dec_lock_ref(cache.root());
        assert_eq!(cache.prefix_match_hashes_len(&key), 0);
        assert_pool_accounting(&cache);
    }

    #[test]
    fn host_aware_match_without_host_hit_matches_device_only_path() {
        let mut cache = RadixCache::new_write_back(16, 4, 4);
        let key = keys(&[1, 2]);
        let node = insert_pages(&mut cache, &key).unwrap();

        let matched = cache.match_prefix_with_host_and_lock(&key);

        assert_eq!((matched.device_tokens, matched.last_node), (8, node));
        assert!(matched.host_hit.is_none());
        assert_eq!(matched.host_tokens(), 0);
        assert_eq!(device_lock_ref(&cache, node), 1);
        cache.dec_lock_ref(node);
    }

    #[test]
    fn load_back_below_threshold_is_refused_with_locks_balanced() {
        // Two pages of two tokens: 4 tokens, below SGLang's threshold of 10.
        let mut cache = RadixCache::new_write_back(8, 2, 4);
        let key = keys(&[1, 2]);
        let (node, host) = demoted(&mut cache, &key);
        let matched = cache.match_prefix_with_host_and_lock(&key);
        assert_eq!(matched.host_tokens(), 4);

        let result = cache.load_back(matched.host_hit.unwrap(), None);

        assert!(matches!(
            result,
            LoadBackResult::Refused {
                reason: LoadBackRefusal::BelowThreshold,
                ref eviction,
            } if *eviction == EvictionOutcome::default()
        ));
        assert_eq!(device_lock_ref(&cache, node), 0);
        assert!(host_pages(&cache, node).is_some());
        assert_eq!(cache.available_tokens(), 8);
        // The request keeps its lock where the match put it.
        cache.dec_lock_ref(matched.last_node);
        // No host lock is left: host eviction can reclaim the node.
        let outcome = evict_host(&mut cache, 2);
        assert_eq!(outcome.host_frees().collect::<Vec<_>>(), host);
        assert!(!cache.tree.inspect_contains_node(node));
        assert_pool_accounting(&cache);
    }

    #[test]
    fn load_back_over_quota_or_without_device_room_is_refused() {
        let mut cache = RadixCache::new_write_back(16, 4, 4);
        let key = keys(&[1, 2, 3]);
        let (node, host) = demoted(&mut cache, &key);

        let matched = cache.match_prefix_with_host_and_lock(&key);
        let result = cache.load_back(matched.host_hit.unwrap(), Some(11));
        assert!(matches!(
            result,
            LoadBackResult::Refused {
                reason: LoadBackRefusal::OverQuota,
                ..
            }
        ));
        cache.dec_lock_ref(matched.last_node);

        // Fill the device with a locked prefix so eviction cannot make room.
        let other = keys(&[7, 8]);
        let other_node = insert_pages(&mut cache, &other).unwrap();
        let (_, pinned) = cache.match_prefix_hashes_and_lock(&other);
        assert_eq!(pinned, other_node);
        let filler = cache.page_pool.allocate_pages(2).unwrap();
        let matched = cache.match_prefix_with_host_and_lock(&key);
        let result = cache.load_back(matched.host_hit.unwrap(), None);
        assert!(matches!(
            result,
            LoadBackResult::Refused {
                reason: LoadBackRefusal::NoDeviceRoom,
                ref eviction,
            } if *eviction == EvictionOutcome::default()
        ));
        assert_eq!(device_lock_ref(&cache, node), 0);
        cache.dec_lock_ref(matched.last_node);
        cache.dec_lock_ref(pinned);
        cache.page_pool.free_pages(&filler);

        // Neither refusal left a host lock behind.
        let outcome = evict_host(&mut cache, 3);
        assert_eq!(outcome.host_frees().collect::<Vec<_>>(), host);
        assert_pool_accounting(&cache);
    }

    #[test]
    fn load_back_restores_device_hit_and_pins_until_finished() {
        let mut cache = RadixCache::new_write_back(16, 4, 4);
        let key = keys(&[1, 2, 3]);
        let (node, host) = demoted(&mut cache, &key);
        let matched = cache.match_prefix_with_host_and_lock(&key);

        let loaded = expect_loaded(cache.load_back(matched.host_hit.unwrap(), None));

        assert_eq!(loaded.tokens, 12);
        assert_eq!(loaded.eviction, EvictionOutcome::default());
        assert_eq!(
            loaded.pages.iter().map(|&(h, _)| h).collect::<Vec<_>>(),
            host
        );
        let device: Vec<_> = loaded.pages.iter().map(|&(_, d)| d).collect();
        assert_eq!(cache.available_tokens(), 4);
        // The request lock moved from the matched device node to the loaded node.
        assert_eq!(loaded.last_node, node);
        assert_eq!(
            cache.collect_path_pages_through(
                loaded.last_node,
                matched.device_tokens + loaded.tokens
            ),
            device
        );
        assert_eq!(device_lock_ref(&cache, node), 2, "request + in-flight load");
        assert_eq!(cache.match_prefix_hashes(&key), (12, node));
        assert_eq!(cache.prefix_match_hashes_len(&key), 12);
        assert_pool_accounting_with_loads(&cache, &[&loaded.handle]);

        // The request finishes before the H2D does: the in-flight load still
        // keeps the node off the eviction walk and off demote.
        cache.dec_lock_ref(loaded.last_node);
        assert_eq!(cache.evict(16), EvictionOutcome::default());
        let step = cache.tree.demote(node).unwrap();
        assert!(step.device_frees.is_empty() && step.host_frees.is_empty());
        // The host copy is pinned against coexist reclaim too.
        assert_eq!(evict_host(&mut cache, 3), EvictionOutcome::default());

        cache.finish_load_back(loaded.handle);

        assert_eq!(device_lock_ref(&cache, node), 0);
        assert_eq!(cache.protected_size(), 0);
        assert_eq!(cache.evictable_size(), 12);
        // Host eviction now reclaims the coexisting copy and keeps device KV.
        let outcome = evict_host(&mut cache, 3);
        assert_eq!(
            outcome.host_events,
            host.iter()
                .copied()
                .map(HostPageEvent::Freed)
                .collect::<Vec<_>>()
        );
        assert_eq!(host_pages(&cache, node), None);
        assert_eq!(cache.collect_path_pages(node), device);
        assert_eq!(cache.match_prefix_hashes(&key), (12, node));
        assert_pool_accounting(&cache);
    }

    #[test]
    fn load_back_single_page_reclaims_with_one_free_event() {
        let mut cache = RadixCache::new_write_back(32, 16, 2);
        let key = keys(&[5]);
        let (node, host) = demoted(&mut cache, &key);
        let matched = cache.match_prefix_with_host_and_lock(&key);
        let loaded = expect_loaded(cache.load_back(matched.host_hit.unwrap(), None));
        cache.dec_lock_ref(loaded.last_node);
        cache.finish_load_back(loaded.handle);

        let outcome = evict_host(&mut cache, 1);

        assert_eq!(outcome.host_events, vec![HostPageEvent::Freed(host[0])]);
        assert_eq!(cache.collect_path_pages(node), vec![loaded.pages[0].1]);
        assert_pool_accounting(&cache);
    }

    #[test]
    fn load_back_below_a_device_prefix_extends_the_request_prefix() {
        let mut cache = RadixCache::new_write_back(24, 4, 4);
        let prefix = insert_pages(&mut cache, &keys(&[1])).unwrap();
        let key = keys(&[1, 2, 3, 4]);
        let leaf = insert_pages(&mut cache, &key).unwrap();
        cache.inc_lock_ref(prefix);
        cache.evict(12);
        cache.dec_lock_ref(prefix);
        let prefix_page = cache.collect_path_pages(prefix);

        let matched = cache.match_prefix_with_host_and_lock(&key);
        assert_eq!((matched.device_tokens, matched.last_node), (4, prefix));
        assert_eq!(matched.host_tokens(), 12);
        let loaded = expect_loaded(cache.load_back(matched.host_hit.unwrap(), None));

        assert_eq!(loaded.last_node, leaf);
        assert_eq!(device_lock_ref(&cache, leaf), 2);
        let mut expected = prefix_page;
        expected.extend(loaded.pages.iter().map(|&(_, d)| d));
        assert_eq!(cache.collect_path_pages_through(leaf, 16), expected);
        cache.dec_lock_ref(loaded.last_node);
        cache.finish_load_back(loaded.handle);
        assert_eq!(device_lock_ref(&cache, prefix), 0);
        assert_eq!(cache.protected_size(), 0);
        assert_pool_accounting(&cache);
    }

    #[test]
    fn load_back_under_device_pressure_writes_back_other_leaves() {
        let mut cache = RadixCache::new_write_back(16, 4, 8);
        let key = keys(&[1, 2, 3]);
        let (node, _) = demoted(&mut cache, &key);
        let other = insert_pages(&mut cache, &keys(&[7, 8, 9])).unwrap();
        let other_device = cache.collect_path_pages(other);
        assert_eq!(cache.available_tokens(), 4);

        let matched = cache.match_prefix_with_host_and_lock(&key);
        let loaded = expect_loaded(cache.load_back(matched.host_hit.unwrap(), None));

        assert_eq!(loaded.eviction.device_pages, other_device);
        assert_eq!(loaded.eviction.evicted_tokens, 12);
        assert_eq!(
            loaded
                .eviction
                .stored_host()
                .map(|(_, d)| d)
                .collect::<Vec<_>>(),
            other_device
        );
        assert!(host_pages(&cache, other).is_some());
        assert_eq!(loaded.tokens, 12);
        assert_eq!(cache.match_prefix_hashes(&key), (12, node));
        assert_pool_accounting_with_loads(&cache, &[&loaded.handle]);
        cache.dec_lock_ref(loaded.last_node);
        cache.finish_load_back(loaded.handle);
        assert_pool_accounting(&cache);
    }

    #[test]
    fn stale_host_hit_is_refused() {
        let mut cache = RadixCache::new_write_back(16, 4, 4);
        let key = keys(&[1, 2, 3]);
        demoted(&mut cache, &key);
        let matched = cache.match_prefix_with_host_and_lock(&key);
        cache.dec_lock_ref(matched.last_node);
        evict_host(&mut cache, 3);
        cache.inc_lock_ref(matched.last_node);

        let result = cache.load_back(matched.host_hit.unwrap(), None);

        assert!(matches!(
            result,
            LoadBackResult::Refused {
                reason: LoadBackRefusal::Stale,
                ..
            }
        ));
        cache.dec_lock_ref(matched.last_node);
        assert_pool_accounting(&cache);
    }

    #[test]
    fn host_page_pool_recycles_freed_pages() {
        let mut pool = HostPagePool::new(3);
        assert!(pool.allocate_pages(4).is_none());
        let first = pool.allocate_pages(2).unwrap();
        assert_eq!(first, vec![HostPageId(0), HostPageId(1)]);
        assert_eq!(pool.available_pages(), 1);
        pool.free_pages(&first[..1]);
        assert_eq!(
            pool.allocate_pages(2).unwrap(),
            vec![HostPageId(0), HostPageId(2)]
        );
        assert_eq!(pool.available_pages(), 0);
        assert!(pool.allocate_pages(1).is_none());
        assert_eq!(pool.total_pages(), 3);
    }

    #[test]
    #[should_panic(expected = "never allocated")]
    fn host_page_pool_rejects_foreign_pages() {
        HostPagePool::new(2).free_pages(&[HostPageId(0)]);
    }

    #[test]
    fn query_methods_report_physical_token_capacity() {
        let cache = RadixCache::new(100, 4);
        assert_eq!(cache.available_tokens(), 100);
        assert_eq!(cache.total_tokens(), 100);
        assert_eq!(cache.root(), 0);
    }
}
