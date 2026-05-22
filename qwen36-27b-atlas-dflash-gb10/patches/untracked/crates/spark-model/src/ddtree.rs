// SPDX-License-Identifier: AGPL-3.0-only

//! Native DDTree core for the fixed Luce DFlash path.
//!
//! This module intentionally contains no Python/vLLM compatibility surface.
//! It is the host-side semantic core used before wiring tree verification into
//! Atlas GPU kernels: fixed-parameter validation, reference heap construction,
//! packed child maps, equality walk, tree-bias construction, and single-use
//! per-step payload ownership.

use anyhow::{Result, bail};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

pub const LUCE_TREE_BUDGET: usize = 18;
pub const LUCE_QUERY_BUDGET: usize = 15;
pub const LUCE_PARENT_WIDTH: usize = LUCE_QUERY_BUDGET + 1;
pub const LUCE_DDTREE_TEMP: f32 = 1.05;
pub const LUCE_MAX_CTX: usize = 1024;
pub const LUCE_FA_WINDOW: usize = 2048;
pub const LUCE_DEFAULT_TOPK_WIDTH: usize = LUCE_TREE_BUDGET;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheDType {
    F16,
    Bf16,
    Fp8,
    Nvfp4,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LuceFixedDdTreeConfig {
    pub tree_budget: usize,
    pub query_budget: usize,
    pub ddtree_temp: f32,
    pub max_ctx: usize,
    pub fa_window: usize,
    pub key_cache_dtype: CacheDType,
    pub value_cache_dtype: CacheDType,
    pub prefix_cache_slots: usize,
    pub prefill_cache_slots: usize,
    pub ignore_eos_stop: bool,
    pub no_chain_seed: bool,
    pub candidate_posterior: bool,
    pub force_accept: bool,
}

impl LuceFixedDdTreeConfig {
    #[must_use]
    pub fn fixed() -> Self {
        Self {
            tree_budget: LUCE_TREE_BUDGET,
            query_budget: LUCE_QUERY_BUDGET,
            ddtree_temp: LUCE_DDTREE_TEMP,
            max_ctx: LUCE_MAX_CTX,
            fa_window: LUCE_FA_WINDOW,
            key_cache_dtype: CacheDType::F16,
            value_cache_dtype: CacheDType::F16,
            prefix_cache_slots: 0,
            prefill_cache_slots: 0,
            ignore_eos_stop: true,
            no_chain_seed: true,
            candidate_posterior: false,
            force_accept: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.tree_budget != LUCE_TREE_BUDGET {
            bail!("Luce DDTree requires tree_budget={LUCE_TREE_BUDGET}");
        }
        if self.query_budget != LUCE_QUERY_BUDGET {
            bail!("Luce DDTree requires query_budget={LUCE_QUERY_BUDGET}");
        }
        if self.query_budget + 1 != LUCE_PARENT_WIDTH {
            bail!("Luce DDTree requires parent_width={LUCE_PARENT_WIDTH}");
        }
        if (self.ddtree_temp - LUCE_DDTREE_TEMP).abs() > f32::EPSILON {
            bail!("Luce DDTree requires ddtree_temp={LUCE_DDTREE_TEMP}");
        }
        if self.max_ctx != LUCE_MAX_CTX {
            bail!("Luce DDTree requires max_ctx={LUCE_MAX_CTX}");
        }
        if self.fa_window != LUCE_FA_WINDOW {
            bail!("Luce DDTree requires fa_window={LUCE_FA_WINDOW}");
        }
        if self.key_cache_dtype != CacheDType::F16 || self.value_cache_dtype != CacheDType::F16 {
            bail!("Luce DDTree requires ctk/ctv f16 analogs");
        }
        if self.prefix_cache_slots != 0 || self.prefill_cache_slots != 0 {
            bail!("Luce DDTree requires prefix and prefill caches disabled");
        }
        if !self.ignore_eos_stop {
            bail!("Luce DDTree requires ignore_eos_stop=true");
        }
        if !self.no_chain_seed {
            bail!("Luce DDTree requires ddtree-no-chain-seed");
        }
        if self.candidate_posterior {
            bail!("candidate posterior is forbidden in genuine DDTree");
        }
        if self.force_accept {
            bail!("force accept is forbidden in genuine DDTree");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildEdge {
    pub token_id: u32,
    pub node_id: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DdTree {
    /// Node token IDs excluding root. Tree index `i` maps to
    /// `node_token_ids[i - 1]` for `i > 0`.
    pub node_token_ids: Vec<u32>,
    /// Node depths including root at index 0. Root depth is 0.
    pub depths: Vec<u8>,
    /// Parents including root at index 0. Root parent is -1.
    pub parents: Vec<i16>,
    /// Packed child ranges per tree node. Range indexes into `child_edges`.
    pub child_ranges: Vec<(usize, usize)>,
    pub child_edges: Vec<ChildEdge>,
    /// Ancestor-only visibility matrix including root.
    pub visibility: Vec<Vec<bool>>,
}

impl DdTree {
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_token_ids.len()
    }

    pub fn child_with_token(&self, parent: usize, token_id: u32) -> Option<usize> {
        let (start, end) = self.child_ranges.get(parent).copied()?;
        self.child_edges[start..end]
            .iter()
            .find(|edge| edge.token_id == token_id)
            .map(|edge| edge.node_id)
    }

    #[must_use]
    pub fn has_branching(&self) -> bool {
        let mut child_counts = vec![0usize; self.node_count() + 1];
        for &parent in self.parents.iter().skip(1) {
            if parent >= 0 {
                child_counts[parent as usize] += 1;
            }
        }
        child_counts.into_iter().any(|count| count > 1)
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.node_count() + 1;
        if self.parents.len() != n || self.depths.len() != n || self.child_ranges.len() != n {
            bail!("DDTree metadata width mismatch");
        }
        if self.parents[0] != -1 || self.depths[0] != 0 {
            bail!("DDTree root metadata is invalid");
        }
        for node in 1..n {
            let parent = self.parents[node];
            if parent < 0 || parent as usize >= node {
                bail!("DDTree parent must be parent-before-child");
            }
            if self.depths[node] != self.depths[parent as usize] + 1 {
                bail!("DDTree depth must equal parent depth plus one");
            }
        }
        for row in 0..n {
            if self.visibility[row].len() != n {
                bail!("DDTree visibility row width mismatch");
            }
            if !self.visibility[row][row] {
                bail!("DDTree row must see itself");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DdTreeDraft {
    pub config: LuceFixedDdTreeConfig,
    pub tree: DdTree,
    pub top_token_ids: Vec<Vec<u32>>,
    pub top_log_probs: Vec<Vec<f32>>,
    pub topk_width: usize,
}

impl DdTreeDraft {
    pub fn new(
        config: LuceFixedDdTreeConfig,
        top_token_ids: Vec<Vec<u32>>,
        top_log_probs: Vec<Vec<f32>>,
        topk_width: usize,
    ) -> Result<Self> {
        config.validate()?;
        if top_token_ids.len() < config.query_budget {
            bail!("DDTree draft rows smaller than fixed Luce query budget");
        }
        if top_log_probs.len() != top_token_ids.len() {
            bail!("DDTree draft token/logprob row mismatch");
        }
        if topk_width < 2 {
            bail!("DDTree requires at least two candidates per row");
        }
        let tree = build_ddtree_from_topk(
            &top_token_ids[..config.query_budget],
            &top_log_probs[..config.query_budget],
            config.tree_budget,
        )?;
        if tree.node_count() != config.tree_budget {
            bail!("DDTree did not fill the fixed Luce tree budget");
        }
        if !tree.has_branching() {
            bail!("DDTree must be genuinely non-flat");
        }
        Ok(Self {
            config,
            tree,
            top_token_ids,
            top_log_probs,
            topk_width,
        })
    }
}

#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn extract_topk_logprobs_from_bf16_logits(
    logits_bf16_le: &[u8],
    rows: usize,
    vocab_size: usize,
    topk_width: usize,
    temperature: f32,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<f32>>)> {
    if rows == 0 || vocab_size == 0 || topk_width == 0 {
        bail!("top-k extraction requires non-empty rows, vocab, and width");
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        bail!("top-k extraction requires positive finite temperature");
    }
    let expected = rows
        .checked_mul(vocab_size)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("logits byte size overflow"))?;
    if logits_bf16_le.len() < expected {
        bail!("logits buffer smaller than rows * vocab * bf16");
    }

    let width = topk_width.min(vocab_size);
    let mut all_ids = Vec::with_capacity(rows);
    let mut all_log_probs = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_start = row * vocab_size * 2;
        let row_bytes = &logits_bf16_le[row_start..row_start + vocab_size * 2];
        let mut max_logit = f32::NEG_INFINITY;
        let mut top_values: Vec<(u32, f32)> = Vec::with_capacity(width);

        for token_id in 0..vocab_size {
            let offset = token_id * 2;
            let bits = u16::from_le_bytes([row_bytes[offset], row_bytes[offset + 1]]);
            let value = bf16_to_f32(bits) / temperature;
            max_logit = max_logit.max(value);
            insert_top_value(&mut top_values, width, token_id as u32, value);
        }
        if !max_logit.is_finite() {
            bail!("logits row {row} has no finite values");
        }

        let mut exp_sum = 0.0f32;
        for token_id in 0..vocab_size {
            let offset = token_id * 2;
            let bits = u16::from_le_bytes([row_bytes[offset], row_bytes[offset + 1]]);
            let value = bf16_to_f32(bits) / temperature;
            exp_sum += (value - max_logit).exp();
        }
        let log_z = max_logit + exp_sum.ln();
        let ids = top_values.iter().map(|(token_id, _)| *token_id).collect();
        let log_probs = top_values.iter().map(|(_, value)| *value - log_z).collect();
        all_ids.push(ids);
        all_log_probs.push(log_probs);
    }
    Ok((all_ids, all_log_probs))
}

fn insert_top_value(top_values: &mut Vec<(u32, f32)>, width: usize, token_id: u32, value: f32) {
    if !value.is_finite() {
        return;
    }
    let insert_at = top_values
        .iter()
        .position(|&(existing_id, existing_value)| {
            value
                .total_cmp(&existing_value)
                .then_with(|| existing_id.cmp(&token_id))
                .is_gt()
        })
        .unwrap_or(top_values.len());
    if insert_at < width {
        top_values.insert(insert_at, (token_id, value));
        top_values.truncate(width);
    }
}

pub fn build_luce_ddtree_draft_from_bf16_logits(
    logits_bf16_le: &[u8],
    rows: usize,
    vocab_size: usize,
) -> Result<DdTreeDraft> {
    let config = LuceFixedDdTreeConfig::fixed();
    config.validate()?;
    if rows < config.query_budget {
        bail!("DFlash logits rows smaller than fixed Luce DDTree query budget");
    }
    let (top_token_ids, top_log_probs) = extract_topk_logprobs_from_bf16_logits(
        logits_bf16_le,
        config.query_budget,
        vocab_size,
        LUCE_DEFAULT_TOPK_WIDTH,
        config.ddtree_temp,
    )?;
    DdTreeDraft::new(
        config,
        top_token_ids,
        top_log_probs,
        LUCE_DEFAULT_TOPK_WIDTH.min(vocab_size),
    )
}

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    log_weight: f32,
    parent_index: usize,
    depth: usize,
    rank: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.log_weight == other.log_weight
            && self.parent_index == other.parent_index
            && self.depth == other.depth
            && self.rank == other.rank
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.log_weight
            .partial_cmp(&other.log_weight)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.rank.cmp(&self.rank))
            .then_with(|| other.parent_index.cmp(&self.parent_index))
    }
}

pub fn build_ddtree_from_topk(
    top_token_ids: &[Vec<u32>],
    top_log_probs: &[Vec<f32>],
    budget: usize,
) -> Result<DdTree> {
    if budget == 0 || top_token_ids.is_empty() {
        return Ok(empty_tree());
    }
    if top_token_ids.len() != top_log_probs.len() {
        bail!("top token/logprob depth mismatch");
    }
    let first_width = top_token_ids[0].len();
    if first_width == 0 {
        return Ok(empty_tree());
    }
    for (row, (ids, probs)) in top_token_ids.iter().zip(top_log_probs).enumerate() {
        if ids.is_empty() || ids.len() != probs.len() {
            bail!("top-k row {row} has invalid token/logprob widths");
        }
    }

    let topk = budget.min(first_width);
    let depth_limit = top_token_ids.len();
    let mut heap = std::collections::BinaryHeap::new();
    heap.push(HeapEntry {
        log_weight: top_log_probs[0][0],
        parent_index: 0,
        depth: 1,
        rank: 0,
    });

    let mut node_token_ids = Vec::with_capacity(budget);
    let mut depths = vec![0u8];
    let mut parents = vec![-1i16];
    let mut child_maps: Vec<BTreeMap<u32, usize>> = vec![BTreeMap::new()];

    while let Some(entry) = heap.pop() {
        if node_token_ids.len() >= budget {
            break;
        }
        if entry.depth == 0
            || entry.depth > depth_limit
            || entry.rank >= top_token_ids[entry.depth - 1].len()
        {
            bail!("heap entry out of top-k bounds");
        }

        let token_id = top_token_ids[entry.depth - 1][entry.rank];
        let current_index = node_token_ids.len() + 1;
        node_token_ids.push(token_id);
        depths.push(u8::try_from(entry.depth)?);
        parents.push(i16::try_from(entry.parent_index)?);
        child_maps.push(BTreeMap::new());
        child_maps[entry.parent_index].insert(token_id, current_index);

        if entry.rank + 1 < topk && entry.rank + 1 < top_token_ids[entry.depth - 1].len() {
            let row_probs = &top_log_probs[entry.depth - 1];
            let sibling_log_weight =
                entry.log_weight - row_probs[entry.rank] + row_probs[entry.rank + 1];
            heap.push(HeapEntry {
                log_weight: sibling_log_weight,
                parent_index: entry.parent_index,
                depth: entry.depth,
                rank: entry.rank + 1,
            });
        }

        if entry.depth < depth_limit {
            heap.push(HeapEntry {
                log_weight: entry.log_weight + top_log_probs[entry.depth][0],
                parent_index: current_index,
                depth: entry.depth + 1,
                rank: 0,
            });
        }
    }

    pack_tree(node_token_ids, depths, parents, child_maps)
}

fn empty_tree() -> DdTree {
    DdTree {
        node_token_ids: Vec::new(),
        depths: vec![0],
        parents: vec![-1],
        child_ranges: vec![(0, 0)],
        child_edges: Vec::new(),
        visibility: vec![vec![true]],
    }
}

fn pack_tree(
    node_token_ids: Vec<u32>,
    depths: Vec<u8>,
    parents: Vec<i16>,
    child_maps: Vec<BTreeMap<u32, usize>>,
) -> Result<DdTree> {
    let n = node_token_ids.len() + 1;
    let mut child_ranges = Vec::with_capacity(n);
    let mut child_edges = Vec::new();
    for children in child_maps {
        let start = child_edges.len();
        for (token_id, node_id) in children {
            child_edges.push(ChildEdge { token_id, node_id });
        }
        child_ranges.push((start, child_edges.len()));
    }

    let mut visibility = vec![vec![false; n]; n];
    visibility[0][0] = true;
    for node in 1..n {
        let parent = usize::try_from(parents[node])?;
        for col in 0..node {
            visibility[node][col] = visibility[parent][col];
        }
        visibility[node][node] = true;
    }

    let tree = DdTree {
        node_token_ids,
        depths,
        parents,
        child_ranges,
        child_edges,
        visibility,
    };
    tree.validate()?;
    Ok(tree)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkResult {
    pub accepted_indices: Vec<usize>,
    pub bonus_token: u32,
}

pub fn equality_walk(tree: &DdTree, posterior_tokens: &[u32]) -> Result<WalkResult> {
    let n = tree.node_count() + 1;
    if posterior_tokens.len() < n {
        bail!("posterior token count must cover root plus tree nodes");
    }
    let mut accepted_indices = vec![0usize];
    let mut current = 0usize;
    let mut next_token = posterior_tokens[current];

    while let Some(child) = tree.child_with_token(current, next_token) {
        accepted_indices.push(child);
        current = child;
        next_token = posterior_tokens[current];
    }

    Ok(WalkResult {
        accepted_indices,
        bonus_token: next_token,
    })
}

pub fn build_tree_bias_2d(tree: &DdTree, width: usize) -> Result<Vec<Vec<f32>>> {
    let n = tree.node_count() + 1;
    if width < n {
        bail!("tree bias width smaller than tree");
    }
    let mut bias = vec![vec![f32::NEG_INFINITY; width]; width];
    for row in 0..n {
        for col in 0..n {
            if tree.visibility[row][col] {
                bias[row][col] = 0.0;
            }
        }
    }
    Ok(bias)
}

pub fn build_tree_bias_3d(trees: &[DdTree], width: usize) -> Result<Vec<Vec<Vec<f32>>>> {
    let mut batch = Vec::with_capacity(trees.len());
    for tree in trees {
        batch.push(build_tree_bias_2d(tree, width)?);
    }
    Ok(batch)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestId(pub String);

#[derive(Debug, Clone)]
pub struct DdTreeStep {
    pub request_id: RequestId,
    pub scheduler_step: u64,
    pub config: LuceFixedDdTreeConfig,
    pub tree: DdTree,
    consumed: bool,
}

impl DdTreeStep {
    pub fn new(
        request_id: RequestId,
        scheduler_step: u64,
        config: LuceFixedDdTreeConfig,
        tree: DdTree,
    ) -> Result<Self> {
        config.validate()?;
        tree.validate()?;
        if tree.node_count() > config.tree_budget {
            bail!("DDTree payload exceeds fixed Luce tree budget");
        }
        Ok(Self {
            request_id,
            scheduler_step,
            config,
            tree,
            consumed: false,
        })
    }

    pub fn mark_consumed(&mut self, request_id: &RequestId, scheduler_step: u64) -> Result<()> {
        if &self.request_id != request_id {
            bail!("DDTree payload request id mismatch");
        }
        if self.scheduler_step != scheduler_step {
            bail!("DDTree payload scheduler step mismatch");
        }
        if self.consumed {
            bail!("DDTree payload consumed twice");
        }
        self.consumed = true;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct DdTreePayloadRegistry {
    steps: HashMap<(RequestId, u64), DdTreeStep>,
}

impl DdTreePayloadRegistry {
    pub fn publish(&mut self, step: DdTreeStep) -> Result<()> {
        let key = (step.request_id.clone(), step.scheduler_step);
        if self.steps.contains_key(&key) {
            bail!("DDTree payload already exists for request/step");
        }
        self.steps.insert(key, step);
        Ok(())
    }

    pub fn consume(&mut self, request_id: &RequestId, scheduler_step: u64) -> Result<DdTreeStep> {
        let key = (request_id.clone(), scheduler_step);
        let mut step = self
            .steps
            .remove(&key)
            .ok_or_else(|| anyhow::anyhow!("DDTree payload missing for request/step"))?;
        step.mark_consumed(request_id, scheduler_step)?;
        Ok(step)
    }

    pub fn clear_request(&mut self, request_id: &RequestId) {
        self.steps.retain(|(rid, _), _| rid != request_id);
    }

    pub fn clear_all(&mut self) {
        self.steps.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> DdTree {
        let ids = vec![vec![10, 11], vec![20, 21], vec![30, 31]];
        let probs = vec![vec![-0.1, -2.0], vec![-0.1, -2.0], vec![-0.1, -2.0]];
        build_ddtree_from_topk(&ids, &probs, 4).expect("tree")
    }

    #[test]
    fn luce_fixed_config_accepts_only_fixed_values() {
        LuceFixedDdTreeConfig::fixed()
            .validate()
            .expect("fixed config");

        let mut bad = LuceFixedDdTreeConfig::fixed();
        bad.candidate_posterior = true;
        assert!(bad.validate().is_err());

        let mut bad = LuceFixedDdTreeConfig::fixed();
        bad.force_accept = true;
        assert!(bad.validate().is_err());

        let mut bad = LuceFixedDdTreeConfig::fixed();
        bad.no_chain_seed = false;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn heap_builder_creates_parent_before_child_nonflat_tree() {
        let tree = sample_tree();
        assert_eq!(tree.node_count(), 4);
        assert_eq!(tree.parents[0], -1);
        for node in 1..=tree.node_count() {
            assert!((tree.parents[node] as usize) < node);
        }
        assert!(tree.parents.iter().skip(1).filter(|&&p| p == 0).count() >= 2);
        assert_eq!(tree.depths[1], 1);
        assert_eq!(tree.node_token_ids[0], 10);
    }

    #[test]
    fn visibility_is_ancestor_only() {
        let tree = sample_tree();
        let branch = (1..=tree.node_count())
            .find(|&node| tree.parents[node] == 0 && node != 1)
            .expect("branch child of root");
        assert!(tree.visibility[branch][0]);
        assert!(tree.visibility[branch][branch]);
        assert!(!tree.visibility[branch][1]);
        assert!(!tree.visibility[1][branch]);
    }

    #[test]
    fn equality_walk_follows_child_maps_and_returns_target_miss() {
        let tree = sample_tree();
        let posterior = vec![10, 20, 999, 0, 0];
        let walked = equality_walk(&tree, &posterior).expect("walk");
        assert_eq!(walked.accepted_indices, vec![0, 1, 2]);
        assert_eq!(walked.bonus_token, 999);
    }

    #[test]
    fn tree_bias_3d_is_per_request_and_padded() {
        let tree = sample_tree();
        let batch = build_tree_bias_3d(&[tree.clone(), tree], LUCE_PARENT_WIDTH).expect("bias");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), LUCE_PARENT_WIDTH);
        assert_eq!(batch[0][0].len(), LUCE_PARENT_WIDTH);
        assert_eq!(batch[0][0][0], 0.0);
        assert_eq!(batch[0][0][1], f32::NEG_INFINITY);
    }

    #[test]
    fn bf16_topk_logprobs_are_full_vocab_normalized() {
        let rows = 1usize;
        let vocab = 4usize;
        let logits = [1.0f32, 3.0, 2.0, -1.0];
        let mut bytes = Vec::with_capacity(rows * vocab * 2);
        for value in logits {
            let bits = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }

        let (ids, log_probs) =
            extract_topk_logprobs_from_bf16_logits(&bytes, rows, vocab, 2, 1.0).expect("topk");
        assert_eq!(ids, vec![vec![1, 2]]);

        let log_z =
            3.0 + ((1.0f32 - 3.0).exp() + 1.0 + (2.0f32 - 3.0).exp() + (-4.0f32).exp()).ln();
        assert!((log_probs[0][0] - (3.0 - log_z)).abs() < 1e-5);
        assert!((log_probs[0][1] - (2.0 - log_z)).abs() < 1e-5);
    }

    #[test]
    fn luce_draft_from_logits_enforces_budget_and_branching() {
        let rows = LUCE_QUERY_BUDGET;
        let vocab = 32usize;
        let mut bytes = Vec::with_capacity(rows * vocab * 2);
        for row in 0..rows {
            for token in 0..vocab {
                let value: f32 = if token == row {
                    10.0
                } else if token == row + 1 {
                    9.0
                } else {
                    0.0
                };
                let bits = (value.to_bits() >> 16) as u16;
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
        }

        let draft = build_luce_ddtree_draft_from_bf16_logits(&bytes, rows, vocab).expect("draft");
        assert_eq!(draft.tree.node_count(), LUCE_TREE_BUDGET);
        assert_eq!(draft.top_token_ids.len(), LUCE_QUERY_BUDGET);
        assert!(draft.tree.has_branching());
    }

    #[test]
    fn payload_registry_is_single_use_and_request_step_owned() {
        let tree = sample_tree();
        let request_id = RequestId("req-a".to_string());
        let step = DdTreeStep::new(request_id.clone(), 7, LuceFixedDdTreeConfig::fixed(), tree)
            .expect("step");

        let mut registry = DdTreePayloadRegistry::default();
        registry.publish(step).expect("publish");
        let consumed = registry.consume(&request_id, 7).expect("consume");
        assert_eq!(consumed.request_id, request_id);
        assert!(
            registry
                .consume(&RequestId("req-a".to_string()), 7)
                .is_err()
        );
    }
}
