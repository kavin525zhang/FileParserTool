import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from chunker.splitter import (
    Chunk, 
    ChildChunk, 
    SplitterConfig,
    split_text,
    ParentChildResult
)

from chunker.profiler import (
    DocProfile,
    select_strategy,
    profile_document
)

from chunker.validator import ValidationResult
from chunker.heading_splitter import split_by_headings_impl
from chunker.heuristic_splitter import split_by_heuristics_impl
from chunker.tokens import (
    LANG_MIXED,
    chars_for_token_limit
)

# Strategy values for SplitterConfig.strategy
STRATEGY_AUTO = "auto"
STRATEGY_HEADING = "heading"
STRATEGY_HEURISTIC = "heuristic"
STRATEGY_RECURSIVE = "recursive"
STRATEGY_LEGACY = "legacy"

# Strategy tiers (implementation details)
TIER_HEADING = "heading"
TIER_HEURISTIC = "heuristic"
TIER_LEGACY = "legacy"


def get_logger():
    return logging.getLogger(__name__)


def ensure_defaults(cfg: SplitterConfig) -> SplitterConfig:
    """
    Fill in zero-value config fields with sane defaults.
    Mirrors buildSplitterConfig in internal/application/service/knowledge.go
    so direct callers of this package get the same numbers.

    When cfg.token_limit is set, chunk_size is clamped to the character budget
    that fits within that token limit (with a 10% safety factor). This makes
    chunks safe for embedding APIs that have hard token caps.
    """
    if cfg.chunk_size <= 0:
        cfg.chunk_size = getattr(cfg, 'default_chunk_size', 1000)

    if cfg.chunk_overlap <= 0:
        cfg.chunk_overlap = getattr(cfg, 'default_chunk_overlap', 200)

    if not cfg.separators:
        cfg.separators = ["\n\n", "\n", "。"]

    if cfg.token_limit > 0:
        lang = LANG_MIXED
        if cfg.languages:
            lang = cfg.languages[0]

        char_budget = chars_for_token_limit(cfg.token_limit, lang)
        if char_budget > 0 and (cfg.chunk_size == 0 or char_budget < cfg.chunk_size):
            cfg.chunk_size = char_budget

    # Guard against pathological overlap configurations: if overlap exceeds
    # half of chunk_size, almost every chunk is duplicate content. Cap it at
    # chunk_size/2 so overlap stays a useful smoothing band rather than a
    # near-clone of the previous chunk.
    if cfg.chunk_overlap > (cfg.chunk_size // 2) and cfg.chunk_size > 0:
        cfg.chunk_overlap = cfg.chunk_size // 2

    return cfg

def validate_chunks(chunks: List[Chunk], total_chars: int, chunk_size: int) -> 'ValidationResult':
    """
    Validate that the chunks meet the required criteria.
    This is a placeholder function - the actual implementation would check
    that chunks are properly sized, non-overlapping (except for the specified
    overlap), etc.
    """
    # Placeholder implementation
    return ValidationResult(ok=True, reason="")


def merge_breadcrumbs(parent: str, child: str) -> str:
    """
    Combine the parent and child heading breadcrumbs into a
    single ContextHeader. When the child re-runs heading detection on parent
    content, its first breadcrumb line typically duplicates the parent's last
    line (the parent's leading heading sits at the top of the child's input);
    drop that duplicate so the embedding context isn't redundant.
    """
    if not parent:
        return child
    if not child:
        return parent

    parent_lines = parent.split('\n')
    child_lines = child.split('\n')

    if (parent_lines and child_lines and
        parent_lines[-1].strip() == child_lines[0].strip()):
        child_lines = child_lines[1:]

    if not child_lines:
        return parent

    return parent + "\n" + "\n".join(child_lines)


def run_tier(tier: str, text: str, cfg: SplitterConfig, profile: Optional[DocProfile]) -> List[Chunk]:
    """
    Dispatch the splitter implementation for the given tier.
    split_by_headings / split_by_heuristics are package-level vars overridden
    from heading_splitter.py / heuristic_splitter.py via imports; legacy
    runs split_text. The default branch is defensive for future
    StrategyTier additions.

    profile may be None when the caller did not run the document profiler
    (explicit non-auto strategies skip profiling); splitters that need a
    profile compute one on demand.
    """
    if tier == TIER_HEADING:
        return split_by_headings_impl(text, cfg, profile)
    elif tier == TIER_HEURISTIC:
        return split_by_heuristics_impl(text, cfg, profile)
    elif tier == TIER_LEGACY:
        return split_text(text, cfg)

    return split_text(text, cfg)


def resolve_chain_with_profile(text: str, cfg: SplitterConfig) -> Tuple[List[str], Optional[DocProfile]]:
    """
    Return the strategy chain to attempt and, when
    the chain was selected by the profiler (auto strategy), the DocProfile
    that drove the selection. Profile is None for explicit non-auto strategies
    so callers don't pay for an unused profiling pass.
    """
    if cfg.strategy == STRATEGY_HEADING:
        return [TIER_HEADING, TIER_LEGACY], None
    elif cfg.strategy == STRATEGY_HEURISTIC:
        return [TIER_HEURISTIC, TIER_LEGACY], None
    elif cfg.strategy == STRATEGY_RECURSIVE:
        # "recursive" is a public-API alias for "legacy": both invoke
        # split_text. Kept for backwards compatibility with stored configs.
        return [TIER_LEGACY], None
    elif cfg.strategy in (STRATEGY_LEGACY, ""):
        # Empty == legacy preserves backwards compatibility with stored
        # ChunkingConfig rows that pre-date the Strategy field.
        return [TIER_LEGACY], None
    elif cfg.strategy == STRATEGY_AUTO:
        profile = profile_document(text)
        return select_strategy(profile), profile
    else:
        # Default case for unknown strategies
        profile = profile_document(text)
        return select_strategy(profile), profile


def split(text: str, cfg: SplitterConfig) -> List[Chunk]:
    """
    Split chunks text using the strategy configured in cfg. When cfg.strategy
    is empty or "auto" the document profiler picks the tier. The function
    always returns a non-nil result: on tier failure the chain falls through
    to the legacy splitter, which is the original Tier 3 implementation.

    Hot path: avoids the Diagnostics struct allocation that
    split_with_diagnostics performs (matters in split_parent_child where
    split is called once per parent).
    """
    if not text:
        return []

    cfg = ensure_defaults(cfg)
    chain, profile = resolve_chain_with_profile(text, cfg)
    total_chars = len(text)  # Note: In Go, this was len([]rune(text)) for Unicode counting

    last_out = []
    for i, tier in enumerate(chain):
        out = run_tier(tier, text, cfg, profile)
        v = validate_chunks(out, total_chars, cfg.chunk_size)
        if v.ok:
            return out
        else:
            get_logger().debug(f"chunker: tier {tier} rejected: {v.reason}")

        if tier == TIER_LEGACY and i == len(chain) - 1:
            last_out = out
    if last_out:
        return last_out

    return split_text(text, cfg)

@dataclass
class TierRejection:
    """
    Records why a tier was rejected by the validator and the
    chain advanced to the next tier. Surfaced by split_with_diagnostics for
    the debug/preview endpoint.
    """
    tier: str
    reason: str

@dataclass
class Diagnostics:
    """
    Captures which tier produced the returned chunks plus the
    chain that was attempted, any rejected tiers along the way, and the
    document profile that drove tier selection.

    Useful for surfacing in a debug UI; not produced by the normal split
    path. The JSON shape is part of the public preview-endpoint API — keep
    field names stable.
    """
    selected_tier: str = TIER_LEGACY  # Default to legacy so an empty diag never carries the zero string
    tier_chain: List[str] = field(default_factory=list)
    rejected: List[TierRejection] = field(default_factory=list)
    # Profile is set when the auto strategy resolved the chain via the
    # document profiler. None when an explicit Strategy bypassed profiling.
    profile: Optional[DocProfile] = None


def split_with_diagnostics(text: str, cfg: SplitterConfig) -> Tuple[List[Chunk], Diagnostics]:
    """
    The same as split but also returns the diagnostic trace (selected tier, full chain, rejection reasons,
    profile when available). Use this for the chunker preview endpoint
    where the caller wants to know which tier won and why others lost.
    """
    # Default selected tier to legacy so an empty diag never carries the
    # zero string — that would render as a blank tag in the debug UI.
    diag = Diagnostics(selected_tier=TIER_LEGACY)

    if not text:
        return [], diag

    cfg = ensure_defaults(cfg)
    chain, profile = resolve_chain_with_profile(text, cfg)
    print(f"chain:{chain}, profile:{profile}")
    diag.tier_chain = chain
    diag.profile = profile
    total_chars = len(text)

    last_out = []
    last_tier = None

    for i, tier in enumerate(chain):
        out = run_tier(tier, text, cfg, profile)
        v = validate_chunks(out, total_chars, cfg.chunk_size)

        if v.ok:
            diag.selected_tier = tier
            return out, diag

        diag.rejected.append(TierRejection(tier=tier, reason=v.reason))
        get_logger().debug(f"chunker: tier {tier} rejected: {v.reason}")

        if tier == TIER_LEGACY and i == len(chain) - 1:
            last_out = out
            last_tier = tier

    if last_out:
        diag.selected_tier = last_tier
        return last_out, diag

    # Defensive last-ditch fallback.
    return split_text(text, cfg), diag

def split_parent_child(text: str, parent_cfg: SplitterConfig, child_cfg: SplitterConfig) -> ParentChildResult:
    """
    The strategy-aware analog of split_text_parent_child.
    It runs the tier selector for parent splitting, then re-splits each
    parent into children with the small-chunk config.

    Child splitting honours child_cfg.strategy. If it is empty/auto and a
    parent has its own internal structure (sub-headings, numbered sub-
    sections), the appropriate tier picks it up so child chunks carry a
    finer-grained breadcrumb than the parent's. Re-profiling each parent
    is bounded by O(sum(parent_size)) ≈ O(N) total, which is the same
    order as the original parent profiling pass.
    """
    if not text:
        return ParentChildResult()

    parent_cfg = ensure_defaults(parent_cfg)
    child_cfg = ensure_defaults(child_cfg)

    parents = split(text, parent_cfg)
    if not parents:
        return ParentChildResult()

    new_parents = []
    children = []
    child_seq = 0

    for parent in parents:
        subs = split(parent.content, child_cfg)

        parent_index = -1
        if len(subs) > 1 or (len(subs) == 1 and subs[0].content != parent.content):
            parent_index = len(new_parents)
            new_parents.append(parent)

        for sub in subs:
            children.append(ChildChunk(content=sub.content,
                                       seq = child_seq, 
                                       start = sub.start + parent.start,
                                       end = sub.end + parent.start,
                                       context_header = merge_breadcrumbs(parent.context_header, sub.context_header),
                                       parent_index=parent_index))
            child_seq += 1

    return ParentChildResult(parents=new_parents, children=children)