import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from splitter import (
    Chunk, 
    ChildChunk, 
    SplitterConfig,
    split_text,
    ParentChildResult
)

from profiler import (
    DocProfile,
    select_strategy,
    profile_document
)

from validator import ValidationResult
from heading_splitter import split_by_headings_impl
from heuristic_splitter import split_by_heuristics_impl
from tokens import (
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


if __name__ == "__main__":
    text = '''# 二、董事履职具体情况

## （一）履行忠实义务

全体董事忠实履行诚信义务，在履职过程中严格保守公司商业秘密，遵守内幕交易相关规定，保证披露信息真实、准确、完整、及时、公平，不存在泄露公司尚未披露信息的行为。各位董事如实向公司告知本职、兼职情况，并保证所任职务与在公司的任职不存在利益冲突。报告期内，公司董事不存在利用董事地位为自己或第三人谋取不正当利益，也不存在为股东利益而损害公司利益的行为。2025 年，公司未发现董事提前公开或泄露公司信息或利用该等信息进行内幕交易的行为，未收到监管机构针对董事违背忠实义务的处罚。

## （二）履行勤勉义务

全体董事根据相关法律法规和公司章程规定，积极参加董事会及相关专门委员会会议，勤勉履行职责。各位董事在会前认真审阅会议资料；会议过程中，各位董事均能就审议议案展开充分讨论，做出独立、专业、客观的判断。部分因公务原因无法亲自出席会议的董事，均能在会前认真审阅相关议案并表达审议意见，按照规定委托其他董事代行表决权。2025 年，公司董事会共召开 8 次会议，其中：现场结合通讯方式召开会议6次，书面传签会议2次。全体董事均达到“亲自出席当年三分之二以上的董事会现场会议”的规定。

## （三）履职专业性

全体董事认真研阅公司提供的各类会议材料及参阅信息，持续深入了解公司经营及管理情况，认真研究重大问题，审慎决策公司战略管理、公司治理、资本管理、风险管理、内控合规、审计监督、激励约束等事项，切实保护公司整体及股东合法权益，维护金融消费者和其他利益相关方利益，为推动公司高质量发展作出富有成效的贡献。2025年，董事会共召开8 次会议，审议议案103项，听取议案33项；董事会专门委员会共召开会议31次，审议议案124项，听取议案46项；独立董事专门会议共召开会议3 次，审议议案16项，听取议案3项。全体董事发挥决策主体核心作用，确保董事会决策科学高效、

依法合规。

## （四）履职独立性与道德水准

全体董事保持履职所需独立性和较高的职业道德准则，不受主要股东和内部人控制或干预，推动公司公平对待全体股东、维护利益相关者的合法权益。在审议关联交易议案时，利益相关董事均严格履行回避义务，未发现各位董事与公司存在利益冲突、违反关联交易和履职回避相关规定的情形，未发现独立董事存在影响独立性的情形。

## （五）履职合规性

全体董事遵守法律法规、监管规定及公司章程，持续规范自身履职行为，依法合规履行相应职责，推动和监督公司守法合规经营。未发现存在未经公司章程规定或者董事会合法授权的前提下，以个人名义代表公司或者董事会行事的情形。
'''
    cfg= SplitterConfig(strategy=STRATEGY_AUTO)
    chunks, diag = split_with_diagnostics(text, cfg)
    print(f"chunks:{chunks}")
    print(f"diag:{diag}")