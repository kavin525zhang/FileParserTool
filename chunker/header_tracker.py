import re
import string
from typing import Dict, List, Tuple, Optional

class HeaderTrackerHook:
    """Defines a pattern pair for detecting contextual headers.

    When start_pattern matches a unit's text, that text becomes an "active header".
    The header stays active until end_pattern matches a subsequent unit.
    """
    def __init__(self, start_pattern: str, end_pattern: str, priority: int):
        self.start_pattern = re.compile(start_pattern, re.IGNORECASE | re.DOTALL)
        self.end_pattern = re.compile(end_pattern, re.IGNORECASE | re.DOTALL)
        self.priority = priority

default_header_hooks = [
    # Markdown table: header row + separator row (e.g. "| A | B |\n| --- | --- |\n")
    # When startPattern matches, that text becomes an active header
    # End when we see an empty/whitespace line or a line that doesn't start with | or whitespace
    HeaderTrackerHook(
        #start_pattern=r'^\s*(?:\|[^|\n]*)+[\r\n]+\s*(?:\|\s*:?-{3,}:?\s*)+\|?[\r\n]+$',
        start_pattern=r'^\s*(?:\|[^|\n]*)+[\r\n]+\s*(?:\|\s*:?-{3,}:?\s*)+\|?[\r\n]+$',
        #end_pattern=r'^\s*$|^\s*[^|\s].*$',
        end_pattern=r'^\s*$|^\s*[^|\s].*$',
        priority=15
    )
]

table_row_pattern = re.compile(r'^\s*(?:\|[^|\n]*)+\|\s*$', re.MULTILINE)

# Priority for the default markdown table hook
markdown_table_hook_priority = 15

class HeaderTracker:
    """Maintains the state of active headers across split units.

    Detects table headers and signals the merge logic to prepend them to subsequent chunks.
    This preserves context when a large Markdown table is split across multiple chunks.
    """
    def __init__(self, hooks: List[HeaderTrackerHook] = None):
        self.hooks = hooks or default_header_hooks
        self.active_headers: Dict[int, str] = {}  # priority -> header text
        self.ended_headers: Dict[int, bool] = {}  # priorities that have been ended
        self.pending_extend: Dict[int, bool] = {}  # headers with empty column names awaiting first data row
        # 当表格行单元以段落换行符结束时，会设置 pendingTableBreak 标志
        # （表格间的空行已被 \n\n 分割操作消耗掉）。表头状态会保持有效，
        # 直到遇到下一个单元，以便我们能够识别出新的表格。
        self.pending_table_break = False
        # header_ended_this_unit 指示 merge_units 在新表格开始时（因列不匹配或 pending_table_break 加上表格行触发）
        # 在当前单元之前进行刷新（flush）。
        self.header_ended_this_unit = False

    # 更新检查会针对头部起始/结束标记拆分文本，并更新内部状态。
    def update(self, split: str) -> None:
        """Check split text for header start/end markers and update internal state."""
        self.header_ended_this_unit = False

        # Handle pending table break from previous unit
        if self.pending_table_break:
            self.pending_table_break = False
            # markdown_table_hook_priority ?
            if markdown_table_hook_priority in self.active_headers:
                if self.first_table_row_column_count(split) > 0:
                    self.clear_table_header()
                    self.header_ended_this_unit = True
                else:
                    self.clear_table_header()

        # 1. Check for header-end markers among currently active headers
        for hook in self.hooks:
            if hook.priority in self.active_headers:
                if hook.end_pattern.search(split):
                    self.ended_headers[hook.priority] = True
                    if hook.priority in self.active_headers:
                        del self.active_headers[hook.priority]
                    if hook.priority in self.pending_extend:
                        del self.pending_extend[hook.priority]


        # 1b. Paragraph splits consume the blank line between tables. Mark a break
        # after "| last row |\n\n" and resolve on the next unit; also end when a new
        # table row has a different column count than the active header.
        if markdown_table_hook_priority in self.active_headers:
            if markdown_table_hook_priority not in self.pending_extend:
                if self.split_ends_with_paragraph_break(split):
                    self.pending_table_break = True
                else:
                    self.end_table_header_on_column_mismatch(split)

        # 2. If a header has an empty column-name row (e.g. "||"), replace it with
        #    a proper Markdown table header using the first data row as column names.
        #
        #    Before: "||"           + "| --- | --- |\n"
        #    After:  "| col1 | col2 |\n" + "| --- | --- |\n"
        pending_extend_keys = list(self.pending_extend.keys())
        for p in pending_extend_keys:
            if p in self.active_headers and table_row_pattern.search(split):
                sep = self.extract_separator_line(self.active_headers[p])
                self.active_headers[p] = split + sep
            if p in self.pending_extend:
                del self.pending_extend[p]

        # 3. Check for new header-start markers (only for hooks that are neither active nor ended)
        for hook in self.hooks:
            if hook.priority in self.active_headers:
                continue
            if hook.priority in self.ended_headers:
                continue
            match = hook.start_pattern.search(split)
            if match:
                self.active_headers[hook.priority] = match.group(0)
                if self.is_empty_table_header_row(self.active_headers[hook.priority]):
                    self.pending_extend[hook.priority] = True

        # 4. If all headers ended, clear the ended set so future tables can be tracked
        if not self.active_headers:
            self.ended_headers.clear()

    def get_headers(self) -> str:
        """Return all active headers concatenated, sorted by priority descending."""
        if not self.active_headers:
            return ""

        # Sort entries by priority descending
        sorted_entries = sorted(self.active_headers.items(), key=lambda x: x[0], reverse=True)

        # Join the header texts with newlines
        return "\n".join(header_text for _, header_text in sorted_entries)

    def is_empty_table_header_row(self, header: str) -> bool:
        """Check if the header row (the line before the separator) contains only pipes and whitespace.

        This means the column names are empty. This is common with MarkItDown and similar
        converters that produce tables like:

        ||
        | --- | --- |
        | real column A | real column B |
        """
        # Find the first newline
        idx = header.find('\n')
        if idx < 0:
            return False

        # Get the first row (before the separator)
        row = header[:idx].strip()

        # Check if all characters are pipes, spaces, or tabs
        for char in row:
            if char not in '| \t':
                return False
        return True

    def extract_separator_line(self, header: str) -> str:
        """Return the separator line (e.g. "| --- | --- |\n") from a table header string.
        Looks for the line containing "---"."""
        for line in header.split('\n'):
            if '---' in line:
                return line + '\n'
        return ""

    def clear_table_header(self) -> None:
        """Clear the active table header state."""
        self.ended_headers[markdown_table_hook_priority] = True
        if markdown_table_hook_priority in self.active_headers:
            del self.active_headers[markdown_table_hook_priority]
        if markdown_table_hook_priority in self.pending_extend:
            del self.pending_extend[markdown_table_hook_priority]

    def end_table_header_on_column_mismatch(self, split: str) -> None:
        """End the table header if the next row has a different column count."""
        if markdown_table_hook_priority not in self.active_headers:
            return

        header = self.active_headers[markdown_table_hook_priority]
        row_cols = self.first_table_row_column_count(split)
        header_cols = self.header_table_column_count(header)

        if row_cols > 0 and header_cols > 0 and row_cols != header_cols:
            self.clear_table_header()
            self.header_ended_this_unit = True

    def split_ends_with_paragraph_break(self, split: str) -> bool:
        """Check if the split ends with a paragraph break (\n\n or \r\n\r\n)."""
        trimmed = split.rstrip(' \t\r')
        return trimmed.endswith('\n\n') or trimmed.endswith('\r\n\r\n')

    def table_row_column_count(self, line: str) -> int:
        """Count the number of columns in a Markdown table row."""
        line = line.strip()
        if not line.startswith('|'):
            return 0

        parts = line.split('|')
        # Remove empty first part if line starts with |
        if parts and not parts[0].strip():
            parts = parts[1:]
        # Remove empty last part if line ends with |
        if parts and not parts[-1].strip():
            parts = parts[:-1]

        return len(parts)

    def first_table_row_column_count(self, text: str) -> int:
        """Find the first table row in the text and return its column count."""
        for line in text.split('\n'):
            line = line.strip()
            if line and table_row_pattern.search(line):
                return self.table_row_column_count(line)
        return 0

    def header_table_column_count(self, header: str) -> int:
        """Extract the column count from a table header (excluding separator lines)."""
        for line in header.split('\n'):
            line = line.strip()
            if not line or '---' in line:
                continue
            n = self.table_row_column_count(line)
            if n > 0:
                return n
        return 0

def header_column_mismatch(headers: str, next_unit: str) -> bool:
    """Report whether the next split unit starts a new table whose width differs
    from the active markdown table header."""
    header_cols = HeaderTracker().header_table_column_count(headers)
    row_cols = HeaderTracker().first_table_row_column_count(next_unit)
    return header_cols > 0 and row_cols > 0 and header_cols != row_cols