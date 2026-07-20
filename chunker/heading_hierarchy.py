import re
from typing import Tuple

class HeadingHierarchy:
    """
    HeadingHierarchy maintains a stack of active Markdown headings indexed by
    level (1..6). Pushing a level-N heading pops every entry of level >= N
    because the previous siblings/descendants are no longer in scope.
    """

    # MarkdownHeadingPattern matches an ATX-style Markdown heading at line start.
    # Capture groups: (1) hashes, (2) heading text.
    MARKDOWN_HEADING_PATTERN = re.compile(r'(?m)^(#{1,6})\s+(.+?)\s*#*\s*$')

    def __init__(self):
        """Initialize an empty hierarchy."""
        # stack[i] holds the heading text for level i+1 (so stack[0] = H1)
        # Entries beyond the deepest active level are empty strings.
        self._stack = ["" for _ in range(6)]
        self._depth = 0  # current deepest active level (0 if no active heading)

    def observe(self, line: str) -> Tuple[int, str]:
        """
        Parse line and update the hierarchy if line is a Markdown heading.
        Returns the (level, heading_text) when a heading was recognized,
        or (0, "") otherwise. Lines that look like headings inside fenced code
        blocks are NOT detected here — callers must avoid feeding code-block
        content to observe (the heading splitter does so).
        """
        match = self.MARKDOWN_HEADING_PATTERN.search(line)
        if not match:
            return 0, ""

        level = len(match.group(1))
        if level < 1 or level > 6:
            return 0, ""

        heading = match.group(2).strip()
        # Replace this level and clear deeper ones — siblings/descendants of
        # the previous heading at this level are no longer in scope.
        self._stack[level-1] = heading
        for i in range(level, 6):
            self._stack[i] = ""

        if level > self._depth:
            self._depth = level
        else:
            # Recompute depth: it might shrink if we just pushed a shallower heading.
            self._depth = 0
            for i in range(6):
                if self._stack[i] != "":
                    self._depth = i + 1

        return level, heading

    def breadcrumb(self) -> str:
        """
        Returns the current heading path joined by " > ", e.g.
        "Chapter 1 > Section 2 > Subsection a". Returns "" when no headings
        are active.
        """
        if self._depth == 0:
            return ""

        parts = []
        for i in range(self._depth):
            if self._stack[i] != "":
                parts.append(self._stack[i])

        return " > ".join(parts)

    def breadcrumb_with_hashes(self) -> str:
        """
        Returns the path with the original `#` prefixes,
        suitable for embedding back into chunk content as a context header.
        Example: "# Chapter 1\n## Section 2\n### Subsection a"
        """
        if self._depth == 0:
            return ""

        lines = []
        for i in range(self._depth):
            if self._stack[i] == "":
                continue
            line = "#" * (i + 1) + " " + self._stack[i]
            lines.append(line)

        return "\n".join(lines)

    def depth(self) -> int:
        """Returns the current deepest active heading level."""
        return self._depth

    def reset(self) -> None:
        """Clears all state."""
        for i in range(len(self._stack)):
            self._stack[i] = ""
        self._depth = 0