import unittest
import pytest
from typing import List, Dict, Any
from dataclasses import dataclass
from splitter import (
    extract_image_refs, 
    build_units_with_protection, 
    split_by_separators,
    split_text,
    split_text_parent_child,
    protected_spans,
    SplitterConfig,
    Chunk
)

from header_tracker import HeaderTracker


def restore_text_from_chunks(chunks: List[Chunk]) -> str:
    """Reconstructs the original text using only chunk Start/End positions.

    For chunks with prepended headers, the header is a "virtual" prefix whose length
    is len(Content) - (End - Start). The original text portion is the last (End-Start)
    runes of Content.
    """
    if not chunks:
        return ""

    # Sort by End (ascending), then Start (ascending)
    sorted_chunks = sorted(chunks, key=lambda c: (c.end, c.start))

    result = []
    last_end = 0

    for chunk in sorted_chunks:
        if chunk.end <= last_end:
            continue  # fully contained in a previously processed chunk

        content = chunk.content
        content_chars = list(content)
        span_len = chunk.end - chunk.start
        header_len = len(content_chars) - span_len
        if header_len < 0:
            header_len = 0

        # original_portion is the text[Start:End] part, excluding any prepended header
        original_portion = content_chars[header_len:]

        # Only take the portion after last_end (skip overlap)
        new_start = 0
        if last_end > chunk.start:
            new_start = last_end - chunk.start

        if new_start < len(original_portion):
            result.extend(original_portion[new_start:])

        last_end = chunk.end

    return ''.join(result)


class TestSplit(unittest.TestCase):
    def test_split_text_basic_ascii(self):
        text = "Hello world. This is a test."
        cfg = SplitterConfig(
            chunk_size = 100,
            chunk_overlap = 0,
            separators = ['. ']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) > 0, "expected at least one chunk"

        combined = "".join(chunk.content for chunk in chunks)
        assert combined == text, f"combined content mismatch:\n  got:  {repr(combined)}\n  want: {repr(text)}"


    def test_split_text_chinese_text_start_end_are_rune_offsets(self):
        # Each Chinese character is 1 character in Python
        text = "你好世界这是一个测试文本用于检验分割位置"
        char_count = len(text)
        byte_count = len(text.encode('utf-8'))

        assert char_count != byte_count, "test requires multi-byte characters"

        cfg = SplitterConfig(
            chunk_size = 100,
            chunk_overlap = 0,
            separators = ['\n']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) == 1, f"expected 1 chunk, got {len(chunks)}"

        c = chunks[0]
        assert c.start == 0, f"Start: got {c.start}, want 0"
        assert c.end == char_count, f"End: got {c.end}, want {char_count} (charCount); byteCount would be {byte_count}"


    def test_split_text_chinese_multi_chunk_start_end_consistency(self):
        # Build a long Chinese text that will be split into multiple chunks.
        line = "这是一段中文内容用于测试分割功能是否正确。"
        text = (line + "\n\n") * 20
        text = text.rstrip("\n")

        cfg = SplitterConfig(
            chunk_size = 30,
            chunk_overlap = 5,
            separators = ['\n\n', '\n', '。']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) >= 2, f"expected multiple chunks, got {len(chunks)}"

        text_chars = list(text)
        for i, c in enumerate(chunks):
            content_chars = list(c.content)
            content_char_len = len(content_chars)

            # End - Start must equal the char length of the content
            span_len = c.end - c.start
            assert span_len == content_char_len, f"chunk[{i}]: End({c.end}) - Start({c.start}) = {span_len}, but char len of content = {content_char_len}"

            # Start must be non-negative and End must not exceed total char count
            assert c.start >= 0, f"chunk[{i}]: Start is negative: {c.start}"
            assert c.end <= len(text_chars), f"chunk[{i}]: End {c.end} exceeds total char count {len(text_chars)}"

            # Content from char slice must match the chunk content
            if 0 <= c.start <= c.end <= len(text_chars):
                sliced = ''.join(text_chars[c.start:c.end])
                assert sliced == c.content, f"chunk[{i}]: content mismatch via char slice:\n  got:  {repr(sliced)}\n  want: {repr(c.content)}"


    def test_split_text_mixed_chinese_and_ascii(self):
        text = "Hello你好World世界Test测试"
        cfg = SplitterConfig(
            chunk_size = 100,
            chunk_overlap = 0,
            separators = ['\n']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) == 1, f"expected 1 chunk, got {len(chunks)}"

        c = chunks[0]
        expected_char_len = len(text)
        assert c.end - c.start == expected_char_len, f"End({c.end}) - Start({c.start}) = {c.end-c.start}, want char len {expected_char_len} (byte len would be {len(text.encode('utf-8'))})"


    def test_split_text_protected_pattern_chinese_context(self):
        # Test protected markdown images in Chinese context.
        text = "这是前面的中文内容。![图片描述](http://example.com/img.png)这是后面的中文内容。"
        cfg = SplitterConfig(
            chunk_size = 200,
            chunk_overlap = 0,
            separators = ['。']
        )
        chunks = split_text(text, cfg)

        text_chars = list(text)
        for i, c in enumerate(chunks):
            assert 0 <= c.start <= c.end <= len(text_chars), f"chunk[{i}]: out of char range [{c.start}, {c.end}), total chars {len(text_chars)}"

            sliced = ''.join(text_chars[c.start:c.end])
            assert sliced == c.content, f"chunk[{i}]: char-slice mismatch:\n  sliced: {repr(sliced)}\n  content: {repr(c.content)}"


    def test_split_text_simulate_merge_slicing(self):
        # Simulate what merge.go:104-106 does to ensure it won't panic.
        # This is the exact pattern that caused the production crash.
        line = "这是第一段内容用于模拟知识库问答的文本"
        text = line + "\n\n" + line + "\n\n" + line

        cfg = SplitterConfig(
            chunk_size = 25,
            chunk_overlap = 5,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) >= 2, f"need at least 2 chunks for overlap test, got {len(chunks)}"

        for i in range(1, len(chunks)):
            prev = chunks[i-1]
            curr = chunks[i]

            if curr.start > prev.end:
                continue  # non-overlapping, no merge needed

            # This is the exact merge.go logic:
            content_chars = list(curr.content)
            offset = len(content_chars) - (curr.end - prev.end)

            assert offset >= 0, f"chunk[{i}] merge panic: offset={offset} < 0 (contentChars={len(content_chars)}, curr.end={curr.end}, prev.end={prev.end})"
            assert offset <= len(content_chars), f"chunk[{i}] merge panic: offset={offset} > len(contentChars)={len(content_chars)}"

            _ = ''.join(content_chars[offset:])


    def test_split_text_recursive_separators_no_oversize_chunks(self):
        # One paragraph break, then 50 short newline-separated lines forming
        # ~1500 chars in the second paragraph.
        body = "This is one fairly short line of text.\n" * 50
        text = "lead paragraph that is short.\n\n" + body
        cfg = SplitterConfig(
            chunk_size = 300,
            chunk_overlap = 30,
            separators = ['\n\n', '\n', '. ']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) >= 2, f"expected multiple chunks, got {len(chunks)}"

        # No chunk should exceed roughly 1.5x ChunkSize — recursive splitting
        # at the next-priority separator should keep this bounded.
        max_allowed = cfg.chunk_size * 3 // 2
        for i, c in enumerate(chunks):
            l = len(c.content)
            assert l <= max_allowed, f"chunk {i} is {l} chars, > 1.5x ChunkSize ({max_allowed}) — recursive split missing"


    def test_split_text_empty(self):
        cfg = SplitterConfig()
        chunks = split_text("", cfg)
        assert len(chunks) == 0, f"expected 0 chunks for empty text, got {len(chunks)}"


    def test_split_text_single_char_chinese(self):
        text = "你"
        cfg = SplitterConfig(
            chunk_size = 10,
            chunk_overlap = 0,
            separators = ['\n']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) == 1, f"expected 1 chunk, got {len(chunks)}"
        assert chunks[0].start == 0 and chunks[0].end == 1, f"expected [0,1), got [{chunks[0].start},{chunks[0].end})"


    def test_split_text_latex_block_in_chinese(self):
        text = "前面的文字$$E=mc^2$$后面的文字"
        cfg = SplitterConfig(
            chunk_size = 200,
            chunk_overlap = 0,
            separators = ['\n']
        )
        chunks = split_text(text, cfg)

        text_chars = list(text)
        for i, c in enumerate(chunks):
            span_len = c.end - c.start
            content_char_len = len(c.content)
            assert span_len == content_char_len, f"chunk[{i}]: span {span_len} != char len {content_char_len}"
            assert c.end <= len(text_chars), f"chunk[{i}]: End {c.end} > total chars {len(text_chars)}"


    def test_split_text_code_block_in_chinese(self):
        text = "中文描述\n```python\nprint('hello')\n```\n继续中文"
        cfg = SplitterConfig(
            chunk_size = 200,
            chunk_overlap = 0,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        text_chars = list(text)
        for i, c in enumerate(chunks):
            assert 0 <= c.start <= c.end <= len(text_chars), f"chunk[{i}]: out of range [{c.start},{c.end}), total {len(text_chars)}"

            sliced = ''.join(text_chars[c.start:c.end])
            assert sliced == c.content, f"chunk[{i}]: char-slice mismatch:\n  sliced: {repr(sliced)}\n  content: {repr(c.content)}"


    def test_split_text_overlap_chunks_non_negative_start(self):
        # When overlap is used, start of the next chunk could go before 0 if broken.
        text = ("中文测试内容，") * 50
        cfg = SplitterConfig(
            chunk_size = 20,
            chunk_overlap = 5,
            separators = ['，']
        )
        chunks = split_text(text, cfg)

        for i, c in enumerate(chunks):
            assert c.start >= 0, f"chunk[{i}]: negative Start {c.start}"
            assert c.end >= c.start, f"chunk[{i}]: End {c.end} < Start {c.start}"


    def test_build_units_with_protection_rune_offsets(self):
        text = "你好世界"
        units = build_units_with_protection(text, None, ['\n'], 0)

        assert len(units) == 1, f"expected 1 unit, got {len(units)}"

        u = units[0]
        expected_char_len = 4  # 4 Chinese characters
        byte_len = len(text.encode('utf-8'))  # 12 bytes

        assert u.start == 0, f"start: got {u.start}, want 0"
        assert u.end == expected_char_len, f"end: got {u.end}, want {expected_char_len} (char len); byte len is {byte_len}"


    def test_build_units_with_protection_with_protected_span(self):
        text = "前面![alt](url)后面"
        protected = protected_spans(text)
        units = build_units_with_protection(text, protected, ['\n'], 0)

        text_chars = list(text)
        for i, u in enumerate(units):
            content_char_len = len(u.text)
            span_len = u.end - u.start
            assert span_len == content_char_len, f"unit[{i}] {repr(u.text)}: span {span_len} != char len {content_char_len} (byte len {len(u.text.encode('utf-8'))})"
            assert 0 <= u.start <= u.end <= len(text_chars), f"unit[{i}]: out of range [{u.start},{u.end}), total chars {len(text_chars)}"


    def test_split_by_separators(self):
        tests = [
            {"text": "a\n\nb\n\nc", "separators": ["\n\n"], "want_parts": 5},
            {"text": "abc", "separators": ["\n"], "want_parts": 1},
            {"text": "a\nb\nc", "separators": ["\n"], "want_parts": 5},
            {"text": "", "separators": ["\n"], "want_parts": 1},
        ]

        for tt in tests:
            parts = split_by_separators(tt['text'], tt['separators'], 0)
            assert len(parts) == tt['want_parts'], f"split_by_separators({repr(tt['text'])}, {tt['separators']}): got {len(parts)} parts {parts}, want {tt['want_parts']}"


    def test_extract_image_refs(self):
        text = "hello ![alt1](url1) world ![alt2](url2) end"
        refs = extract_image_refs(text)

        assert len(refs) == 2, f"expected 2 refs, got {len(refs)}"

        assert refs[0].original_ref == "url1" and refs[0].alt_text == "alt1", f"ref[0] mismatch: {refs[0]}"
        assert refs[1].original_ref == "url2" and refs[1].alt_text == "alt2", f"ref[1] mismatch: {refs[1]}"


    def test_split_text_large_chinese_document(self):
        # Simulate a real document with paragraphs of Chinese text.
        sb = []
        for i in range(100):
            sb.append(f"第{i}段：这是一段用于测试的中文内容，包含各种常见的汉字和标点符号。")
            sb.append("\n\n")
        text = ''.join(sb)

        cfg = SplitterConfig(
            chunk_size = 50,
            chunk_overlap = 10,
            separators = ['\n\n', '\n', '。']
        )
        chunks = split_text(text, cfg)

        text_chars = list(text)
        for i, c in enumerate(chunks):
            content_char_len = len(c.content)
            span_len = c.end - c.start
            assert span_len == content_char_len, f"chunk[{i}]: End({c.end})-Start({c.start})={span_len} != charLen({content_char_len})"
            assert c.start >= 0, f"chunk[{i}]: negative Start {c.start}"
            assert c.end <= len(text_chars), f"chunk[{i}]: End {c.end} > total chars {len(text_chars)}"

            if 0 <= c.start <= c.end <= len(text_chars):
                sliced = ''.join(text_chars[c.start:c.end])
                assert sliced == c.content, f"chunk[{i}]: content mismatch via char-slice"

        # Simulate merge.go logic on all overlapping chunk pairs
        for i in range(1, len(chunks)):
            prev = chunks[i-1]
            curr = chunks[i]
            if curr.start > prev.end:
                continue

            content_chars = list(curr.content)
            offset = len(content_chars) - (curr.end - prev.end)

            assert offset >= 0 and offset <= len(content_chars), f"chunk[{i}] merge would panic: offset={offset}, contentChars={len(content_chars)}, curr.end={curr.end}, prev.end={prev.end}"


    def test_split_text_table_header_prepended_to_chunks(self):
        # A markdown table large enough to span multiple chunks.
        # Each chunk after the first should have the header row + separator prepended.
        text = (
            "前面的文字\n\n" +
            "| 姓名 | 年龄 | 城市 |\n" +
            "| --- | --- | --- |\n" +
            "| 张三 | 25 | 北京 |\n" +
            "| 李四 | 30 | 上海 |\n" +
            "| 王五 | 28 | 广州 |\n" +
            "| 赵六 | 35 | 深圳 |\n" +
            "| 孙七 | 22 | 杭州 |\n" +
            "| 周八 | 40 | 成都 |\n" +
            "\n后面的文字"
        )

        table_header = "| 姓名 | 年龄 | 城市 |\n| --- | --- | --- |\n"

        cfg = {
            'chunk_size': 60,
            'chunk_overlap': 5,
            'separators': ['\n\n', '\n']
        }
        cfg = SplitterConfig(
            chunk_size = 60,
            chunk_overlap = 5,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) >= 3, f"expected at least 3 chunks, got {len(chunks)}"

        # Find chunks that contain table row data but not the original header position.
        # These should have the header prepended.
        header_prepend_count = 0
        for c in chunks:
            if ("| 张三" in c.content or "| 李四" in c.content or "| 王五" in c.content or
                "| 赵六" in c.content or "| 孙七" in c.content or
                "| 周八" in c.content):
                if "| 张三" not in c.content:
                    # This is a chunk with table rows but not the first row;
                    # it should have the header prepended.
                    if not c.content.startswith(table_header):
                        pytest.fail(f"chunk (seq={c.seq}) has table rows but is missing prepended header:\n{c.content}")
                    else:
                        header_prepend_count += 1

        assert header_prepend_count > 0, "expected at least one chunk to have prepended table header, found none"


    def test_split_text_no_header_for_non_table_content(self):
        # Ensure header prepending doesn't affect non-table content.
        text = ("这是一段普通的中文文本，不包含任何表格。\n\n") * 10

        cfg = SplitterConfig(
            chunk_size = 30,
            chunk_overlap = 5,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        text_chars = list(text)
        for i, c in enumerate(chunks):
            content_char_len = len(c.content)
            span_len = c.end - c.start
            assert span_len == content_char_len, f"chunk[{i}]: span {span_len} != char len {content_char_len} (no table, should be exact)"
            assert c.end <= len(text_chars), f"chunk[{i}]: End {c.end} exceeds total chars {len(text_chars)}"


    def test_split_text_table_header_ended_by_empty_line(self):
        # After the table ends (empty line), subsequent chunks should NOT have the header.
        text = (
            "| A | B |\n" +
            "| --- | --- |\n" +
            "| 1 | 2 |\n" +
            "| 3 | 4 |\n" +
            "\n" +
            "这是表格之后的普通文本内容，不应该包含表头。\n" +
            "更多的普通文本内容用于填充。"
        )

        cfg = SplitterConfig(
            chunk_size = 40,
            chunk_overlap = 5,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        for c in chunks:
            has_table_row = "| A | B |" in c.content or "|" in c.content
            has_plain_text = "这是表格之后" in c.content or "更多的普通" in c.content

            if has_plain_text and not has_table_row:
                # This chunk is purely post-table text; should NOT have table header
                if "| --- |" in c.content:
                    pytest.fail(f"post-table chunk should not contain table header:\n{c.content}")


    def test_header_tracker_basic_lifecycle(self):
        ht = HeaderTracker()

        # Before table: no headers
        ht.update("Some regular text")
        assert ht.get_headers() == "", f"expected no headers before table, got {repr(ht.get_headers())}"

        # Table header unit
        ht.update("| A | B |\n| --- | --- |\n")
        assert ht.get_headers() != "", "expected active header after table header unit"

        # Table row: header should stay active
        ht.update("| 1 | 2 |\n")
        assert ht.get_headers() != "", "header should remain active during table rows"

        # Empty line: header should end
        ht.update("\n")
        assert ht.get_headers() == "", f"header should be cleared after empty line, got {repr(ht.get_headers())}"

        # New table can be tracked after the old one ended
        ht.update("| X | Y |\n| --- | --- |\n")
        assert ht.get_headers() != "", "expected new header to be tracked after previous table ended"


    def test_header_tracker_empty_header_row_rewrite(self):
        # Some converters (e.g., MarkItDown) produce tables with empty header rows:
        #   ||
        #   | --- | --- |
        #   | real col A | real col B |
        # The tracker should rewrite the header to be a proper Markdown table header:
        #   | real col A | real col B |
        #   | --- | --- |
        ht = HeaderTracker()

        # Empty header row + separator
        ht.update("||\n| --- | --- | --- |\n")
        h = ht.get_headers()
        assert h != "", "expected active header after empty header unit"

        # First data row → becomes the real column names
        ht.update("| 测试用例 ID | 测试模块 | 备注 |\n")
        h = ht.get_headers()

        assert "测试用例 ID" in h, f"rewritten header should contain column names, got:\n{h}"
        assert "||" not in h, f"rewritten header should NOT contain empty '||' row, got:\n{h}"
        assert "---" in h, f"rewritten header should contain separator, got:\n{h}"

        # Column names should come BEFORE the separator
        col_idx = h.find("测试用例 ID")
        sep_idx = h.find("---")
        assert col_idx < sep_idx, f"column names should appear before separator in rewritten header:\n{h}"

        # Subsequent data rows should NOT be absorbed
        ht.update("| TC-001 | 模块A | 备注1 |\n")
        h2 = ht.get_headers()
        assert "TC-001" not in h2, f"header should NOT include subsequent data rows, got:\n{h2}"

        # Table end
        ht.update("\n")
        assert ht.get_headers() == "", "header should be cleared after empty line"


    def test_header_tracker_normal_header_no_extension(self):
        # A table with proper column names in the header row should NOT be extended.
        ht = HeaderTracker()

        ht.update("| 姓名 | 年龄 |\n| --- | --- |\n")
        h = ht.get_headers()
        assert h != "", "expected active header"

        # First data row should NOT be absorbed into the header
        ht.update("| 张三 | 25 |\n")
        h2 = ht.get_headers()
        assert "张三" not in h2, f"normal header should not absorb data rows, got:\n{h2}"


    def test_split_text_empty_header_row_prepend(self):
        # Simulate MarkItDown output: empty header row, real column names in first data row.
        text = (
            "前言\n\n" +
            "||\n" +
            "| --- | --- | --- |\n" +
            "| 用例ID | 模块 | 步骤 |\n" +
            "| TC-001 | A | 步骤1 |\n" +
            "| TC-002 | B | 步骤2 |\n" +
            "| TC-003 | C | 步骤3 |\n" +
            "| TC-004 | D | 步骤4 |\n" +
            "\n" +
            "结尾"
        )

        cfg = SplitterConfig(
            chunk_size = 80,
            chunk_overlap = 5,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        for c in chunks:
            has_later_row = ("TC-002" in c.content or
                            "TC-003" in c.content or
                            "TC-004" in c.content)

            if has_later_row and "TC-001" not in c.content:
                # Should have column names prepended
                assert "用例ID" in c.content, f"chunk with data rows should have real column names prepended:\n{c.content}"

                # Should NOT have the empty || row
                lines = c.content.split('\n')
                for line in lines:
                    trimmed = line.strip()
                    if trimmed and all(c in ' |' for c in trimmed):
                        pytest.fail(f"chunk should NOT contain empty pipe row {repr(trimmed)}:\n{c.content}")

            # No line should appear as a duplicate in any chunk
            lines = c.content.rstrip('\n').split('\n')
            seen = {}
            for line in lines:
                trimmed = line.strip()
                if not trimmed or "---" in trimmed:
                    continue
                if trimmed in seen:
                    seen[trimmed] += 1
                    assert seen[trimmed] <= 1, f"line appears {seen[trimmed]} times in chunk (seq={c['seq']}): {repr(trimmed)}"
                else:
                    seen[trimmed] = 1

        # Verify restoration still works
        restored = restore_text_from_chunks(chunks)
        assert restored == text, f"restoration failed for empty-header table\n  original: {repr(text)}\n  restored: {repr(restored)}"


    def test_header_tracker_column_mismatch_ends_table(self):
        ht = HeaderTracker()
        ht.update("| Name | Game | Fame | Blame |\n| --- | --- | --- | --- |\n")
        assert ht.get_headers() != "", "expected active table header"

        ht.update("| Sinple | Table |\n")
        assert ht.get_headers() == "", f"2-col row should end 4-col table header, still active:\n{ht.get_headers()}"


    def test_header_tracker_paragraph_break_ends_on_next_unit(self):
        ht = HeaderTracker()
        ht.update("| Name | Game | Fame | Blame |\n| --- | --- | --- | --- |\n")
        ht.update("| Russell Wilson | Football | High | Tacky uniform |\n\n")
        assert ht.get_headers() != "", "paragraph break alone should not clear header yet"
        assert ht.pending_table_break, "expected pendingTableBreak after row ending with \n\n"

        ht.update("| Sinple | Table |\n")
        assert ht.get_headers() == "", f"next table row should clear previous header, got {repr(ht.get_headers())}"
        assert ht.header_ended_this_unit, "expected flush signal when new table starts after paragraph break"


    def test_split_text_en_tables_no_cross_table_header(self):
        text = (
            "## A table, with and without a header row\n\n" +
            "| Name | Game | Fame | Blame |\n" +
            "| --- | --- | --- | --- |\n" +
            "| Lebron James | Basketball | Very High | Leaving Cleveland |\n" +
            "| Ryan Braun | Baseball | Moderate | Steroids |\n" +
            "| Russell Wilson | Football | High | Tacky uniform |\n\n" +
            "| Sinple | Table |\n" +
            "| Without | Header |\n\n" +
            "| Simple  Multiparagraph | Table  Full |\n" +
            "| Of  Paragraphs | In each  Cell. |\n"
        )

        cfg = SplitterConfig(
            chunk_size = 200,
            chunk_overlap = 20,
            separators = ['\n\n', '\n', '。']
        )
        chunks = split_text(text, cfg)

        assert len(chunks) >= 2, f"expected multiple chunks, got {len(chunks)}"

        for i, c in enumerate(chunks):
            has_sinple = "| Sinple | Table |" in c.content
            has_simple = "| Simple  Multiparagraph |" in c.content

            if has_sinple or has_simple:
                if "| Name | Game | Fame | Blame |" in c.content:
                    pytest.fail(f"chunk[{i}] must not carry table-1 header into later tables:\n{c.content}")


    def test_split_text_multiple_tables_in_document(self):
        text = (
            "第一个表格：\n\n" +
            "| 名称 | 值 |\n" +
            "| --- | --- |\n" +
            "| A | 1 |\n" +
            "| B | 2 |\n" +
            "| C | 3 |\n" +
            "\n" +
            "中间的文字\n\n" +
            "| 项目 | 状态 |\n" +
            "| --- | --- |\n" +
            "| X | 完成 |\n" +
            "| Y | 进行中 |\n" +
            "| Z | 未开始 |\n" +
            "\n" +
            "结尾文字"
        )

        cfg = SplitterConfig(
            chunk_size = 50,
            chunk_overlap = 5,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        # Verify that if a chunk has rows from table 2, it has table 2's header, not table 1's.
        for c in chunks:
            if "| Y |" in c.content and "| X |" not in c.content:
                assert "| 项目 | 状态 |" in c.content, f"chunk with table-2 rows should have table-2 header:\n{c.content}"
                assert "| 名称 | 值 |" not in c.content, f"chunk with table-2 rows should NOT have table-1 header:\n{c.content}"


    def test_split_text_restore_text_no_table(self):
        # Plain Chinese text without any tables — baseline restoration check.
        sb = []
        for i in range(30):
            sb.append(f"第{i}段：这是一段用于测试的中文内容，包含各种标点符号。")
            sb.append("\n\n")
        text = ''.join(sb)

        cfg = SplitterConfig(
            chunk_size = 50,
            chunk_overlap = 10,
            separators = ['\n\n', '\n', '。']
        )
        chunks = split_text(text, cfg)

        restored = restore_text_from_chunks(chunks)
        assert restored == text, f"restoration failed for plain text\n  original len: {len(text)}\n  restored len: {len(restored)}"


    def test_split_text_restore_text_with_table(self):
        # Document with a table large enough to span multiple chunks.
        # Use a 2-column table (shorter header ~24 chars) + ChunkSize 80 so the
        # header can be prepended (header 24 + row ~16 = 40 < 80).
        text = (
            "这是文档前言部分的内容。\n\n" +
            "| 姓名 | 城市 |\n" +
            "| --- | --- |\n" +
            "| 张三 | 北京 |\n" +
            "| 李四 | 上海 |\n" +
            "| 王五 | 广州 |\n" +
            "| 赵六 | 深圳 |\n" +
            "| 孙七 | 杭州 |\n" +
            "| 周八 | 成都 |\n" +
            "| 吴九 | 武汉 |\n" +
            "| 郑十 | 南京 |\n" +
            "\n" +
            "这是表格之后的文字内容。\n" +
            "这里还有更多的普通段落。"
        )

        cfg = SplitterConfig(
            chunk_size = 80,
            chunk_overlap = 5,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        # 1. Verify basic position invariants
        text_chars = list(text)
        for i, c in enumerate(chunks):
            assert c.start >= 0, f"chunk[{i}]: Start {c.start} < 0"
            assert c.end <= len(text_chars), f"chunk[{i}]: End {c.end} > total chars {len(text_chars)}"
            assert c.end >= c.start, f"chunk[{i}]: End {c.end} < Start {c.start}"

        # 2. Verify text[Start:End] matches the non-header portion of Content
        for i, c in enumerate(chunks):
            content_chars = list(c.content)
            span_len = c.end - c.start
            header_len = len(content_chars) - span_len
            if header_len < 0:
                header_len = 0

            if 0 <= c.start <= c.end <= len(text_chars):
                original_slice = ''.join(text_chars[c.start:c.end])
                content_suffix = ''.join(content_chars[header_len:])
                assert original_slice == content_suffix, f"chunk[{i}]: text[{c.start}:{c.end}] != Content[{header_len}:]" + \
                    f"\n  text slice:     {repr(original_slice)}" + \
                    f"\n  content suffix: {repr(content_suffix)}"

        # 3. Restore original text and compare
        restored = restore_text_from_chunks(chunks)
        assert restored == text, f"restoration FAILED" + \
            f"\n  original char len: {len(text_chars)}" + \
            f"\n  restored char len: {len(restored)}"

        # 4. Verify full coverage — Start/End spans must cover [0, len(text_chars))
        covered = [False] * len(text_chars)
        for c in chunks:
            for p in range(c.start, min(c.end, len(text_chars))):
                covered[p] = True

        for i, v in enumerate(covered):
            if not v:
                pytest.fail(f"char position {i} is not covered by any chunk")


    def test_split_text_restore_text_with_multiple_tables(self):
        text = (
            "前言\n\n" +
            "| A | B |\n| --- | --- |\n" +
            "| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n| 7 | 8 |\n" +
            "\n中间文字\n\n" +
            "| X | Y |\n| --- | --- |\n" +
            "| a | b |\n| c | d |\n| e | f |\n" +
            "\n结尾"
        )

        cfg = SplitterConfig(
            chunk_size = 50,
            chunk_overlap = 10,
            separators = ['\n\n', '\n']
        )
        chunks = split_text(text, cfg)

        # Restore and compare
        restored = restore_text_from_chunks(chunks)
        assert restored == text, f"multi-table restoration failed\n  original: {repr(text)}\n  restored: {repr(restored)}"

        # Verify text[Start:End] matches content suffix for every chunk
        text_chars = list(text)
        for i, c in enumerate(chunks):
            content_chars = list(c.content)
            span_len = c.end - c.start
            header_len = len(content_chars) - span_len
            if header_len < 0:
                header_len = 0

            if c.end <= len(text_chars):
                original_slice = ''.join(text_chars[c.start:c.end])
                content_suffix = ''.join(content_chars[header_len:])
                assert original_slice == content_suffix, f"chunk[{i}]: text[Start:End] mismatch with content suffix"


    def test_split_text_restore_text_with_overlap(self):
        # Larger overlap to stress the overlap+header interaction.
        text = (
            "| 列1 | 列2 | 列3 |\n" +
            "| --- | --- | --- |\n" +
            "| 数据A1 | 数据A2 | 数据A3 |\n" +
            "| 数据B1 | 数据B2 | 数据B3 |\n" +
            "| 数据C1 | 数据C2 | 数据C3 |\n" +
            "| 数据D1 | 数据D2 | 数据D3 |\n" +
            "| 数据E1 | 数据E2 | 数据E3 |\n" +
            "| 数据F1 | 数据F2 | 数据F3 |\n" +
            "\n" +
            "表后文本。"
        )

        for overlap in [0, 3, 10, 20]:
            cfg = SplitterConfig(
                chunk_size = 60,
                chunk_overlap = overlap,
                separators = ['\n\n', '\n']
            )
            chunks = split_text(text, cfg)

            restored = restore_text_from_chunks(chunks)
            assert restored == text, f"restoration failed with overlap={overlap}\n  orig len={len(text)}  rest len={len(restored)}"


    def test_split_text_parent_child_with_table_headers(self):
        text = (
            "前言\n\n" +
            "| 列A | 列B |\n" +
            "| --- | --- |\n" +
            "| 数据1 | 数据2 |\n" +
            "| 数据3 | 数据4 |\n" +
            "| 数据5 | 数据6 |\n" +
            "| 数据7 | 数据8 |\n" +
            "\n" +
            "结尾"
        )

        parent_cfg = SplitterConfig(
            chunk_size = 200,
            chunk_overlap = 0,
            separators = ['\n\n', '\n']
        )
        child_cfg = SplitterConfig(
            chunk_size = 40,
            chunk_overlap = 5,
            separators = ['\n\n', '\n']
        )
        result = split_text_parent_child(text, parent_cfg, child_cfg)

        assert len(result.children) > 0, "expected child chunks"

        # Verify child chunk positions don't exceed parent document
        text_chars = list(text)
        for i, child in enumerate(result.children):
            assert child.start >= 0, f"child[{i}]: negative Start {child.start}"
            assert child.end <= len(text_chars), f"child[{i}]: End {child.end} exceeds text char count {len(text_chars)}"


if __name__ == "__main__":
    unittest.main()