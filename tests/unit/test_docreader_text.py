from pathlib import Path

from app.rag.docreader import DocReaderTextSplitter, SplitterConfig, parse_novel_file, parse_novel_text
from app.rag.embeddings.structure_aware_chunker import build_novel_structure_documents


def test_parse_novel_text_extracts_chapters_and_context() -> None:
    parsed = parse_novel_text(
        "序言\n\n三体I\n\n上部\n\n第一章 科学边界\n\n叶文洁站在窗前。\n\nChapter 2\n\n杨冬说：你好。"
    )

    assert parsed.stats["chapter_count"] == 2
    assert [section.title for section in parsed.chapters] == ["第一章 科学边界", "Chapter 2"]
    assert parsed.chapters[0].chapter_number == 1
    assert parsed.chapters[0].header_path == ("三体I", "上部", "第一章 科学边界")
    assert parsed.stats["paragraph_count"] >= 2


def test_decode_text_file_supports_gb18030(tmp_path: Path) -> None:
    path = tmp_path / "novel.txt"
    path.write_bytes("第一章\n你好".encode("gb18030"))

    parsed = parse_novel_file(path)

    assert parsed.source.encoding == "gb18030"
    assert "你好" in parsed.source.text


def test_splitter_preserves_protected_span_and_overlap() -> None:
    splitter = DocReaderTextSplitter(SplitterConfig(chunk_size=12, chunk_overlap=4))
    chunks = splitter.split_text("第一段内容。第二段内容。```a\nb\nc```。第三段内容。")

    assert len(chunks) >= 2
    assert any("```a\nb\nc```" in chunk.text for chunk in chunks)
    assert any(set(chunks[i].text) & set(chunks[i - 1].text) for i in range(1, len(chunks)))


def test_storyrole_index_build_uses_docreader_metadata(tmp_path: Path) -> None:
    path = tmp_path / "novel.txt"
    path.write_text("第一章 开始\n\n叶文洁走进房间。\n\n第二章 继续\n\n汪淼看着她。", encoding="utf-8")

    documents = build_novel_structure_documents(path, chunk_size=80, chunk_overlap=8)

    assert documents
    metadata = documents[0].metadata
    assert metadata["parser_name"] == "novel-text-parser-v2"
    assert metadata["section_title"] == "第一章 开始"
    assert "chunk_start_offset" in metadata
    assert metadata["source_line_start"] >= 1
    assert "第一章 开始" in documents[0].page_content


def test_storyrole_chunk_ids_include_section_identity(tmp_path: Path) -> None:
    path = tmp_path / "novel.txt"
    path.write_text(
        "第一章 开始\n\n相同的章节开头。\n\n第二章 继续\n\n相同的章节开头。",
        encoding="utf-8",
    )

    documents = build_novel_structure_documents(path, chunk_size=80, chunk_overlap=8)

    chunk_ids = [str(document.metadata["chunk_id"]) for document in documents]
    assert len(chunk_ids) == len(set(chunk_ids))



