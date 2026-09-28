from pathlib import Path

from langchain_core.documents import Document

from app.rag.indexing.bm25_index import PersistentBM25Index
from app.rag.indexing.novel_indexer import NovelIndexer


def test_persistent_bm25_has_independent_keyword_recall(tmp_path: Path) -> None:
    index = PersistentBM25Index(tmp_path / "bm25.json")
    index.upsert_documents(
        [
            Document(page_content="量子通讯的关键实验", metadata={"chunk_id": "c1", "source_id": "novel"}),
            Document(page_content="普通叙事片段", metadata={"chunk_id": "c2", "source_id": "novel"}),
        ]
    )

    results = index.search("量子通讯", k=1)

    assert results[0][0].metadata["chunk_id"] == "c1"
    assert (tmp_path / "bm25.json").exists()


def test_novel_indexer_flattens_nested_chroma_metadata() -> None:
    metadata = NovelIndexer._chroma_metadata({
        "chunk_id": "c1",
        "parser_stats": {"chapter_count": 3},
        "keywords": ["三体", "罗辑"],
        "quality": 0.9,
    })

    assert metadata["parser_stats"] == '{"chapter_count":3}'
    assert metadata["keywords"] == '["三体","罗辑"]'
    assert metadata["quality"] == 0.9
