"""Unit tests for retrieval module"""

import pytest
from unittest.mock import Mock, patch

from src.retrieval import (
    get_llm,
    create_chain,
    query,
    format_docs,
    DEFAULT_MODEL,
    DEFAULT_K,
    PROMPT_TEMPLATE,
)


class TestGetLLM:
    """Tests for get_llm function"""

    @patch("src.retrieval.ChatOpenAI")
    def test_initializes_default_llm(self, mock_chat_openai):
        """Should initialize ChatOpenAI with default settings"""
        get_llm()

        mock_chat_openai.assert_called_once_with(
            model=DEFAULT_MODEL,
            temperature=0
        )

    @patch("src.retrieval.ChatOpenAI")
    def test_initializes_custom_llm(self, mock_chat_openai):
        """Should initialize ChatOpenAI with custom settings"""
        custom_model = "gpt-4"
        custom_temp = 0.7

        get_llm(model=custom_model, temperature=custom_temp)

        mock_chat_openai.assert_called_once_with(
            model=custom_model,
            temperature=custom_temp
        )


class TestFormatDocs:
    """Tests for format_docs function"""

    def test_formats_multiple_docs(self):
        """Should join document contents with double newlines"""
        doc1 = Mock()
        doc1.page_content = "First document"
        doc2 = Mock()
        doc2.page_content = "Second document"

        result = format_docs([doc1, doc2])

        assert result == "First document\n\nSecond document"

    def test_formats_empty_list(self):
        """Should return empty string for empty list"""
        result = format_docs([])

        assert result == ""


class TestCreateChain:
    """Tests for create_chain function"""

    @patch("src.retrieval.RunnablePassthrough")
    @patch("src.retrieval.StrOutputParser")
    @patch("src.retrieval.ChatPromptTemplate")
    @patch("src.retrieval.get_llm")
    def test_creates_chain_with_retriever(self, mock_get_llm, mock_prompt_cls, mock_parser, mock_passthrough):
        """Should create LCEL chain with retriever attached"""
        mock_index = Mock()
        mock_retriever = Mock()
        # Make retriever support | operator
        mock_retriever.__or__ = Mock(return_value=Mock())
        mock_index.as_retriever.return_value = mock_retriever

        chain = create_chain(mock_index)

        mock_index.as_retriever.assert_called_once_with(search_kwargs={"k": DEFAULT_K})
        mock_get_llm.assert_called_once_with(DEFAULT_MODEL)
        mock_prompt_cls.from_template.assert_called_once_with(PROMPT_TEMPLATE)
        assert chain.retriever == mock_retriever

    @patch("src.retrieval.RunnablePassthrough")
    @patch("src.retrieval.StrOutputParser")
    @patch("src.retrieval.ChatPromptTemplate")
    @patch("src.retrieval.get_llm")
    def test_passes_custom_params(self, mock_get_llm, mock_prompt_cls, mock_parser, mock_passthrough):
        """Should respect custom model and k parameters"""
        mock_index = Mock()
        mock_retriever = Mock()
        mock_retriever.__or__ = Mock(return_value=Mock())
        mock_index.as_retriever.return_value = mock_retriever
        custom_k = 10
        custom_model = "gpt-3.5-turbo"

        create_chain(mock_index, model=custom_model, k=custom_k)

        mock_get_llm.assert_called_once_with(custom_model)
        mock_index.as_retriever.assert_called_once_with(search_kwargs={"k": custom_k})


class TestQuery:
    """Tests for query function"""

    def test_executes_query_and_formats_response(self):
        """Should invoke chain and format output correctly"""
        mock_chain = Mock()
        mock_retriever = Mock()
        mock_chain.retriever = mock_retriever

        # Setup mock documents
        mock_doc1 = Mock()
        mock_doc1.metadata = {"source": "doc1.pdf", "page": 1}
        mock_doc2 = Mock()
        mock_doc2.metadata = {"source": "doc2.pdf", "page": 5}

        mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
        mock_chain.invoke.return_value = "This is the answer."

        question = "test question"
        result = query(mock_chain, question)

        mock_retriever.invoke.assert_called_once_with(question)
        mock_chain.invoke.assert_called_once_with(question)

        assert result == {
            "answer": "This is the answer.",
            "sources": [
                {"source": "doc1.pdf", "page": 1},
                {"source": "doc2.pdf", "page": 5}
            ]
        }
