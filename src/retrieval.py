"""
Query the index and generate answers using retrieved context
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.embeddings import load_index
from src.config import settings

# default settings
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_K = 4

PROMPT_TEMPLATE = """Use the following context to answer the question.
If you cannot answer based on the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0):
    """Initialize ChatOpenAI"""
    return ChatOpenAI(model=model, temperature=temperature)


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def create_chain(index, model: str = DEFAULT_MODEL, k: int = DEFAULT_K):
    """Create retrieval chain using LCEL"""
    llm = get_llm(model)
    retriever = index.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Store retriever for source document access
    chain.retriever = retriever
    return chain


def query(chain, question: str) -> dict:
    """Query the RAG system."""
    docs = chain.retriever.invoke(question)
    answer = chain.invoke(question)

    citations = [doc_to_citation(doc, i) for i, doc in enumerate(docs)]

    return {
        "answer": answer,
        "citations": citations  # Changed from "sources"
    }


def doc_to_citation(doc, index: int) -> dict:
    """Convert a LangChain Document to a citation dict.

    Adapter pattern: transforms LangChain's Document format
    to our stable API citation format.
    """
    return {
        "source": doc.metadata.get("source", "unknown"),
        "chunk_id": doc.metadata.get("chunk_id", f"chunk_{index}"),
        "snippet": doc.page_content[:200].strip(),
        "page": doc.metadata.get("page"),
        "section": doc.metadata.get("section"),
    }


if __name__ == "__main__":
    # load index and create chain
    index = load_index()
    chain = create_chain(index)

    # test query
    response = query(chain, "What is the total revenue?")
    print(f"Answer: {response['answer']}")
    print(f"Sources: {response["sources"]}")