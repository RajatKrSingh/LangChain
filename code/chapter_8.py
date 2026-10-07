from typing import List
import uuid
from langchain_chroma import Chroma
from utils import get_llm_model, get_hf_embeddings, log_function
import httpx
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import asyncio
from pydantic import BaseModel, Field

uk_destinations = [
    "Cornwall", "North_Cornwall", "South_Cornwall", "West_Cornwall", 
    "Tintagel", "Bodmin", "Wadebridge", "Penzance", "Newquay",
    "St_Ives", "Port_Isaac", "Looe", "Polperro", "Porthleven"
    "East_Sussex", "Brighton", "Battle", "Hastings_(England)", 
    "Rye_(England)", "Seaford", "Ashdown_Forest"
]
uk_destinations = ["Cornwall","Brighton"]

wikivoyage_root_url = "https://en.wikivoyage.org/wiki"
uk_destination_urls = [f'{wikivoyage_root_url}/{d}' for d in uk_destinations]

@log_function
def advanced_rag():
    llm = get_llm_model(False, True)
    embedding_model = get_hf_embeddings()
    uk_granular_collection = Chroma(
        collection_name = "uk_granular",
        embedding_function = embedding_model
    )
    uk_granular_collection.reset_collection()

    uk_coarse_collection = Chroma(
        collection_name = "uk_coarse",
        embedding_function=embedding_model
    )
    uk_coarse_collection.reset_collection()

    asyncio.run(load_html_pages_into_collections(uk_destination_urls, uk_granular_collection, uk_coarse_collection))
    granular_results = uk_granular_collection.similarity_search(
        query="Events or festivals in East Sussex",k=4)
    print(granular_results[0])

@log_function
async def load_html_pages_into_collections(uk_destination_urls, uk_granular_collection, uk_coarse_collection):
    for destination_url in uk_destination_urls:
        docs = await load_html_pages(destination_url)
        
        for doc in docs:
            granular_chunks = split_docs_into_granular_chunks(docs)
            uk_granular_collection.add_documents(documents=granular_chunks)

            coarse_chunks = split_docs_into_coarse_chunks(docs)
            uk_coarse_collection.add_documents(documents=coarse_chunks)

@log_function
def split_docs_into_coarse_chunks(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 300
    )
    text_docs = html_to_text_transform(docs)
    coarse_chunks = text_splitter.split_documents(
        text_docs
    )
    return coarse_chunks

def html_to_text_transform(docs: List[Document]) -> List[Document]:
    import html2text

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True

    transformed_docs = []

    for doc in docs:
        new_doc = doc.model_copy(deep=True)
        new_doc.page_content = converter.handle(doc.page_content)
        transformed_docs.append(new_doc)
    return docs

@log_function
def split_docs_into_granular_chunks(docs):
    from langchain_text_splitters import HTMLSectionSplitter
    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
    html_section_splitter = HTMLSectionSplitter(
        headers_to_split_on=headers_to_split_on
    )
    all_chunks = []
    for doc in docs:
        html_string = doc.page_content
        temp_chunks = html_section_splitter.split_text(
            html_string
        )
        all_chunks.extend(temp_chunks)

    return all_chunks


async def load_html_pages(url: str) -> Document :
    headers = {
        "User-Agent": "MyTravelResearchBot/1.0 (contact: your-email@example.com)"
    }
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30,
        headers = headers) as client:
        response = await client.get(url)
        try:
            response.raise_for_status()
        except:
            return []

    soup = BeautifulSoup(response.text, "html.parser")

    return [Document(
        page_content=soup.get_text(separator="\n", strip=True),
        metadata={"source": url}
    )]

async def load_html_into_parent_collections(parent_doc_retriever, uk_destination_urls):
    for destination_url in uk_destination_urls:
        docs = await load_html_pages(destination_url)
        text_docs = html_to_text_transform(docs)
        if len(text_docs)>0:
            parent_doc_retriever.add_documents(text_docs, ids=None)

async def load_html_into_multi_vector(multi_vector_retriever , parent_splitter, child_splitter, uk_destination_urls, doc_key):
    for destination_url in uk_destination_urls:
        docs = await load_html_pages(destination_url)
        text_docs = html_to_text_transform(docs)

        coarse_chunks = parent_splitter.split_documents(text_docs)
        coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]

        all_granular_chunks = []
        for i, coarse_chunk in enumerate(coarse_chunks):
            coarse_chunk_id = coarse_chunks_ids[i]
            granular_chunks = child_splitter.split_documents([coarse_chunk])

            for granular_chunk in granular_chunks:
                granular_chunk.metadata[doc_key] = coarse_chunk_id

            all_granular_chunks.extend(granular_chunks)
        if len(all_granular_chunks)>0:
            multi_vector_retriever.vectorstore.add_documents(all_granular_chunks)
            multi_vector_retriever.docstore.mset(list(zip(coarse_chunks_ids, coarse_chunks)))


def advanced_rag_parent_document_retriever():
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_classic.storage import InMemoryStore
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)

    embedding_model = get_hf_embeddings()
    child_chunks_collection = Chroma(
        collection_name = "uk_child_chunks",
        embedding_function = embedding_model
    )

    child_chunks_collection.reset_collection()

    doc_store = InMemoryStore()
    parent_doc_retriever = ParentDocumentRetriever(
        vectorstore = child_chunks_collection,
        docstore = doc_store,
        child_splitter = child_splitter,
        parent_splitter = parent_splitter
    )
    asyncio.run(load_html_into_parent_collections(parent_doc_retriever, parent_splitter, child_splitter, uk_destination_urls))

    
def advanced_rag_multi_vector_retriever():
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    from langchain_classic.storage import InMemoryByteStore
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embedding_function = get_hf_embeddings()
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
    child_chunks_collection = Chroma(
        collection_name="uk_child_chunks",
        embedding_function=embedding_function
    )
    child_chunks_collection.reset_collection()
    doc_byte_store = InMemoryByteStore()
    doc_key = "doc_id"

    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=child_chunks_collection, 
        byte_store=doc_byte_store
    )

    asyncio.run(load_html_into_multi_vector(multi_vector_retriever, parent_splitter, child_splitter, uk_destination_urls, doc_key))
    child_docs_only = child_chunks_collection.similarity_search("Cornwall Ranger")
    print(child_docs_only[0])

async def load_html_for_summaries(summarization_chain, parent_splitter, multi_vector_retriever, uk_destination_urls, doc_key):
    for destination_url in uk_destination_urls:
        html_docs = await load_html_pages(destination_url)
        docs = html_to_text_transform(html_docs)
        coarse_chunks = parent_splitter.split_documents(docs)

        coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]
        all_summaries = []
        for i, coarse_chunk in enumerate(coarse_chunks):
            coarse_chunk_id = coarse_chunks_ids[i]
            summary_text = summarization_chain.invoke(coarse_chunk)
            summary_doc = Document(
                page_content=summary_text,
                metadata={doc_key: coarse_chunk_id}
            )
            all_summaries.append(summary_doc)
        multi_vector_retriever.vectorstore.add_documents(all_summaries)
        multi_vector_retriever.docstore.mset(list(zip(coarse_chunks_ids, coarse_chunks)))

def advanced_rag_summary_multi_vector_retriever():
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    from langchain_classic.storage import InMemoryByteStore
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    embedding_function = get_hf_embeddings()
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
    summaries_collection = Chroma(
        collection_name = "uk_summaries",
        embedding_function = embedding_function
    )
    summaries_collection.reset_collection()
    
    doc_byte_store = InMemoryByteStore()
    doc_key = "doc_id"

    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=summaries_collection,
        byte_store=doc_byte_store
    )

    llm = get_llm_model(False, True)
    summarization_chain = (
        {"document": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{document}")
        | llm
        | StrOutputParser()
    )

    asyncio.run(load_html_for_summaries(summarization_chain, parent_splitter ,multi_vector_retriever, uk_destination_urls, doc_key))

    summary_docs_only =  summaries_collection.similarity_search("Cornwall Travel")
    print(summary_docs_only[0])

class HypotheticalQuestions(BaseModel):
    questions: List[str] = Field(..., description="List of hypothetical questions for given text")

async def load_html_into_hypotheticalquestions_retriever(multi_vector_retriever, parent_splitter, uk_destination_urls, hypothetical_questions_chain):
    for destination_url in uk_destination_urls:
        html_docs = await load_html_pages(destination_url)
        text_docs =  html_to_text_transform(html_docs)

        coarse_chunks = parent_splitter.split_documents(text_docs)
        coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]
        all_hypothetical_questions = []
        for i, coarse_chunk in enumerate(coarse_chunks):
            coarse_chunk_id = coarse_chunks_ids[i]
            hypothetical_questions = hypothetical_questions_chain.invoke(coarse_chunk)
            hypothetical_questions_docs = [
                Document(
                    page_content=question, 
                    metadata={doc_key: coarse_chunk_id}
                )
                for question in hypothetical_questions
            ]
            all_hypothetical_questions.extend(hypothetical_questions_docs)

        multi_vector_retriever.vectorstore.add_documents(all_hypothetical_questions)
        multi_vector_retriever.docstore.mset(list(zip(coarse_chunks_ids, coarse_chunks)))

def advanced_rag_hypothetical_questions_retriever():
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    from langchain_classic.storage import InMemoryByteStore
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(
        pydantic_object=HypotheticalQuestions
    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
    embedding_function = get_hf_embeddings()

    hypothetical_questions_collection = Chroma(
        collection_name = "uk_hypothetical_questions",
        embedding_function=embedding_function
    )

    doc_byte_store = InMemoryByteStore()
    doc_key = "doc_id"

    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=hypothetical_questions_collection,
        byte_store=doc_byte_store
    )

    llm = get_llm_model(False, True)
    hypothetical_questions_chain = (
        {"document_text": lambda x: x.page_content} 
        | ChatPromptTemplate.from_template( 
            "Generate a list of exactly 4 hypothetical questions that the below text could be used to answer:\n\n{document_text}"
        ).partial(format_instructions=parser.get_format_instructions())
        | llm 
        | parser
        | (lambda x: x.questions) 
    )
    asyncio.run(load_html_into_hypotheticalquestions_retriever(multi_vector_retriever, parent_splitter, uk_destination_urls, hypothetical_questions_chain))

    hypothetical_question_docs_only = hypothetical_questions_collection.similarity_search("How can you go to Brighton from London?")
    print(hypothetical_question_docs_only[0])

async def load_html_rag_chunk_retriever(multi_vector_retriever, granular_chunk_splitter, uk_destination_urls, doc_key):
    for destination_url in uk_destination_urls:
        html_docs =  load_html_pages(destination_url)
        text_docs = html_to_text_transform(html_docs)

        granular_chunks = granular_chunk_splitter.split_documents(text_docs)
        expanded_chunk_store_items = []
        for i, granular_chunk in enumerate(granular_chunks):

            this_chunk_num = i
            previous_chunk_num = i-1
            next_chunk_num = i+1
            
            if i==0:
                previous_chunk_num = None
            elif i==(len(granular_chunks)-1):
                next_chunk_num = None

            expanded_chunk_text = ""
            if previous_chunk_num:
                expanded_chunk_text += granular_chunks[
                    previous_chunk_num].page_content
                expanded_chunk_text += "\n"

            expanded_chunk_text += granular_chunks[
                this_chunk_num].page_content
            expanded_chunk_text += "\n"

            if next_chunk_num:
                expanded_chunk_text += granular_chunks[
                    next_chunk_num].page_content
                expanded_chunk_text += "\n"

            expanded_chunk_id = str(uuid.uuid4())
            expanded_chunk_doc = Document(page_content=expanded_chunk_text)

            expanded_chunk_store_item = (expanded_chunk_id, expanded_chunk_doc)
            expanded_chunk_store_items.append(expanded_chunk_store_item)

            granular_chunk.metadata[doc_key] = expanded_chunk_id

        multi_vector_retriever.vectorstore.add_documents(granular_chunks)
        multi_vector_retriever.docstore.mset(expanded_chunk_store_items)

def advanced_rag_chunk_expansion_retriever():
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    from langchain_classic.storage import InMemoryByteStore
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    granular_chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    embedding_model = get_hf_embeddings()
    granular_chunks_collection = Chroma(
        collection_name="uk_granular_chunks",
        embedding_function=embedding_model,
    )

    granular_chunks_collection.reset_collection()
    expanded_chunk_store = InMemoryByteStore()
    doc_key = "doc_id"

    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=granular_chunks_collection,
        byte_store=expanded_chunk_store
    )
    asyncio.run(load_html_rag_chunk_retriever(multi_vector_retriever, granular_chunk_splitter, uk_destination_urls, doc_key))

    retrieved_docs = multi_vector_retriever.invoke("Cornwall Ranger")
    print(retrieved_docs[0])


def main():
    # advanced_rag()
    # advanced_rag_parent_document_retriever()
    # advanced_rag_multi_vector_retriever()
    # advanced_rag_summary_multi_vector_retriever()
    # advanced_rag_hypothetical_questions_retriever()
    advanced_rag_chunk_expansion_retriever()

if __name__=="__main__":
    main()