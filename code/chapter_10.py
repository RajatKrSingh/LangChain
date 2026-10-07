from utils import log_function, get_hf_embeddings, get_llm_model
from langchain_chroma import Chroma
from chapter_8 import html_to_text_transform, load_html_pages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Literal, Optional, Tuple, List
from pydantic import BaseModel, Field
from langchain_classic.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain_classic.retrievers.self_query.chroma import ChromaTranslator
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, BaseOutputParser
import asyncio
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sqlalchemy import create_engine, text

with open("config/chapter_10.yaml", "r") as fd:
    CONFIG = yaml.safe_load(fd)

uk_destinations = [
    ("Cornwall", "Cornwall"), ("North_Cornwall", "Cornwall"), 
    ("South_Cornwall", "Cornwall"), ("West_Cornwall", "Cornwall"),
    ("Tintagel", "Cornwall"), ("Bodmin", "Cornwall"), 
    ("Wadebridge", "Cornwall"),
    ("Penzance", "Cornwall"), ("Newquay", "Cornwall"), 
    ("St_Ives", "Cornwall"),
    ("Port_Isaac", "Cornwall"), ("Looe", "Cornwall"), 
    ("Polperro", "Cornwall"),
    ("Porthleven", "Cornwall"),
    ("East_Sussex", "East_Sussex"), ("Brighton", "East_Sussex"),
    ("Battle", "East_Sussex"), ("Hastings_(England)", "East_Sussex"),
    ("Rye_(England)", "East_Sussex"), ("Seaford", "East_Sussex"), 
    ("Ashdown_Forest", "East_Sussex")
]

wikivoyage_root_url = "https://en.wikivoyage.org/wiki"
uk_destination_url_with_metadata = [
    ( f'{wikivoyage_root_url}/{destination}', destination, region)
    for destination, region in uk_destinations
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
embedding_model = get_hf_embeddings()

uk_with_metadata_collection = Chroma(
    collection_name="uk_with_metadata_collection",
    embedding_function=embedding_model
)
uk_with_metadata_collection.reset_collection()

async def enrich_docs_with_metadata():
    for (url, destination, region) in uk_destination_url_with_metadata:
        docs = await load_html_pages(url)
        
        docs_with_metadata = [
            Document(page_content=d.page_content,
            metadata = {
                'source': url,
                'destination': destination,
                'region': region})
            for d in docs]
                
        chunks = split_docs_into_chunks(docs_with_metadata)
        uk_with_metadata_collection.add_documents(documents=chunks)

def split_docs_into_chunks(docs):
    text_docs = html_to_text_transform(docs)
    chunks = text_splitter.split_documents(text_docs)
    return chunks


def qna_enrich_metadata():
    asyncio.run(enrich_docs_with_metadata())
    question =  "Events or festivals"
    metadata_retriever = uk_with_metadata_collection.as_retriever(
        search_kwargs={'k':2, 'filter':{'destination': 'Newquay'}})

    result_docs = metadata_retriever.invoke(question)
    print(result_docs[0])

def self_query_retriever():
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
    from langchain_classic.chains.query_constructor.base import AttributeInfo

    metadata_field_info = [
        AttributeInfo(
            name="destination",
            description="The specific UK destination to be searched",
            type="string",
        ),
        AttributeInfo(
            name="region",
            description="The name of the UK region to be searched",
            type="string",
        )
    ]
    question = "Tell me about events or festivals in the UK town of Newquay"
    llm = get_llm_model(False, True)
    self_query_retriever = SelfQueryRetriever.from_llm(
        llm, uk_with_metadata_collection, question, 
        metadata_field_info, verbose=True
    )
    result_docs = self_query_retriever.invoke(question)
    print(result_docs[0])

class DestinationSearch(BaseModel):
    """Search over a vector database of tourist destinations."""

    content_search: str = Field(
        "",
        description="""Similarity search query applied 
        to tourist destinations.""",
    )
    destination: str = Field(
        ...,
        description="The specific UK destination to be searched.",
    )
    region: str = Field(
        ...,
        description="The name of the UK region to be searched.",
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(
                self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

def build_filter(destination_search: DestinationSearch):
    comparisons = []

    destination = destination_search.destination
    region = destination_search.region
    
    if destination and destination != '':
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="destination",
                value=destination,
            )
        )
    if region and region != '':
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="region",
                value=region,
            )
        )    

    search_filter = Operation(operator=Operator.AND, 
                              arguments=comparisons)

    chroma_filter = ChromaTranslator().visit_operation(
        search_filter)
        
    return chroma_filter

def query_using_llm():
    import datetime
    system_message_template = CONFIG.get("system_message_template")
    system_message_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_template),
            ("human", "{question}"),
        ]
    )
    llm = get_llm_model(False, True)
    structured_llm = llm.with_structured_output(
        DestinationSearch, method="function_calling"
    )
    query_generator = system_message_prompt | structured_llm

    question = "Tell me about events or festivals in the UK town of Newquay"
    structured_query =query_generator.invoke(question)
    search_query = structured_query.content_search
    search_filter = build_filter(structured_query)
    metadata_retriever = uk_with_metadata_collection.as_retriever(
        search_kwargs={'k':3, 'filter': search_filter})
    answer = metadata_retriever.invoke(search_query)
    print(answer)

def execute_sql_query(engine, query):
    with engine.connect() as connection:
        result = connection.execute(text(query))
        return str(result.fetchall())

class RouteQuery(BaseModel):
    """Route a user question to the most relevant datasource."""

    datasource: Literal["tourist_info_store", "uk_booking_db"] = Field(
        ...,
        description="""Given a user question, 
        route it either to a tourist info vector store 
        or a UK accomodation booking relational database.""",
    )


def structured_sql_query():

    engine = create_engine("sqlite:///auxillary_files/chapter_10/UkBooking.db")
    llm = get_llm_model(False, True)
    question = "Give me some offers for Cardiff, including the hotel name"
    clean_sql_prompt_template = CONFIG.get("clean_sql_prompt_template")
    clean_sql_prompt = ChatPromptTemplate.from_template(clean_sql_prompt_template)
    clean_sql_chain = clean_sql_prompt | llm

    full_sql_gen_chain = llm | StrOutputParser() | clean_sql_chain | StrOutputParser()

    tourist_info_retriever_chain = RunnableLambda(
        lambda x: x['question']) \
        | uk_with_metadata_collection.as_retriever(
            search_kwargs={'k':2})

    sql_query_exec_chain = RunnableLambda(
        lambda query_result: execute_sql_query(engine, query_result)
    )

    uk_accommodation_retriever_chain =  full_sql_gen_chain \
        | sql_query_exec_chain | StrOutputParser()


    parser = PydanticOutputParser(pydantic_object=RouteQuery)
    # structured_llm_router = llm.with_structured_output(RouteQuery)
    travel_system_message_template_prompt = CONFIG.get("travel_system_message_template")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", travel_system_message_template_prompt),
            ("human", "{question}")
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    question_router_chain = (
        prompt
        | llm
        | StrOutputParser()
        | parser
    )

    selected_data_source = question_router_chain.invoke(
        {"question": "Have you got any offers in Brighton?"}
    )
    
    retriever_chains = {
        "tourist_info_store": tourist_info_retriever_chain,
        'uk_booking_db': uk_accommodation_retriever_chain
    }
    def retriever_chooser(question):
        selected_data_source = question_router_chain.invoke({"question": question})
        return retriver_chains[selected_data_source.datasource]

    chosen = retriever_chooser("Tell me about events or festivals in the UK town of Newquay")
    rag_prompt_template = CONFIG.get("rag_prompt_template")
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template) 

    def execute_rag_chain(question, chosen_retriever):
        full_rag_chain = (
            {
                "context": {"question": RunnablePassthrough()} 
                    | chosen_retriever,
                "question": RunnablePassthrough(),
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        return full_rag_chain.invoke(question)

    question = """Give me some offers for Cardiff, 
    including the accommodation name"""
    chosen_retriever = retriever_chooser(question)
    answer = execute_rag_chain(question, chosen_retriever)
    print(answer)

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse out a question from each output line."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  

def reciprocal_rank_fusion(results_groups:list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple groups of 
        ranked documents and an optional parameter k used in 
        the Reciprocal Rank Fusion (RRF) formula """

    fused_scores = {}

    for results_group in results_groups:
        for local_rank, doc in enumerate(results_group):

            if doc not in fused_scores:
                fused_scores[doc] = 0

            fused_scores[doc] += 1 / (local_rank + k)

    reranked_results = [
        (doc, score)
        for doc, score in sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]
    return reranked_results

def rag_fusion():
    multi_query_gen_prompt_template = CONFIG.get("multi_query_gen_prompt_template")
    multi_query_gen_prompt = ChatPromptTemplate.from_template(multi_query_gen_prompt_template)

    question_parser = LineListOutputParser()

    llm = get_llm_model(False, True)
    multi_query_gen_chain = multi_query_gen_prompt | llm | question_parser

    retriever = uk_with_metadata_collection.as_retriever(search_kwargs={'k':3})
    top_three_results = RunnableLambda(
        lambda x: x[0:3])
    rag_fusion_retrieval_chain =multi_query_gen_chain \
        | retriever.map() | reciprocal_rank_fusion \
        | top_three_results
            
    docs = rag_fusion_retrieval_chain.invoke(
        {"question": question})
    print(docs)

    rag_prompt_rag_fusion_template = CONFIG.get("rag_prompt_template_rag_fusion")
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_rag_fusion_template) 

    rag_chain = (
        {
            "context": {"question": RunnablePassthrough()} | rag_fusion_retrieval_chain,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    user_question = "Can you give me some tips for a trip to Brighton?"
    answer = rag_chain.invoke(user_question)
    print(answer)

def main():
    # qna_enrich_metadata()
    # self_query_retriever()
    # query_using_llm()
    # structured_sql_query()
    rag_fusion()

if __name__=="__main__":
    main()