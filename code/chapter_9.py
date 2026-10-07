from utils import log_function, get_hf_embeddings, get_llm_model
from langchain_chroma import Chroma
import yaml
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

with open("config/chapter_9.yaml", "r") as fd:
    CONFIG = yaml.safe_load(fd)

def question_rewrite():
    embedding_model = get_hf_embeddings()
    uk_granular_collection = Chroma(
        collection_name="uk_granular",
        embedding_function=embedding_model,
    )

    uk_granular_collection.reset_collection()

    llm = get_llm_model(False, True)
    rewriter_prompt_template = CONFIG.get("rewriter_prompt_template")
    rewriter_prompt = ChatPromptTemplate.from_template(rewriter_prompt_template)

    rewriter_chain = rewriter_prompt | llm | StrOutputParser()

    rag_prompt_template = CONFIG.get("rag_prompt_template")
    rag_prompt = ChatPromptTemplate.from_template(
        rag_prompt_template) 

    retriever = uk_granular_collection.as_retriever()
    rewrite_retrieve_read_rag_chain = (
        {
            "context": {"user_question": RunnablePassthrough()} 
                | rewriter_chain | retriever,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    user_question = "Tell me some things I can do in Cornwall"
    answer = rewrite_retrieve_read_rag_chain.invoke(user_question)
    print(answer)

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse out a question from each output line."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  



def multi_query_retriever():
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    embedding_model = get_hf_embeddings()
    uk_granular_collection = Chroma(
        collection_name="uk_granular",
        embedding_function=embedding_model,
    )

    uk_granular_collection.reset_collection()

    multi_query_gen_prompt_template = CONFIG.get(multi_query_gen_prompt_template)
    multi_query_gen_prompt = ChatPromptTemplate.from_template(multi_query_gen_prompt_template)
    questions_parser = LineListOutputParser()
    llm = get_llm_model(False, True)

    multi_query_gen_chain = multi_query_gen_prompt | llm | questions_parser

    basic_retriever = uk_granular_collection.as_retriever()

    multi_query_retriever = MultiQueryRetriever(
        retriever=basic_retriever, llm_chain=multi_query_gen_chain, 
        parser_key="lines"
    )

    user_question = "Tell me some fun things I can do in Cornwall"
    retrieved_docs = multi_query_retriever.invoke(user_question)
    print(retrieved_docs[0])

    # USE STANDARD MULTIQUERYRETRIEVER  
    std_multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=basic_retriever, llm=llm
    )
    retrieved_docs = std_multi_query_retriever.invoke(user_question)
    print(retrieved_docs[0])

def step_back_question():
    embedding_model = get_hf_embeddings()
    llm = get_llm_model(False, True)
    step_back_prompt_template = CONFIG.get("step_back_prompt_template")
    step_back_prompt = ChatPromptTemplate.from_template(step_back_prompt_template)
    step_back_question_gen_chain = step_back_prompt | llm | StrOutputParser()
    uk_granular_collection = Chroma(
        collection_name="uk_granular",
        embedding_function=embedding_model,
    )

    uk_granular_collection.reset_collection()
    retriever = uk_granular_collection.as_retriever()
    rag_prompt_template = CONFIG.get("rag_prompt_template")
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    step_back_question_rag_chain = (
        {
            "context": {"detailed_question": RunnablePassthrough()} 
            | step_back_question_gen_chain | retriever,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    user_question = "Can you give me some tips for a trip to Brighton?"
    answer = step_back_question_rag_chain.invoke(user_question)
    print(answer)

def hyde():
    embedding_model = get_hf_embeddings()
    llm = get_llm_model(False, True)
    hyde_prompt_template = CONFIG.get("hyde_prompt_template")
    hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)
    hyde_chain = hyde_prompt | llm | StrOutputParser()
    uk_granular_collection = Chroma(
        collection_name="uk_granular",
        embedding_function=embedding_model,
    )

    uk_granular_collection.reset_collection()
    retriever = uk_granular_collection.as_retriever()

    rag_prompt_template = CONFIG.get("rag_prompt_template")
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    hyde_rag_chain = (
        {
            "context": {"question": RunnablePassthrough()} 
            | hyde_chain | retriever,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    user_question = "What are the best beaches in Cornwall?"
    answer = hyde_rag_chain.invoke(user_question)
    print(answer)


def main():
    # question_rewrite()
    # multi_query_retriever()
    # step_back_question()
    hyde()

if __name__=="__main__":
    main()