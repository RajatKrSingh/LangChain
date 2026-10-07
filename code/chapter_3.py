from utils import get_llm_model, log_function
import yaml

with open("config/chapter_3.yaml", "r") as fd:
    CONFIG = yaml.safe_load(fd)

@log_function
def summarization_engine_single_document():
    ''' Summarize single document whose size is greater that the LLM context window
    '''
    from langchain_text_splitters import TokenTextSplitter
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnableParallel

    llm = get_llm_model(False, True)

    with open("auxillary_files/chapter_3/Moby-Dick.txt", "r") as fd:
        text_to_summarize = fd.read()
    # text_to_summarize = "This is the text I want to summarize"

    # STEP 1: Split text
    text_chunks_chain = (
        RunnableLambda(lambda x:
            [
                {
                    'chunk': text_chunk,
                }
                for text_chunk in TokenTextSplitter(chunk_size=3000, chunk_overlap=100).split_text(x)
            ]
        )
    )

    # STEP 2: Map all chunks
    summarize_chunk_prompt_template = CONFIG.get("summarize_chunk_prompt_template", "Return empty response")
    summarize_chunk_prompt = PromptTemplate.from_template(summarize_chunk_prompt_template)
    summary_chunk_chain = summarize_chunk_prompt | llm
    summarize_map_chain = (
        RunnableParallel (
            {
                "summary": summary_chunk_chain | StrOutputParser()
            }
        )
    )

    # STEP 3: Reduce all map chunks
    summarize_summaries_prompt_template = CONFIG.get("summarize_summaries_prompt", "Return empty response")
    summarize_summaries_prompt = PromptTemplate.from_template(summarize_summaries_prompt_template)
    summarize_reduce_chain = (
        RunnableLambda( lambda x:
            {
                "summaries": '\n'.join([i.get("summary", "") for i in x])
            }
        )|
        summarize_summaries_prompt |
        llm |
        StrOutputParser()
    )
    
    # STEP 4: Orchestrate entire map reduce chain
    map_reduce_chain = (
        text_chunks_chain |
        summarize_map_chain.map() |
        summarize_reduce_chain
    )

    # STEP 5: Call the map reduce chain
    summary = map_reduce_chain.invoke(
        text_to_summarize, 
        config = {"max_concurrency": 1}
    )
    print(summary)

@log_function
def summarization_engine_multiple_documents():
    ''' Summarize multiple documents using refine operation
    '''
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
    from operator import itemgetter

    def load_txt(file_path: str) -> list[Document]:
        ''' Load txt files
        '''
        text = ""
        with open(file_path, "r") as fd:
            text = fd.read()
        word_docs = Document(
            page_content=text,
            metadata = {
                "source":file_path
            }
        )
        return [word_docs]

    def load_docx(file_path: str) -> list[Document]:
        ''' Load docx files
        '''
        import docx
        doc_obj = docx.Document(file_path)
        # Extract text paragraph by paragraph
        paragraphs_text = []
        for paragraph in doc_obj.paragraphs:
            if paragraph.text.strip():  # Skip empty lines/paragraphs
                paragraphs_text.append(paragraph.text)
                
        # Combine paragraphs into a single text block
        full_text = "\n\n".join(paragraphs_text)
        
        # Return as standard LangChain Document
        if full_text.strip():
            doc = Document(
                page_content=full_text,
                metadata={"source": file_path}
            )
            return [doc]
            
        return []

    def load_pdf(file_path: str) -> list[Document]:
        ''' Load docx files
        '''
        from pypdf import PdfReader
        try:
            reader = PdfReader(file_path)
        except Exception as e:
            print(f"Error opening PDF file: {e}")
            return []

        documents = []
        full_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text.append(text)
            
        combined_text = "\n\n".join(full_text)
        if combined_text.strip():
            doc = Document(
                page_content=combined_text,
                metadata={"source": file_path, "total_pages": len(reader.pages)}
            )
            documents.append(doc)
                
        return documents

    def load_wiki(query: str, max_docs: int = 3) -> Document:
        ''' Load Wikipedia text into documents
        '''
        import wikipedia
        # Set the Wikipedia language
        wikipedia.set_lang("en")
        
        # 1. Search for pages matching the query
        try:
            search_results = wikipedia.search(query, results=max_docs)
        except Exception as e:
            print(f"Error searching Wikipedia: {e}")
            return []
        
        documents = []
        
        # 2. Fetch the content for each page found
        for title in search_results:
            try:
                # Fetch page content safely
                page = wikipedia.page(title, auto_suggest=False)
                
                # Construct standard LangChain Document
                doc = Document(
                    page_content=page.content,
                    metadata={
                        "title": page.title,
                        "source": page.url,
                        "summary": page.summary[:200] + "..."  # Snippet summary
                    }
                )
                documents.append(doc)
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle ambiguous titles by picking the first suggestion if desired, or skipping
                print(f"Disambiguation error for '{title}', skipping. Options: {e.options[:3]}")
                continue
            except wikipedia.exceptions.PageError:
                print(f"Page '{title}' not found, skipping.")
                continue
            except Exception as e:
                print(f"Could not load page '{title}': {e}")
                continue
                
        return documents

    word_docs = load_docx("auxillary_files/chapter_3/Paestum-Britannica.docx")

    pdf_docs = load_pdf("auxillary_files/chapter_3/PaestumRevisited.pdf")

    txt_docs = load_txt("auxillary_files/chapter_3/Paestum-Encyclopedia.txt")

    wikipedia_docs = load_wiki("Paestum", 2)

    all_docs = wikipedia_docs + word_docs + pdf_docs + txt_docs
    from langchain_core.prompts import PromptTemplate
    llm = get_llm_model(False, True)

    doc_summary_template = CONFIG.get("doc_summary_template")
    doc_summary_prompt = PromptTemplate.from_template(doc_summary_template)
    doc_summary_chain = RunnableParallel(
        {
            "summary": doc_summary_prompt | llm | StrOutputParser(),
            "current_refined_summary": itemgetter("current_refined_summary")
        }
    )

    refine_summary_template = CONFIG.get("refine_summary_template")
    refine_summary_prompt = PromptTemplate.from_template(refine_summary_template)
    refine_chain = (
        refine_summary_prompt | 
        llm | 
        StrOutputParser()
    )

    consolidated_chain = ( doc_summary_chain | refine_chain
    )

    def refine_summary(docs):
        ''' Refine summary based on the current refined summary and new docs
        '''
        intermediate_steps = []
        current_refined_summary = ''
        for idx, doc in enumerate(docs):
            intermediate_step = {
                "current_refined_summary" : current_refined_summary,
            }
            intermediate_steps.append(intermediate_step)
            current_refined_summary = consolidated_chain.invoke(
                {"text":doc.page_content,
                 "current_refined_summary":current_refined_summary},
                config = {"max_concurrency": 1}
            )
        return {"final_summary": current_refined_summary, "intermediate_steps": intermediate_steps}

    full_summary = refine_summary(all_docs)
    print(full_summary)



def main():
    summarization_engine_single_document()
    # summarization_engine_multiple_documents()

if __name__=="__main__":
    main()