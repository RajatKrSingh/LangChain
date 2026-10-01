from langchain_core.documents import Document
from utils import log_function, get_llm_model, get_hf_embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, yaml

with open("config/chapter_7.yaml", "r") as fd:
    CONFIG = yaml.safe_load(fd)

@log_function
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

@log_function
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

@log_function
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

@log_function
def load_wiki(query: str, max_docs: int = 3) -> Document:
    ''' Load Wikipedia text into documents
    '''
    import wikipedia
    # Set the Wikipedia language
    wikipedia.set_lang("en")
    wikipedia.set_user_agent("MyLangChainProject/1.0 (your-email@example.com)")
    
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

@log_function
def split_and_import(vector_db, text_splitter, loader):
    chunks = text_splitter.split_documents(loader)
    vector_db.add_documents(chunks)
    print(f"Ingested chunks created")

loader_classes = {
    'docx': load_docx,
    'pdf' : load_pdf,
    'txt' : load_txt
}

@log_function
def get_loader(filename):
    _, file_extension = os.path.splitext(filename) #A Extract the file extension
    file_extension = file_extension.lstrip('.') #B Remove the leading dot from the extension
    
    loader_class = loader_classes.get(
        file_extension) #C Get the loader class from the dictionary
    
    if loader_class:
        return loader_class(filename) #D Instantiate and return the correct loader
    else:
        raise ValueError(f"No loader available for file extension '{file_extension}'")

@log_function
def ingest_files_from_folder(folder_path):
    for filename in os.listdir(folder_path): #B iterate over the files in the path
        file_path = os.path.join(folder_path, filename) #C Construct the full path to the file
    
        if os.path.isfile(file_path): #D Check if it is a file (not a directory)
            try:
                loader = get_loader(file_path) #E Instantiate the correct loader for the file
                print(f"Loader for {filename}: {loader}")
                split_and_import(loader) #F Split and ingest
            except ValueError as e:
                print(e)

@log_function
def q_n_a_engine_brute():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0)
    embeddings_model = get_hf_embeddings()
    vector_db = Chroma("tourist_info", embedding_function=embeddings_model)
    wiki_loader = load_wiki("Paestum")
    word_loader = load_docx("./auxillary_files/chapter_7/Paestum-Britannica.docx")
    pdf_loader = load_pdf("./auxillary_files/chapter_7/PaestumRevisited.pdf")
    txt_loader = load_txt("./auxillary_files/chapter_7/Paestum-Encyclopedia.txt")
    split_and_import(vector_db, text_splitter, word_loader)
    split_and_import(vector_db, text_splitter, pdf_loader)
    split_and_import(vector_db, text_splitter, txt_loader)
    split_and_import(vector_db, text_splitter, wiki_loader)

    query = "Where was Poseidonia and who renamed it to Paestum?"
    results = vector_db.similarity_search(query, 4)
    print(results)

@log_function
def q_n_a_engine_chain():
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    question = "Where was Poseidonia and who renamed it to Paestum?"

    embeddings_model = get_hf_embeddings()
    vector_db = Chroma("tourist_info", embedding_function=embeddings_model)

    rag_prompt_template = CONFIG.get("rag_prompt_template", "")
    rag_prompt = PromptTemplate.from_template(rag_prompt_template)

    retriever = vector_db.as_retriever()
    question_feeder = RunnablePassthrough()
    llm = get_llm_model(False, True)

    rag_chain = {
        "context" : retriever,
        "question": question_feeder
    }|rag_prompt |llm

    answer = rag_chain.invoke(question)
    print(answer.content)

@log_function
def q_n_a_engine_history():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    question = "Where was Poseidonia and who renamed it to Paestum?"
    
    embeddings_model = get_hf_embeddings()
    vector_db = Chroma("tourist_info", embedding_function=embeddings_model)

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant, world-class expert in Roman and Greek history, especially in towns located in southern Italy. Provide interesting insights on local history and recommend places to visit with knowledgeable and engaging answers. Answer all questions to the best of your ability, but only use what has been provided in the context. If you don't know, just say you don't know. Use three sentences maximum and keep the answer as concise as possible."),
            ("placeholder", "{chat_history_messages}"),
            ("assistant", "{retrieved_context}"),
            ("human", "{question}"),
        ]
    )

    retriever = vector_db.as_retriever()
    question_feeder = RunnablePassthrough()
    llm = get_llm_model(False, True)
    chat_history_memory = InMemoryChatMessageHistory()

    def get_messages(x):
        return chat_history_memory.messages

    rag_chain = {
        "retrieved_context": retriever, 
        "question": question_feeder,
        "chat_history_messages": RunnableLambda(get_messages)
    } | rag_prompt | llm

    def execute_chain_with_memory(chain, question):
        chat_history_memory.add_user_message(question)
        answer = chain.invoke(question)
        chat_history_memory.add_ai_message(answer)
        print(f'Full chat message history: {chat_history_memory.messages}\n\n')                                      
        return answer

    answer = execute_chain_with_memory(rag_chain, question)
    print(answer)

    question = """And then what did they do? Also tell me the source""" 
    answer = execute_chain_with_memory(rag_chain, question)
    print("Followup: ",answer)

def main():
    # q_n_a_engine_brute()
    # q_n_a_engine_chain()
    q_n_a_engine_history()

if __name__ == "__main__":
    main()
     



