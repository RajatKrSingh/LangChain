from functools import wraps
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

load_dotenv("config/.env")

def get_hf_embeddings():
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
    embeddings = HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_EMBEDDING_MODEL
    )
    return embeddings

def log_function(func):
    ''' Provide neat runtime print statements for marking beginning and end of functions
    '''
    BLUE = "\033[94m"
    RESET = "\033[0m"
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{BLUE}[START] {func.__name__}() started{RESET}")
        
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{BLUE}[END]   {func.__name__}() ended{RESET}")

    return wrapper

def to_obj(text):
    import json
    try:
        return json.loads(text)
    except Exception:
        return {}

def get_huggingface_models():
    ''' Provide name of all models for which inference provided and task of text-generation available
    '''
    from huggingface_hub import model_info

    model = "Qwen/Qwen2.5-0.5B"
    model = "mistralai/Mistral-7B-Instruct-v0.3"
    info = model_info(
        model,
        expand="inferenceProviderMapping"
    )
    # print(info)
    for provider in info.inference_provider_mapping:
        print(provider)
    # print(info.inference_provider_mapping)

@log_function
def get_llm_model(use_openai = False, host_model=False, use_gemini=False):
    ''' get Relevant model either from huggingface or openai
    '''
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")
    HUGGINGFACE_SERVERLESS_MODEL = os.getenv("HUGGINGFACE_SERVERLESS_MODEL")
    HUGGINGFACE_LOCAL_MODEL = os.getenv("HUGGINGFACE_LOCAL_MODEL")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if not use_openai and not host_model:
        from langchain_huggingface import HuggingFaceEndpoint
        from langchain_huggingface import ChatHuggingFace
        llm_model = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_SERVERLESS_MODEL,
            provider="featherless-ai",
            temperature=0.7,
            max_new_tokens=500,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_KEY
        )
        llm = ChatHuggingFace(llm=llm_model)
    elif use_gemini:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0,
            api_key = GEMINI_API_KEY
        )
    elif host_model:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import ( AutoTokenizer, AutoModelForCausalLM, pipeline )
        import torch

        GREEDY_DECODE_FLAG = True
        tokenizer = AutoTokenizer.from_pretrained(
            HUGGINGFACE_LOCAL_MODEL,
            token=HUGGINGFACEHUB_API_KEY
        )

        model = AutoModelForCausalLM.from_pretrained(
            HUGGINGFACE_LOCAL_MODEL,
            dtype=torch.float32,
            token=HUGGINGFACEHUB_API_KEY
        )
        if torch.backends.mps.is_available():
            model = model.to("mps")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            clean_up_tokenization_spaces=False,
            return_full_text=False
        )
        pipe.generation_config.max_new_tokens = 100
        pipe.generation_config.max_length = None
        if GREEDY_DECODE_FLAG:
            pipe.generation_config.do_sample = False
            pipe.generation_config.temperature = None
        else:
            pipe.generation_config.do_sample = True
        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name="gpt-5-nano")
    return llm

def main():
    get_huggingface_models()
    # hf_model_listing()

if __name__=="__main__":
    main()