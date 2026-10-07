from utils import log_function, get_llm_model

@log_function
def testing():
    ''' Test application without calling LLM APIs which can be costly
    '''
    from langchain_core.language_models import FakeListLLM

    fake_llm = FakeListLLM(responses=["Hello, this is fake LLM response"])
    result = fake_llm.invoke("Any input will return Hello")
    print(result)


def raw_openai_api_demonstration():
    ''' Invoke OpenAI api to get result to query in recommended way
    '''
    import getpass
    from openai import OpenAI

    OPENAI_API_KEY = getpass.getpass("Enter your OPENAI_API_KEY")
    client = OpenAI(api_key = OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "How many greek temples are there in Paestum?"
            }
        ],
        temperature=0.7
    )
    print(completion.choices[0].message.content)

def raw_local_mistral_demonstration():
    ''' Invoke local LLM using openai wrapper - using the chat.completions API
    '''
    from openai import OpenAI
    port_number = '8080'
    client = OpenAI(
        base_url=f'http://localhost:{port_number}/v1',
        api_key="NO_KEY_NEEDED"
    )
    completion = client.chat.completions.create(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant"
            },
            {
                "role": "user",
                "content": "How many greek temples are there in Paestum?"
            }
        ],
        temperature=0.7
    )
    print(completion.choices[0].message.content)

@log_function
def langchain_local_mistral_demonstration():
    ''' Invoke local mistral using LangChain rather that the OpenAI chat.completions api
    '''
    llm = get_llm_model(False, True)
    response = llm.invoke("How many greek temples are there in Paestum?\n")
    print(response)

@log_function
def langchain_gemini_demonstration():
    ''' Invoke Gemini api for Chat Model
    '''
    llm = get_llm_model(False, False, True)
    response = llm.invoke("How many greek temples are there in Paestum?\n")
    print(response)

@log_function
def langchain_sentence_completion():
    ''' Perform sentence completion
    '''
    llm = get_llm_model()
    response = llm.invoke("Complete the following sentence naturally."
    "Do not output any text before or after the completed sentence"
    "Output only the continuation, without explanation:\n\n" 
    "It's a hot day, I would like to go to the ...")
    print(response.content)

def prompttemplate_demonstration():
    ''' Use PromptTemplate for prompting LLM
    '''
    from langchain_core.prompts import PromptTemplate
    llm = get_llm_model()
    
    segovia_aqueduct_text = """The Aqueduct of Segovia 
    (Spanish: Acueducto de Segovia) is a Roman aqueduct in Segovia, 
    Spain. It was built around the first century AD to channel water 
    from springs in the mountains 17 kilometres (11 mi) away to the 
    city's fountains, public baths and private houses, and was in 
    use until 1973. Its elevated section, with its complete arcade 
    of 167 arches, is one of the best-preserved Roman aqueduct 
    bridges and the foremost symbol of Segovia, as evidenced by 
    its presence on the city's coat of arms. The Old Town of 
    Segovia and the aqueduct, were declared a UNESCO World 
    Heritage Site in 1985. As the aqueduct lacks a legible 
    inscription (one was apparently located in the structure's 
    attic, or top portion[citation needed]), the date of 
    construction cannot be definitively determined. The general 
    date of the Aqueduct's construction was long a mystery, 
    although it was thought to have been during the 1st century AD, 
    during the reigns of the Emperors Domitian, Nerva, and Trajan. 
    At the end of the 20th century, Géza Alföldy deciphered the 
    text on the dedication plaque by studying the anchors that held 
    the now missing bronze letters in place. He determined that Emperor 
    Domitian (AD 81–96) ordered its construction[1] and the year 98 AD
    was proposed as the most likely date of completion.[2] However, 
    in 2016 archeological evidence was published which points to a 
    slightly later date, after 112 AD, during the government of 
    Trajan or in the beginning of the government of emperor Hadrian, 
    from 117 AD."""

    prompt_template = PromptTemplate.from_template(""""You are an exprerienced
    copywriter. Write a {num_words} words summary of the following text, using a {tone}: {text}""")

    prompt_input = prompt_template.format(
        text=segovia_aqueduct_text,
        num_words=20,
        tone="knowledgeable and engaging"
    )
    response = llm.invoke(prompt_input)
    print(response.content)

def chaining_demonstration():
    ''' Use chaining to create workflow
    '''
    from langchain_core.prompts import PromptTemplate

    llm = get_llm_model()

    segovia_aqueduct_text = """The Aqueduct of Segovia 
        (Spanish: Acueducto de Segovia) is a Roman aqueduct in Segovia, 
        Spain. It was built around the first century AD to channel water 
        from springs in the mountains 17 kilometres (11 mi) away to the 
        city's fountains, public baths and private houses, and was in 
        use until 1973. Its elevated section, with its complete arcade 
        of 167 arches, is one of the best-preserved Roman aqueduct 
        bridges and the foremost symbol of Segovia, as evidenced by 
        its presence on the city's coat of arms. The Old Town of 
        Segovia and the aqueduct, were declared a UNESCO World 
        Heritage Site in 1985. As the aqueduct lacks a legible 
        inscription (one was apparently located in the structure's 
        attic, or top portion[citation needed]), the date of 
        construction cannot be definitively determined. The general 
        date of the Aqueduct's construction was long a mystery, 
        although it was thought to have been during the 1st century AD, 
        during the reigns of the Emperors Domitian, Nerva, and Trajan. 
        At the end of the 20th century, Géza Alföldy deciphered the 
        text on the dedication plaque by studying the anchors that held 
        the now missing bronze letters in place. He determined that Emperor 
        Domitian (AD 81–96) ordered its construction[1] and the year 98 AD
        was proposed as the most likely date of completion.[2] However, 
        in 2016 archeological evidence was published which points to a 
        slightly later date, after 112 AD, during the government of 
        Trajan or in the beginning of the government of emperor Hadrian, 
        from 117 AD."""

    prompt_template = PromptTemplate.from_template("""You are an experienced copywriter. Write a {num_words} words
    summary fo the following text, using a {tone} tone: {text}""")

    chain = prompt_template | llm
    response = chain.invoke(
        {
            "text": segovia_aqueduct_text,
            "num_words": 20,
            "tone": "knowledgeable and engaging"
        }
    )
    print(response.content)

def main():
    # testing()
    # raw_openai_api_demonstration()
    # raw_local_mistral_demonstration()
    langchain_local_mistral_demonstration()
    langchain_gemini_demonstration()
    # langchain_sentence_completion()
    # prompttemplate_demonstration()
    # chaining_demonstration()
    pass

if __name__=="__main__":
    main()