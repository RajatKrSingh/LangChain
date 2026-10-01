from utils import get_llm_model
from langchain_core.messages import AIMessage

def fewshotprompt_demonstration():
    ''' Few shot prompting using plain python  
    ''' 
    llm = get_llm_model(False, True)
    prompt_input = """Classify the following numbers as Abra, Kadabra or Abra Kadabra.:

    3, 4, 5, 7, 8, 10, 11, 13, 35

    Examples: 
    6 // not divisible by 5, not divisible by 7 // None
    15 // divisible by 5, not divisible by 7 // Abra
    12 // not divisible by 5, not divisible by 7 // None
    21 // not divisible by 5, divisible by 7 // Kadabra
    70 // divisible by 5, divisible by 7 // Abra Kadabra
    """

    response = llm.invoke(prompt_input)
    if isinstance(response, AIMessage):
        response = response.content
    print(response)

def fewshotprompt_template_demonstration():
    ''' Few Shot Prompting using Langchain template
    '''
    from langchain_core.prompts.few_shot import FewShotPromptTemplate
    from langchain_core.prompts.prompt import PromptTemplate

    llm = get_llm_model(False, True)

    examples = [
        {
            "number": 6,
            "reasoning": "not divisible by 5 nor by 7",
            "result": "None"
        },
        {
            "number": 15,
            "reasoning": "divisible by 5 but not by 7",
            "result": "Abra"
        },
        {
            "number": 12,
            "reasoning": "not divisible by 5 nor by 7",
            "result": "None"
        },
        {
            "number": 21,
            "reasoning": "divisible by 7 but not by 5",
            "result": "Kadabra"
        },
        {
            "number": 70,
            "reasoning": "divisible by 5 and by 7",
            "result": "Abra Kadabra"
        }
    ]
    example_prompt = PromptTemplate(input_variables=["number", "reasoning", "result"], template="{number} \\ {reasoning} \\ {result}")
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Classify the following numbers as Abra, Kadabra or Abra Kadabra: {comma_delimited_input_numbers}",
        input_variables=["comma_delimited_input_numbers"]
    )

    prompt_input = few_shot_prompt.format(comma_delimited_input_numbers="3, 4, 5, 7, 8, 10, 11, 13, 35.")
    response = llm.invoke(prompt_input)
    if isinstance(response, AIMessage):
        response = response.content
    print(response)

def main():
    # fewshotprompt_demonstration()
    # fewshotprompt_template_demonstration()
    pass

if __name__=="__main__":
    main()