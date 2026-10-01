from utils import get_llm_model, log_function, to_obj
from typing import List

class SummarizationEngine:
    @log_function
    def __init__(self, question):
        import yaml
        from langchain_core.prompts import PromptTemplate

        # SET PROMPTS
        with open("config/chapter_4.yaml", "r") as fd:
            self.CONFIG = yaml.safe_load(fd)

        assistant_selection_prompt_template = self.CONFIG.get("ASSISSTANT_SELECTION_INSTRUCTIONS")
        self.assistant_selection_prompt = PromptTemplate.from_template(assistant_selection_prompt_template)

        web_search_prompt_template = self.CONFIG.get("WEB_SEARCH_INSTRUCTIONS")
        self.web_search_prompt = PromptTemplate.from_template(web_search_prompt_template)

        summary_prompt_template = self.CONFIG.get("SUMMARY_INSTRUCTIONS")
        self.summary_prompt = PromptTemplate.from_template(summary_prompt_template)

        research_prompt_template = self.CONFIG.get("RESEARCH_REPORT_INSTRUCTIONS")
        self.research_prompt = PromptTemplate.from_template(research_prompt_template)

        # SET CONSTANTS
        self.NUM_SEARCH_QUERIES = 2
        self.NUM_SEARCH_RESULTS_PER_QUERY = 2
        self.RESULT_TEXT_MAX_CHARACTERS = 10000
        self.question = question


    @log_function
    def web_search(web_query: str, num_results: int)-> List[str]:
        from ddgs import DDGS

        with DDGS() as ddgs:
            return [x["href"] for x in list(ddgs.text(web_query, max_results = num_results))]
        return []

    @log_function
    def web_scrape(url: str)-> str:
        import requests
        from bs4 import BeautifulSoup
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9"
            }
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                page_text = soup.get_text(separator=" ", strip=True)
                return page_text
            else:
                return f"Could not retrieve the webpage : Status code {response.status_code}"

        except Exception as e:
            print(e)
            return f"Could not retrieve the webpage: {e}"

    @log_function
    def run_without_lcel(self):
        ''' Run the summarization engine
        '''
        llm = get_llm_model()

        # GET LLM assistant for the queries
        assistant_selection_prompt = self.assistant_selection_prompt.format(user_question=self.question)
        assistant_instructions = llm.invoke(assistant_selection_prompt)
        assistant_instructions_dict = to_obj(assistant_instructions.content)

        # GET Web Search Queries from query rewriting
        web_search_prompt = self.web_search_prompt.format(
            assistant_instructions = assistant_instructions_dict['assistant_instructions'],
            num_search_queries = self.NUM_SEARCH_QUERIES,
            user_question = assistant_instructions_dict['user_question']
        )
        web_search_queries = llm.invoke(web_search_prompt)
        web_search_queries_list = to_obj(web_search_queries.content.replace('\n', ''))

        # GET URLS from web search
        searches_and_result_urls = [{
                'result_urls': SummarizationEngine.web_search(
                    web_query=wq['search_query'],
                    num_results=self.NUM_SEARCH_RESULTS_PER_QUERY
                ),
                'search_query': wq['search_query']
            }
            for wq in web_search_queries_list
        ]

        search_query_and_result_url_list = [] # Flatten results
        for query in searches_and_result_urls:
            search_query_and_result_url_list.extend([{
                'search_query' : query['search_query'],
                'result_url' : x
            } for x in query['result_urls']
        ])

        # Scrape URLs for results
        result_text_list = [
            {
                'result_text': SummarizationEngine.web_scrape(
                        url=x['result_url'],
                    )[:self.RESULT_TEXT_MAX_CHARACTERS],
                'result_url': x['result_url'],
                'search_query': x['search_query']
            } for x in search_query_and_result_url_list
        ]

        # Summarize web results
        result_text_summary_list = []
        for x in result_text_list:
            summary_prompt = self.summary_prompt.format(
                search_result_text = x['result_text'],
                search_query = x['search_query']
            )

            text_summary = llm.invoke(summary_prompt)
            result_text_summary_list.append({
                'text_summary': text_summary,
                'result_url': x['result_url'],
                'search_query': x['search_query']
            })

        # Generate Research Report
        stringified_summary_list = [
            f'Source URL: {x["result_url"]}\nSummary: {x["text_summary"]}'
                for x in result_text_summary_list
        ]
        appended_summaries = '\n'.join(stringified_summary_list)
        research_report_prompt = self.research_prompt.format(
            research_summary=appended_summaries,
            user_question=self.question
        )
        research_report = llm.invoke(research_report_prompt)
        print(f'strigified_summary_list={stringified_summary_list}')
        print(f'merged_result_summaries={appended_summaries}')
        print(f'research_report={research_report}')

    @log_function
    def run_with_lcel(self):
        ''' Run Summarization engine with LCEL
        '''
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

        llm = get_llm_model()

        # GET LLM assistant for the queries
        assistant_instructions_chain = ( 
            {'user_question': RunnablePassthrough()}
            | self.assistant_selection_prompt 
            | llm
            | StrOutputParser()
            | to_obj 
        )

        # GET Web Search Queries from query rewriting
        web_searches_chain = (
            RunnableLambda(lambda x:
                {
                    'assistant_instructions': x['assistant_instructions'],
                    'num_search_queries': self.NUM_SEARCH_QUERIES,
                    'user_question': x['user_question']
                }
            )
            | self.web_search_prompt
            | llm
            | StrOutputParser()
            | to_obj
        )

        # GET URLS from web search
        search_result_urls_chain = (
            RunnableLambda(lambda x:
                [
                    {
                        'result_url': url,
                        'search_query': x['search_query'],
                        'user_question': x['user_question']
                    }
                    for url in SummarizationEngine.web_search(
                        web_query = x['search_query'],
                        num_results = self.NUM_SEARCH_RESULTS_PER_QUERY
                    )
                ]
            )
        )

        # Scrape URLs for results and summarize results
        search_result_text_and_summary_chain = (
            RunnableLambda( lambda x:
                {
                    'search_result_text': self.web_scrape(url = x['result_url'])[:self.RESULT_TEXT_MAX_CHARACTERS],
                    'result_url': x['result_url'],
                    'search_query': x['search_query'],
                    'user_question': x['user_question']
                }
            )
            | RunnableParallel(
                {
                    'text_summary': ( self.summary_prompt
                        | llm
                        | StrOutputParser()
                    ),
                    'result_url': lambda x: x['result_url'],
                    'user_question': lambda x: x['user_question']
                }
            )
            | RunnableLambda(lambda x:
                {
                    'summary': f"Source Url: {x['result_url']}\nSummary: {x['text_summary']}",
                    'user_question': x['user_question']
                }
            )
        )

        search_and_summarization_chain = (
            search_result_urls_chain
            | search_result_text_and_summary_chain.map()
            | RunnableLambda(lambda x:
                {
                    'summary': '\n'.join([i['summary'] for i in x]),
                    'user_question': x[0]['user_question'] if len(x)>0 else ''
                }
            )
        )

        # Generate Research Report and create consolidated web research chain
        web_research_chain = (
            assistant_instructions_chain
            | web_searches_chain
            | search_and_summarization_chain.map()
            | RunnableLambda(lambda x:
                {
                    'research_summary': '\n\n'.join([i['summary'] for i in x]),
                    'user_question': x[0]['user_question'] if len(x)>0 else ''
                }
            )
            | self.research_prompt
            | llm
            | StrOutputParser()
        )

        web_research_report = web_research_chain.invoke(self.question)
        print(web_research_report)


def main():
    question = "How many titles did Michael Jordan win?"
    # web_query_result = SummarizationEngine.web_search(
    #     web_query = question,
    #     num_results = 2
    # )
    # print(web_query_result)

    # web_scraping_result = SummarizationEngine.web_scrape(web_query_result[0])
    # print(web_scraping_result)


    summarization_engine = SummarizationEngine(question)
    summarization_engine.run_without_lcel()

    # summarization_engine.run_with_lcel()

if __name__ == "__main__":
    main()