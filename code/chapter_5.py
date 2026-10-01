from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from utils import log_function, get_llm_model
from chapter_4 import SummarizationEngine
import json

''' DEFINE ALL DICTIONARIES REQUIRED FOR LANGGRAPH
'''
class AssistantInfo(TypedDict):
    assistant_type: str
    assistant_instructions: str
    user_question: str

class SearchQuery(TypedDict):
    search_query: str
    user_question: str

class SearchResult(TypedDict):
    result_url: str
    search_query: str
    user_question: str
    is_fallback: Optional[bool]

class SearchSummary(TypedDict):
    summary: str
    result_url: str
    user_question: str
    is_fallback: Optional[bool]

class ResearchReport(TypedDict):
    report: str

class ResearchState(TypedDict):
    user_question: str
    assistant_info: Optional[AssistantInfo]
    search_queries: Optional[List[SearchQuery]]
    search_results: Optional[List[SearchResult]]
    search_summaries: Optional[List[SearchSummary]]
    research_summary: Optional[str]
    final_report: Optional[str]
    used_fallback_search: Optional[bool]
    relevance_evaluation: Optional[Dict[str, Any]]
    should_regenerate_queries: Optional[bool]
    iteration_count: Optional[int]

class SummarizationEngineGraph:
    @log_function
    def __init__(self):
        import yaml
        with open("config/chapter_5.yaml", "r") as fd:
            self.CONFIG = yaml.safe_load(fd)
        self.NUM_SEARCH_QUERIES = 3
        self.NUM_SEARCH_RESULTS_PER_QUERY = 3
        self.RESULT_TEXT_MAX_CHARACTERS = 10000

    @log_function
    def generate_search_queries(self, state: Dict[str, Any])-> Dict[str, Any]:
        ''' Generate search queries based on assistant instruction and user question
        '''
        assistant_info = state["assistant_info"]
        user_question = state["user_question"]
        assistant_instructions = assistant_info["assistant_instructions"]

        iteration_count = state.get("iteration_count", 0)

        previous_queries = state.get("state_queries", [])
        relevance_evaluation = state.get("relevant_evaluation", None)

        if iteration_count == 0:
            print("Generating initial search queries...")
            prompt_template = self.CONFIG["WEB_SEARCH_INSTRUCTIONS_INITIAL"]
            prompt = PromptTemplate.from_template(prompt_template)
            prompt = prompt.format(
                assistant_instructions = assistant_instructions,
                user_question = user_question,
                num_search_queries = self.NUM_SEARCH_QUERIES
            )
        elif iteration_count == 1:
            print("First Regeneration: Creating more specific queries...")
            previous_query_list = ", ".join([q["search_query"] for q in previous_queries])
            relevance_percentage = relevance_evaluation.get("relevance_percentage", 0) if relevance_evaluation else 0
            relevance_explanation = relevance_evaluation.get("explanation", "No explanation provided") if relevance_evaluation else ""

            prompt_template = self.CONFIG["WEB_SEARCH_INSTRUCTIONS_ITERATION_1"]
            prompt = PromptTemplate.from_template(prompt_template)
            prompt = prompt.format(
                assistant_instructions = assistant_instructions,
                user_question = user_question,
                previous_query_list=previous_query_list,
                relevance_percentage=relevance_percentage,
                relevance_explanation=relevance_explanation,
                NUM_SEARCH_QUERIES=self.NUM_SEARCH_QUERIES
            )
        else:
            print(f"Iteration {iteration_count}: Using alternative search strategies...")
            all_previous_queries = ", ".join([q["search_query"] for q in previous_queries])
            prompt_template = self.CONFIG["WEB_SEARCH_INSTRUCTIONS_ITERATION_GT_1"]
            prompt = PromptTemplate.from_template(prompt_template)
            prompt = prompt.format(
                assistant_instructions = assistant_instructions,
                user_question = user_question,
                all_previous_queries=all_previous_queries,
                NUM_SEARCH_QUERIES=self.NUM_SEARCH_QUERIES
            )
        llm = get_llm_model()
        response = llm.invoke(prompt)
        response_text = response.content
        
        # Parse the response to get the search queries
        try:
            # Extract the JSON array from the response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]
            
            # Parse the JSON
            search_queries = json.loads(json_str)
            
            print(f"Generated {len(search_queries)} search queries")
            for i, query in enumerate(search_queries):
                print(f"  Query {i+1}: {query['search_query']}")
            
            # Return the updated state
            return {
                "search_queries": search_queries,
                # Reset the relevance evaluation and regeneration flag when generating new queries
                "relevance_evaluation": None,
                "should_regenerate_queries": None
            }
        except Exception as e:
            print(f"Error parsing search queries: {str(e)}")
            # Fallback to a default search query if parsing fails
            default_queries = [
                {"search_query": f"{user_question} iteration {iteration_count + 1}", "user_question": user_question}
            ]
            print(f"Using default query: {default_queries[0]['search_query']}")
            return {
                "search_queries": default_queries,
                "relevance_evaluation": None,
                "should_regenerate_queries": None
            }

    @log_function
    def perform_web_searches(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform web searches based on the generated search queries.
        """
        search_queries = state["search_queries"]
        search_results = []
        fallback_used = False
        
        print(f"Performing web searches for {len(search_queries)} queries...")
        
        # For each search query, get the search results
        for query_obj in search_queries:
            search_query = query_obj["search_query"]
            user_question = query_obj["user_question"]
            
            try:
                # Get the search results
                print(f"Searching for: {search_query}")
                urls = SummarizationEngine.web_search(web_query=search_query, num_results=self.NUM_SEARCH_RESULTS_PER_QUERY)
                
                # Check if these are likely fallback results (Wikipedia URLs)
                if any("wikipedia.org" in url for url in urls[:2]):
                    print(f"Fallback search was used for query: {search_query}")
                    fallback_used = True
                    is_fallback = True
                else:
                    is_fallback = False
                
                # Add the results to the list
                for url in urls:
                    search_results.append({
                        "result_url": url,
                        "search_query": search_query,
                        "user_question": user_question,
                        "is_fallback": is_fallback
                    })
                    
                print(f"Found {len(urls)} results for query: {search_query}")
            except Exception as e:
                print(f"Error searching for '{search_query}': {str(e)}")
                # Continue with other queries even if one fails
                continue
        
        # If we have no search results at all, add a fallback result
        if not search_results:
            print("No search results found. Using general fallback information.")
            fallback_url = "https://en.wikipedia.org/wiki/Main_Page"
            search_results.append({
                "result_url": fallback_url,
                "search_query": "general information",
                "user_question": state["user_question"],
                "is_fallback": True
            })
            fallback_used = True
        
        # Return the updated state with information about fallback usage
        return {
            "search_results": search_results,
            "used_fallback_search": fallback_used
        }

    @log_function
    def summarize_search_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize the search results.
        """
        search_results = state["search_results"]
        used_fallback_search = state.get("used_fallback_search", False)
        llm = get_llm_model()
        summaries = []
        
        print(f"Summarizing {len(search_results)} search results...")
        
        # For each search result, get the text and summarize it
        for result in search_results:
            result_url = result["result_url"]
            search_query = result["search_query"]
            user_question = result["user_question"]
            is_fallback = result.get("is_fallback", False)
            
            try:
                # Get the webpage content
                print(f"Scraping content from: {result_url}")
                search_result_text = SummarizationEngine.web_scrape(url=result_url)[:self.RESULT_TEXT_MAX_CHARACTERS]
                
                # Skip if the content is an error message or too short
                if search_result_text.startswith("Failed to") or len(search_result_text) < 50:
                    print(f"Skipping {result_url} due to scraping issues or insufficient content")
                    continue
                
                # Format the prompt, with additional context for fallback results
                if is_fallback:
                    prompt_template = self.CONFIG["SUMMARY_INSTRUCTIONS_FALLBACK"]
                else:
                    prompt_template = self.CONFIG["SUMMARY_INSTRUCTIONS"]
                prompt = PromptTemplate.from_template(prompt_template)
                prompt = prompt.format(
                    search_result_text=search_result_text,
                    search_query=search_query
                )
                
                # Get the summary
                summary_response = llm.invoke(prompt)
                text_summary = summary_response.content
                
                # Add a note about fallback sources
                if is_fallback:
                    source_note = "[Note: This information comes from a fallback source and may not directly address the question.]"
                    text_summary = f"{text_summary}\n{source_note}"
                
                # Create the summary object
                summary = {
                    "summary": f"Source Url: {result_url}\nSummary: {text_summary}",
                    "result_url": result_url,
                    "user_question": user_question,
                    "is_fallback": is_fallback
                }
                
                summaries.append(summary)
                print(f"Successfully summarized content from: {result_url}")
            except Exception as e:
                print(f"Error summarizing {result_url}: {str(e)}")
                # Skip this result if there's an error
                continue
        
        # Create the research summary
        if summaries:
            research_summary = "\n\n".join([s["summary"] for s in summaries])
            print(f"Created research summary with {len(summaries)} sources")
            
            # Add a note if fallback search was used
            if used_fallback_search:
                fallback_note = "\n\n[Note: Some or all of this information comes from fallback sources because the primary search engine was unavailable. The information may not be as directly relevant to your question as usual.]"
                research_summary += fallback_note
        else:
            research_summary = "No relevant information found. Please try different search queries."
            print("Warning: No summaries were generated from search results")
        
        # Return the updated state
        return {
            "search_summaries": summaries,
            "research_summary": research_summary,
            "used_fallback_search": used_fallback_search
        }

    @log_function
    def evaluate_search_relevance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the relevance of search summaries to the original question.
        If less than 50% of summaries are relevant, return to search query generation.
        """
        search_summaries = state.get("search_summaries", [])
        user_question = state["user_question"]
        research_summary = state.get("research_summary", "")
        
        print("Evaluating relevance of search summaries to the original question...")
        
        # If there are no summaries, we need to regenerate queries
        if not search_summaries or not research_summary:
            print("No search summaries found. Regenerating search queries...")
            return {"should_regenerate_queries": True}
        
        # Use the LLM to evaluate relevance
        llm = get_llm_model()
        
        # Create a prompt for the LLM to evaluate relevance
        evaluation_prompt_template = self.CONFIG["EVALUATION_INSTRUCTIONS"]
        evaluation_prompt = PromptTemplate.from_template(evaluation_prompt_template)
        evaluation_prompt = evaluation_prompt.format(
            user_question = user_question,
            research_summary = research_summary
        )
        
        try:
            # Get the evaluation from the LLM
            evaluation_response = llm.invoke(evaluation_prompt)
            evaluation_text = evaluation_response.content
            
            # Extract the JSON from the response
            try:
                # Find JSON in the response
                json_start = evaluation_text.find('{')
                json_end = evaluation_text.rfind('}') + 1
                json_str = evaluation_text[json_start:json_end]
                
                # Parse the JSON
                evaluation = json.loads(json_str)
                relevance_percentage = evaluation.get("relevance_percentage", 0)
                
                # Determine if we should regenerate queries (less than 50% relevant)
                should_regenerate = relevance_percentage < 50
                
                if should_regenerate:
                    print(f"Only {relevance_percentage}% of search results are relevant. Regenerating search queries...")
                else:
                    print(f"{relevance_percentage}% of search results are relevant. Proceeding to write research report...")
                
                return {
                    "relevance_evaluation": evaluation,
                    "should_regenerate_queries": should_regenerate
                }
            except Exception as e:
                print(f"Error parsing relevance evaluation: {str(e)}")
                # If we can't parse the evaluation, assume we need to regenerate
                return {"should_regenerate_queries": True}
        except Exception as e:
            print(f"Error during relevance evaluation: {str(e)}")
            # If there's an error, assume we need to regenerate
            return {"should_regenerate_queries": True}

    @log_function
    def select_assistant(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ''' Select appropriate research assistant based on user question
        '''
        user_question = state["user_question"]
        prompt_template = self.CONFIG.get("ASSISSTANT_SELECTION_INSTRUCTIONS")
        prompt = PromptTemplate.from_template(prompt_template)
        prompt = prompt.format(user_question=user_question)

        llm = get_llm_model()
        response = llm.invoke(prompt)
        response_text = response.content

        try:
            # Extract the JSON part from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            
            # Parse the JSON
            assistant_info = json.loads(json_str)
            
            # Return the updated state
            return {"assistant_info": assistant_info}
        except Exception as e:
            # Fallback to a default assistant if parsing fails
            default_assistant = {
                "assistant_type": "General research assistant",
                "assistant_instructions": "You are a general research AI assistant. Your main purpose is to draft comprehensive, informative, unbiased, and well-structured reports on given topics.",
                "user_question": user_question
            }
            return {"assistant_info": default_assistant}

    @log_function
    def write_research_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write a research report based on the summarized search results.
        """
        research_summary = state["research_summary"]
        user_question = state["user_question"]
        
        # Format the prompt
        prompt_template = self.CONFIG["RESEARCH_REPORT_INSTRUCTIONS"]
        prompt = PromptTemplate.from_template(prompt_template)
        prompt = prompt.format(
            research_summary=research_summary,
            user_question=user_question
        )
        
        # Get the LLM response
        llm = get_llm_model()
        response = llm.invoke(prompt)
        report = response.content
        
        # Return the updated state
        return {"final_report": report}
    
    @log_function
    def create_research_graph(self) -> StateGraph:
        ''' Create LangGraph research graph that coordinates the agents
        '''
        graph = StateGraph(ResearchState)

        # Add nodes to the graph
        graph.add_node("select_assistant", self.select_assistant)
        graph.add_node("generate_search_queries", self.generate_search_queries)
        graph.add_node("perform_web_searches", self.perform_web_searches)
        graph.add_node("summarize_search_results", self.summarize_search_results)
        graph.add_node("evaluate_search_relevance", self.evaluate_search_relevance)
        graph.add_node("write_research_report", self.write_research_report)

        # Define the conditional routing function for relevance evaluation
        def route_based_on_relevance(state: Dict[str, Any]) -> str:
            """
            Route to either generate new search queries or continue to report writing
            based on the relevance evaluation.
            """
            # Get the current iteration count
            iteration_count = state.get("iteration_count", 0)
            
            # Increment the iteration count
            new_iteration_count = iteration_count + 1
            
            # Update the state with the new iteration count
            state["iteration_count"] = new_iteration_count
            
            # Check if we've reached the maximum number of iterations (3)
            if new_iteration_count >= 3:
                print(f"Reached maximum iterations ({new_iteration_count}). Proceeding to write report with current results.")
                return "write_research_report"
            
            # Otherwise, check if we should regenerate queries
            if state.get("should_regenerate_queries", False):
                print(f"Iteration {new_iteration_count}: Regenerating search queries.")
                return "generate_search_queries"
            else:
                print(f"Iteration {new_iteration_count}: Search results are relevant. Proceeding to write report.")
                return "write_research_report"
        
        # Define the flow of the graph
        graph.add_edge("select_assistant", "generate_search_queries")
        graph.add_edge("generate_search_queries", "perform_web_searches")
        graph.add_edge("perform_web_searches", "summarize_search_results")
        graph.add_edge("summarize_search_results", "evaluate_search_relevance")
        
        # Add conditional routing based on relevance evaluation
        graph.add_conditional_edges(
            "evaluate_search_relevance",
            route_based_on_relevance,
            {
                "generate_search_queries": "generate_search_queries",
                "write_research_report": "write_research_report"
            }
        )
        
        graph.add_edge("write_research_report", END)
        
        # Set the entry point
        graph.set_entry_point("select_assistant")
        
        return graph

    @log_function
    def run_research(self, question: str) -> str:
        """
        Run the research graph with a user question.
        
        Args:
            question: The user's research question
            
        Returns:
            The final research report
        """
        # Create the graph
        research_graph = self.create_research_graph()
        
        # Compile the graph
        app = research_graph.compile()
        
        # Initialize the state
        initial_state = {
            "user_question": question,
            "assistant_info": None,
            "search_queries": None,
            "search_results": None,
            "search_summaries": None,
            "research_summary": None,
            "final_report": None,
            "used_fallback_search": False,
            "relevance_evaluation": None,
            "should_regenerate_queries": None,
            "iteration_count": 0
        }
        
        # Run the graph
        result = app.invoke(initial_state)
        
        # Extract and return the final report
        return result["final_report"]
    
    def main():
        question = "What can you tell me about Astorga's roman spas"
        obj = SummarizationEngine()
        report = obj.run_research(question)
        print(report)

    # For testing purposes
    if __name__ == "__main__":
        main()
    