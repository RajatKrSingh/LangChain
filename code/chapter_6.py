import chromadb
from utils import get_llm_model, log_function

@log_function
def rag_naive():
    chroma_client = chromadb.Client()
    llm = get_llm_model(False, True)
    tourism_collection = chroma_client.create_collection(name="tourism_collection")

    tourism_collection.add(
        documents=[
            "Paestum, Greek Poseidonia, ancient city in southern Italy near the west coast, 22 miles (35 km) southeast of modern Salerno and 5 miles (8 km) south of the Sele (ancient Silarus) River. Paestum is noted for its splendidly preserved Greek temples.", 
            "Poseidonia was probably founded about 600 BC by Greek colonists from Sybaris, along the Gulf of Taranto, and it had become a flourishing town by 540, judging from its temples. After many years’ resistance the city came under the domination of the Lucanians (an indigenous Italic people) sometime before 400 BC, after which its name was changed to Paestum. Alexander, the king of Epirus, defeated the Lucanians at Paestum about 332 BC, but the city remained Lucanian until 273, when it came under Roman rule and a Latin colony was founded there. The city supported Rome during the Second Punic War. The locality was still prosperous during the early years of the Roman Empire, but the gradual silting up of the mouth of the Silarus River eventually created a malarial swamp, and Paestum was finally deserted after being sacked by Muslim raiders in AD 871. The abandoned site’s remains were rediscovered in the 18th century.",
            "The ancient Greek part of Paestum consists of two sacred areas containing three Doric temples in a remarkable state of preservation. During the ensuing Roman period a typical forum and town layout grew up between the two ancient Greek sanctuaries. Of the three temples, the Temple of Athena (the so-called Temple of Ceres) and the Temple of Hera I (the so-called Basilica) date from the 6th century BC, while the Temple of Hera II (the so-called Temple of Neptune) was probably built about 460 BC and is the best preserved of the three. The Temple of Peace in the forum is a Corinthian-Doric building begun perhaps in the 2nd century BC. Traces of a Roman amphitheatre and other buildings, as well as intersecting main streets, have also been found. The circuit of the town walls, which are built of travertine blocks and are 15–20 feet (5–6 m) thick, is about 3 miles (5 km) in circumference. In July 1969 a farmer uncovered an ancient Lucanian tomb that contained Greek frescoes painted in the early classical style. Paestum’s archaeological museum contains these and other treasures from the site."
        ],
        metadatas=[
            {"source": "https://www.britannica.com/place/Paestum"}, 
            {"source": "https://www.britannica.com/place/Paestum"},
            {"source": "https://www.britannica.com/place/Paestum"}
        ],
        ids=["paestum-br-01", "paestum-br-02", "paestum-br-03"]
    )

    regular_question = "How many Doric temples are in Paestum?"
    trick_question = "How many columns have the 3 temples got in total?"

    # Regular Question RAG
    results = query_vector_database(tourism_collection, regular_question)
    prompt = prompt_template(regular_question, results)
    prompt_response = execute_llm_prompt(prompt, llm)
    print("Regular Question", prompt_response)

    # Trick Question RAG
    results = query_vector_database(tourism_collection, trick_question)
    prompt = prompt_template(trick_question, results)
    prompt_response = execute_llm_prompt(prompt, llm)
    print("Trick Question", prompt_response)

@log_function
def execute_llm_prompt(prompt_input, llm):
    prompt_response = llm.invoke(prompt_input)
    return prompt_response


def prompt_template(question, text):
    return f'Read the following text and answer this question: {question}. \nText: {text}'

@log_function
def query_vector_database(collection, question):
    results = collection.query(
        query_texts=[question],
        n_results=1
    )
    print(results)
    return results


def main():
    rag_naive()

if __name__=="__main__":
    main()