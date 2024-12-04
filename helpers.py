
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_groq import ChatGroq
import os
from pprint import pprint
from langchain.schema import Document
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun


### Router

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

ASTRA_DB_APPLICATION_TOKEN="<Enter Astra DB Token>" # enter the "AstraCS:..." string found in in your Token JSON file"
ASTRA_DB_ID="<Enter the Astra DB ID>"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)


# LLM with function call
groq_api_key='<Enter GROQ API KEY>'
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-70b-versatile")
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to debit card, credit card, masetrcard, VISA card, benefits of credit card, types of credit cards,stocks, mutual funds, euqity markets, future and Options, F&O, Index funds, ETFs, financial services, bonds, policies.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router




### Working With Tools
def wiki_wrapper():
    ## Arxiv and wikipedia Tools
    arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=5000)
    arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=5000)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

    return wiki

def retriever_object():
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    astra_vector_store=Cassandra(
        embedding=embeddings,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None)
    retriever=astra_vector_store.as_retriever()
    return retriever

## Graph

from typing import List
from typing_extensions import TypedDict
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]




def retrieve(state):
    """
    Retrieve documents and summarize the result.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved and summarized documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    retriever = retriever_object()

    documents = retriever.invoke(question)
    retrieved_content = documents[0].page_content

    summary = llm.invoke(f"""Summarize the following content in response to the question and return the output in a paragraph. 
    #Question: {question}
    Content: {retrieved_content}""")

    return {"documents": summary, "question": question}

def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    print("---HELLO--")
    question = state["question"]
    print(question)

    # Wiki search
    wiki = wiki_wrapper()
    docs = wiki.invoke({"query": question})
    #print(docs["summary"])
    wiki_results = docs
    # wiki_results = Document(page_content=wiki_results)
    summary = llm.invoke(f"""Summarize the following content in response to the question and return the output in a paragraph. 
    #Question: {question}
    Content: {wiki_results}""")
    # summary = llm.invoke(f"Summarize the following content in response to the question '{question}': {wiki_results}")
    return {"documents": summary, "question": question}




### Edges ###
def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"



from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "wiki_search", END)
# Compile
app = workflow.compile()













