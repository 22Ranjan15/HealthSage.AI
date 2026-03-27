from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.app.services.retrieval import RetrievalService
from backend.app.agents.state import AgentState
from config.config import Config

# --- 1. Initialize Tools & Models ---

# The Vector Search
vector_retriever = RetrievalService()

# The Web Search (Tavily)
web_tool = TavilySearchResults(max_results=3, tavily_api_key=Config.TAVILY_API_KEY)

# The LLM (Brain)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=Config.GEMINI_API_KEY
)

# --- 2. Node Functions ---

async def retrieve_node(state: AgentState):
    """
    The Researcher Agent: Decides whether to use Vector DB or Web Search.
    (For V1, we will default to Vector, and fallback to Web if empty).
    """
    question = state["question"]
    print(f"---RESEARCHER: Searching for '{question}'---")

    # 1. Try Vector Search First (Trusted Source)
    docs = await vector_retriever.asearch(question, k=3)
    
    # 2. Logic: If Vector DB gives results, use them.
    if docs and len(docs) > 0:
        print("---RESEARCHER: Found data in Vector DB---")
        content = [d.page_content for d in docs]
        return {"documents": content, "source": "vector"}
    
    # 3. Fallback: If Vector DB is empty, use Tavily (Web)
    print("---RESEARCHER: Vector DB empty. Switching to Tavily Web Search---")
    try:
        web_results = web_tool.invoke({"query": question})
        # Tavily returns a list of dicts [{'content': '...', 'url': '...'}, ...]
        content = [res['content'] for res in web_results]
        return {"documents": content, "source": "web"}
    except Exception as e:
        print(f"---RESEARCHER: Web Search Failed: {e}---")
        return {"documents": ["Error: Could not retrieve information."], "source": "error"}

async def generate_node(state: AgentState):
    """
    The Generator Agent: Writes the answer based on the retrieved docs.
    """
    print("---GENERATOR: Writing Answer---")
    question = state["question"]
    documents = state["documents"]
    
    # Simple Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Question: {question} 
        Context: {context} 
        
        Answer:"""
    )
    
    # Chain
    chain = prompt | llm
    
    # Format documents into a single string
    context_str = "\n\n".join(documents)
    
    response = await chain.ainvoke({"question": question, "context": context_str})
    
    return {"generation": response.content}