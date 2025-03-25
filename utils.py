# utils.py

import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from crewai import Agent, Task, Crew, Process, LLM
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

# Settings
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "finance-chatbot"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Connect to the existing Qdrant collection
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME
)

# Initialize Mistral LLM
mistral_llm = LLM(model="mistral/mistral-large-latest", api_key=MISTRAL_API_KEY, temperature=0.7)

# Functions
@lru_cache(maxsize=100)
def search_qdrant(query, top_k=3):
    """Search Qdrant for relevant documents."""
    try:
        retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        results = retriever.invoke(query)
        return [{"text": doc.page_content, "source": doc.metadata.get("source", "Unknown")} for doc in results]
    except Exception:
        return []

def search_news(query, max_results=5):
    """Search for recent financial news using Serper API."""
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "q": f"{query} finance news",
            "num": max_results
        }
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = data.get("organic", [])
        if not results:
            return [{"title": "No recent news available", "url": "", "snippet": "Could not fetch news. Please try again later."}]

        formatted_results = [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")
            }
            for item in results[:max_results]
        ]
        return formatted_results

    except ConnectionError:
        return [{"title": "Connection Error", "url": "", "snippet": "Failed to connect to the news API. Please check your internet connection."}]
    except Timeout:
        return [{"title": "Timeout Error", "url": "", "snippet": "News API request timed out. Please try again later."}]
    except HTTPError as e:
        if response.status_code == 429:
            return [{"title": "Rate Limit Exceeded", "url": "", "snippet": "Too many requests to the news API. Please try again later."}]
        return [{"title": "HTTP Error", "url": "", "snippet": f"Failed to fetch news due to HTTP error: {e}"}]
    except Exception:
        return [{"title": "Error", "url": "", "snippet": "An unexpected error occurred while fetching news. Please try again later."}]

def get_stock_data(symbol):
    """Fetch stock data using Alpha Vantage API."""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get("Global Quote", {})
        if not data:
            return {"symbol": symbol, "error": "No data found for this symbol."}
        return {
            "symbol": symbol,
            "price": data.get("05. price", "N/A"),
            "change": data.get("09. change", "N/A"),
            "change_percent": data.get("10. change percent", "N/A")
        }
    except ConnectionError:
        return {"symbol": symbol, "error": "Failed to connect to the stock API. Please check your internet connection."}
    except Timeout:
        return {"symbol": symbol, "error": "Stock API request timed out. Please try again later."}
    except HTTPError as e:
        if response.status_code == 429:
            return {"symbol": symbol, "error": "Too many requests to the stock API. Please try again later."}
        return {"symbol": symbol, "error": f"Failed to fetch stock data due to HTTP error: {e}"}
    except Exception:
        return {"symbol": symbol, "error": "An unexpected error occurred while fetching stock data. Please try again later."}

@lru_cache(maxsize=100)
def determine_question_type(query):
    """Determine the type of user query using Mistral LLM via CrewAI's task mechanism."""
    prompt = f"""
    Analyze the following user query and determine its category:
    - finance_knowledge: General questions about financial terms, concepts, or strategies
    - market_news: Questions about current market news, trends, or events
    - stock_analysis: Questions about specific stock analysis (e.g., mentioning a stock ticker like AAPL)

    Query: "{query}"

    Provide your response in this format:
    Category: <category>
    Extra Data: <additional info, such as the stock ticker for stock_analysis, or the query itself>
    """

    classifier_agent = Agent(
        role="Query Classifier",
        goal="Classify user queries into appropriate categories.",
        backstory="An expert in natural language understanding, capable of analyzing queries and categorizing them accurately.",
        llm=mistral_llm,
        verbose=True,
        allow_delegation=False
    )

    classifier_task = Task(
        description=prompt,
        agent=classifier_agent,
        expected_output="A classification of the query in the format: Category: <category>\nExtra Data: <additional info>"
    )

    temp_crew = Crew(
        agents=[classifier_agent],
        tasks=[classifier_task],
        process=Process.sequential,
        verbose=False
    )
    
    try:
        response = temp_crew.kickoff()
        response_text = response.raw if hasattr(response, 'raw') else str(response)
        lines = response_text.strip().split("\n")
        if len(lines) < 2:
            raise ValueError("Invalid response format from LLM")
        category_line = lines[0].replace("Category: ", "").strip()
        extra_data_line = lines[1].replace("Extra Data: ", "").strip()
        if category_line not in ["finance_knowledge", "market_news", "stock_analysis"]:
            raise ValueError(f"Invalid category: {category_line}")
        return category_line, extra_data_line
    except Exception:
        return "finance_knowledge", query