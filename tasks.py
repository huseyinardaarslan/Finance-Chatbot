# tasks.py

from utils import search_qdrant, search_news, get_stock_data
from crewai import Task
from agents import finance_knowledge_agent, market_news_agent, stock_analysis_agent, response_refiner_agent

def get_finance_knowledge_task(query):
    """Task for answering general finance knowledge questions."""
    contexts = search_qdrant(query, top_k=3)
    context_text = "\n\n".join([f"Source: {ctx['source']}\nContent: {ctx['text']}" for ctx in contexts])
    is_context_useful = len(context_text) > 50 and any(query.lower() in ctx["text"].lower() for ctx in contexts)

    web_results = search_news(query, max_results=3)
    web_text = "\n\n".join([f"Title: {item['title']}\nSummary: {item['snippet']}" for item in web_results]) if web_results else "No additional info from the web."

    if is_context_useful:
        prompt = f"""
        User query: '{query}'

        You are a Finance Knowledge Expert. Use the following RAG data and web search results to provide a concise, accurate response:

        RAG Data:
        {context_text}

        Web Search Results:
        {web_text}

        ### Instructions:
        - Provide a clear definition or explanation related to the query.
        - Include a practical example or implication if relevant.
        - Cite your sources (e.g., "According to Basics.pdf" or "Based on web search").
        - Keep the response concise, under 200 words.
        """
    else:
        prompt = f"""
        User query: '{query}'

        The RAG data from Qdrant was insufficient:
        {context_text}

        Web search results:
        {web_text}

        ### Instructions:
        - Rely on your own knowledge and web results to provide a concise, accurate response.
        - Note that RAG data was insufficient.
        - Include a practical example or implication if relevant.
        - Cite your sources (e.g., "Based on web search").
        - Keep the response concise, under 200 words.
        """
    return Task(
        description=prompt,
        agent=finance_knowledge_agent,
        expected_output="A concise explanation of the financial concept, with an example and cited sources, under 200 words."
    )

def get_market_news_task(query):
    """Task for summarizing and analyzing market news."""
    news = search_news(query, max_results=3)
    news_text = "\n\n".join([f"Title: {item['title']}\nSummary: {item['snippet']}" for item in news]) if news else "No recent news found."

    prompt = f"""
    User query: '{query}'

    You are a Market News Analyst. Analyze the following news data and provide a summary with actionable insights:

    News Data:
    {news_text}

    ### Instructions:
    - Summarize the key points from the news in 3-4 sentences.
    - Highlight any trends or events that could impact the market.
    - Provide one actionable insight or recommendation for investors.
    - Cite the news sources (e.g., "According to [title]").
    - Keep the response concise, under 200 words.
    """
    return Task(
        description=prompt,
        agent=market_news_agent,
        expected_output="A concise summary of market news, highlighting trends, with an actionable insight, under 200 words."
    )

def get_stock_analysis_task(symbol):
    """Task for analyzing a specific stock with basic technical insights."""
    stock_data = get_stock_data(symbol)
    if "error" in stock_data:
        prompt = f"""
        User query: 'Analyze {symbol}'

        You are a Stock Analysis Expert. There was an error fetching data for the stock:

        Error: {stock_data['error']}

        ### Instructions:
        - Provide a general overview of the stock based on your knowledge.
        - Suggest a potential reason for the error.
        - Recommend an action for the user.
        - Keep the response concise, under 200 words.
        """
    else:
        data_text = f"Price: {stock_data['price']}\nChange: {stock_data['change']} ({stock_data['change_percent']})"
        prompt = f"""
        User query: 'Analyze {symbol}'

        You are a Stock Analysis Expert. Analyze the following stock data with basic technical insights:

        Stock Data:
        {data_text}

        ### Instructions:
        - Interpret the stock's performance and identify any price trend (e.g., upward/downward movement).
        - Identify potential factors influencing the stock (e.g., market trends, sector performance).
        - Provide an investment recommendation (e.g., "Hold", "Buy", "Sell") with a brief rationale.
        - Keep the response concise, under 200 words.
        """
    return Task(
        description=prompt,
        agent=stock_analysis_agent,
        expected_output="A concise analysis of the stock's performance with an investment recommendation, under 150 words."
    )

def get_response_refiner_task(query, initial_response, question_type, rag_note="NO_RAG_NEEDED"):
    """Task for refining and reporting the response."""
    
    # Create a special note for RAG information
    rag_message = ""
    if rag_note == "RAG_NOT_USED":
        rag_message = "Note: No relevant information found in RAG system, web search results were used."
    elif rag_note == "RAG_LIMITED":
        # Don't show any special message when RAG and web search are used together
        rag_message = ""
    elif rag_note == "RAG_SUFFICIENT":
        # Don't show any special message when RAG is sufficient
        rag_message = ""
    
    # Handle out_of_scope case
    if question_type == "out_of_scope":
        prompt = f"""
        User query: '{query}'
        Initial Response: '{initial_response}'
        Question Type: '{question_type}'

        You are a Response Refiner and Reporter. The query is out of scope for a finance assistant.

        ### Instructions:
        - Format as a report indicating the query is out of scope.
        - Do not perform any research or generate additional content.
        - Format as a report:
          **Financial Report for Query: '{query}'**
          - **Summary**: [Explain that the query is not finance-related]
          - **Key Insight**: [Guide the user to ask finance-related questions]
          - **Source/Note**: [Add note indicating no further processing]
        - Keep under 200 words.
        """
        expected_output = f"""
        **Financial Report for Query: '{query}'**
        - **Summary**: This query is not related to finance. I am designed to assist with financial topics like stock analysis, market news, or financial concepts.
        - **Key Insight**: Please try a finance-related question, such as “Analyze META stock performance” or “What is revenue?”
        - **Source/Note**: No further processing performed as query is out of scope.
        """
    else:
        prompt = f"""
        User query: '{query}'
        Initial Response: '{initial_response}'
        Question Type: '{question_type}'
        
        {rag_message}
        
        You are a Response Refiner and Reporter. Refine the initial response and present it in a professional report format.

        ### Instructions:
        - Simplify the language for a general audience.
        - Verify accuracy and logical alignment with the query; add a note if issues are found.
        - Format as a report:
          **Financial Report for Query: '{query}'**
          - **Summary**: [Simplified summary in 3-4 sentences]
          - **Key Insight**: [One key takeaway or recommendation]
          - **Source/Note**: [Cite source or add note]
        - Keep under 200 words.
        """
        expected_output = "A simplified and professionally formatted report, under 200 words."
    
    return Task(
        description=prompt,
        agent=response_refiner_agent,
        expected_output=expected_output
    )