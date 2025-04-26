# agents.py

from utils import gemini_llm
from crewai import Agent

# Agents
finance_knowledge_agent = Agent(
    role="Finance Knowledge Expert",
    goal="Provide accurate, concise, and structured answers to general finance-related questions using provided documents and web data.",
    backstory="An expert with deep knowledge of financial concepts, trained on documents including Basics.pdf, Statementanalysis.pdf, and Financialterms.pdf.",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

market_news_agent = Agent(
    role="Market News Analyst",
    goal="Fetch, summarize, and analyze recent financial news and market trends to provide actionable insights.",
    backstory="A financial journalist with expertise in identifying key market trends and summarizing news for actionable insights.",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

stock_analysis_agent = Agent(
    role="Stock Analysis Expert",
    goal="Provide detailed and actionable analysis of specific stocks, including performance trends and basic technical insights.",
    backstory="A seasoned stock market analyst with expertise in fundamental analysis and basic trend interpretation based on real-time data.",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

response_refiner_agent = Agent(
    role="Response Refiner and Reporter",
    goal="Simplify, verify, and format responses from other agents into a concise, professional report for the user.",
    backstory="A meticulous editor with a background in finance, specializing in simplifying complex information and presenting it in a clear, professional report format.",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)