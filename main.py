# main.py

from crewai import Crew, Process
from agents import finance_knowledge_agent, market_news_agent, stock_analysis_agent, response_refiner_agent
from tasks import get_finance_knowledge_task, get_market_news_task, get_stock_analysis_task, get_response_refiner_task
from utils import determine_question_type, search_qdrant

def main():
    """Main function to run the finance chatbot in terminal."""
    finance_crew = Crew(
        agents=[finance_knowledge_agent, market_news_agent, stock_analysis_agent, response_refiner_agent],
        tasks=[],
        process=Process.sequential,
        verbose=True
    )

    print("📈 Welcome to the Finance Chatbot!")
    print("Examples: 'What is investing?', 'Analyze AAPL', 'What’s the latest market news?'")
    while True:
        query = input("Enter your query (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        try:
            question_type, processed_query = determine_question_type(query)
            finance_crew.tasks = []  # Reset tasks for each query

            rag_note = "RAG_SUFFICIENT"  # Default value
            if question_type == "finance_knowledge":
                contexts = search_qdrant(query, top_k=3)
                context_text = "\n\n".join([f"Source: {ctx['source']}\nContent: {ctx['text']}" for ctx in contexts])
                is_context_useful = len(context_text) > 50 and any(query.lower() in ctx["text"].lower() for ctx in contexts)
                rag_note = "RAG_NOT_USED" if not is_context_useful else "RAG_SUFFICIENT"
                initial_task = get_finance_knowledge_task(query)
            elif question_type == "market_news":
                rag_note = "NO_RAG_NEEDED"  # Market news doesn't use RAG
                initial_task = get_market_news_task(query)
            elif question_type == "stock_analysis":
                rag_note = "NO_RAG_NEEDED"  # Stock analysis doesn't use RAG
                initial_task = get_stock_analysis_task(processed_query)
            else:
                initial_task = get_finance_knowledge_task(query)  # Default

            finance_crew.tasks.append(initial_task)
            initial_response = finance_crew.kickoff()

            refiner_task = get_response_refiner_task(query, initial_response, question_type, rag_note=rag_note)
            finance_crew.tasks = [refiner_task]
            final_report = finance_crew.kickoff()

            print(f"\nFinal Report:\n{final_report}\n")
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try again with a different query.\n")

if __name__ == "__main__":
    main()