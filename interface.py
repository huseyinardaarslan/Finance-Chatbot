# interface.py

import gradio as gr
from crewai import Crew, Process
from agents import finance_knowledge_agent, market_news_agent, stock_analysis_agent, response_refiner_agent
from tasks import get_finance_knowledge_task, get_market_news_task, get_stock_analysis_task, get_response_refiner_task
from utils import determine_question_type, search_qdrant

def get_response(query):
    """Get chatbot response."""
    finance_crew = Crew(
        agents=[finance_knowledge_agent, market_news_agent, stock_analysis_agent, response_refiner_agent],
        tasks=[],
        process=Process.sequential,
        verbose=True
    )

    try:
        question_type, processed_query = determine_question_type(query)
        finance_crew.tasks = []

        # Check RAG usage
        rag_insufficient = False
        rag_used = False
        context_text = ""
        
        if question_type == "finance_knowledge":
            contexts = search_qdrant(query, top_k=2)
            
            if contexts and len(contexts) > 0:
                shortened_contexts = []
                for ctx in contexts:
                    text = ctx["text"]
                    if len(text) > 150:
                        text = text[:147] + "..."
                    shortened_contexts.append({
                        "source": ctx["source"],
                        "text": text
                    })
                
                context_text = "\n\n".join([f"Source: {ctx['source']}\nContent: {ctx['text']}" for ctx in shortened_contexts])
                is_context_useful = len(context_text) > 30 and any(keyword in context_text.lower() for keyword in query.lower().split())
                
                rag_insufficient = not is_context_useful
                rag_used = True
            else:
                rag_insufficient = True
                rag_used = False
            
            initial_task = get_finance_knowledge_task(query)
        elif question_type == "market_news":
            initial_task = get_market_news_task(query)
        elif question_type == "stock_analysis":
            initial_task = get_stock_analysis_task(processed_query)
        else:
            initial_task = get_finance_knowledge_task(query)

        finance_crew.tasks.append(initial_task)
        initial_response = finance_crew.kickoff()

        # Pass RAG status to the refiner
        if question_type == "finance_knowledge":
            if not rag_used:
                note = "RAG_NOT_USED"
            elif rag_insufficient:
                note = "RAG_LIMITED"
            else:
                note = "RAG_SUFFICIENT"
        else:
            note = "NO_RAG_NEEDED"
            
        refiner_task = get_response_refiner_task(query, initial_response, question_type, rag_note=note)
        finance_crew.tasks = [refiner_task]
        final_report = finance_crew.kickoff()

        return final_report
    except Exception as e:
        return f"Error: {e}\nPlease try again."

# CSS
custom_css = """
    /* Base variables */
    :root {
        --primary-color: #00416a;
        --secondary-color: #047fb7;
        --title-color: #FFD700;
        --text-color: #ffffff;
        --bg-dark: #1a1a1a;
        --bg-medium: #2a2a2a;
        --bg-light: #353535;
        --input-bg: #ffffff;
        --input-text: #333333;
        --button-green: #4CAF50;
    }
    
    /* Reset and base styles */
    body, html {
        background-color: #1a1a1a !important;
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow-x: hidden;
    }
    
    /* Container styles */
    .gradio-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #00416a 100%) !important;
        font-family: 'Montserrat', 'Arial', sans-serif !important;
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 25px !important;
        min-height: 100vh !important;
    }
    
    /* WHITE BACKGROUND FOR INPUT/OUTPUT AREAS */
    .gradio-container .input-box,
    .gradio-container .output-box,
    .gr-form,
    .gr-box,
    .gr-padded,
    .gr-panel,
    .gr-input,
    .gr-input-label,
    textarea,
    .gr-textarea,
    .gr-textbox {
        background-color: #ffffff !important;
    }
    
    /* Make sure text in textboxes is black on white */
    textarea,
    .gr-textarea textarea,
    .gr-textbox textarea,
    .gr-textbox input {
        background-color: #ffffff !important;
        color: #333333 !important;
    }
    
    /* Submit button with WHITE TEXT */
    .gr-button.submit-button {
        background-color: #4CAF50 !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        text-align: center !important;
        width: 150px !important;
        margin: 10px auto !important;
        display: block !important;
    }
    
    /* This is critical for the button text */
    button,
    .gr-button span,
    .submit-button span {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* EXAMPLES HEADER IN WHITE */
    .examples-title,
    .gr-examples .gr-interface-title,
    footer div,
    footer span {
        color: #ffffff !important;
        background-color: transparent !important;
        font-weight: bold !important;
    }
    
    /* Input/output boxes styling */
    .gr-box {
        border-radius: 8px !important;
        padding: 12px !important;
        background-color: #ffffff !important;
        border: 1px solid #047fb7 !important;
    }
    
    /* Make labels text white */
    label, .gr-block.gr-box label {
        color: #ffffff !important;
        font-weight: bold !important;
        background-color: transparent !important;
    }
    
    /* Example buttons styling */
    .gr-examples .gr-sample-btn {
        background-color: #e0e0e0 !important;
        border: 1px solid #047fb7 !important;
        color: #333333 !important;
        border-radius: 8px !important;
    }
    
    /* Title and subtitle */
    h1 {
        color: #FFD700 !important;
        text-align: center !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
        font-weight: bold !important;
    }
    
    h3 {
        color: #ffffff !important;
        text-align: center !important;
    }
    
    /* Finance icons and ticker */
    .finance-icons {
        text-align: center !important;
        color: #ffffff !important;
    }
    
    .ticker-tape {
        background-color: #353535 !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        padding: 8px 0 !important;
        margin: 15px 0 !important;
        border-radius: 5px !important;
    }
    
    .ticker-content {
        display: inline-block !important;
        animation: ticker 30s linear infinite !important;
        color: #ffffff !important;
    }
    
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    
    /* Stock symbols */
    .stock-symbol {
        margin: 0 15px !important;
        display: inline-block !important;
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    .up { color: #4CAF50 !important; }
    .down { color: #FF5252 !important; }
    
    /* Ensure white background for example areas */
    .examples {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    /* Remove unwanted elements */
    .footer-container, .footer, .gr-footnote {
        display: none !important;
    }
    
    /* Fix any specific issues with background overrides */
    .gradio-container .prose,
    .gradio-container .prose p {
        background-color: transparent !important;
    }
"""

# HTML components for design
ticker_html = """
<div class="ticker-tape">
    <div class="ticker-content">
        <span class="stock-symbol">AAPL <span class="up">+1.2%</span></span>
        <span class="stock-symbol">MSFT <span class="up">+0.8%</span></span>
        <span class="stock-symbol">GOOGL <span class="down">-0.4%</span></span>
        <span class="stock-symbol">AMZN <span class="up">+2.1%</span></span>
        <span class="stock-symbol">TSLA <span class="down">-1.7%</span></span>
        <span class="stock-symbol">JPM <span class="up">+0.5%</span></span>
        <span class="stock-symbol">BAC <span class="down">-0.2%</span></span>
        <span class="stock-symbol">WMT <span class="up">+0.3%</span></span>
    </div>
</div>
"""

finance_icons = """
<div class="finance-icons">
    <span style="color: #4CAF50;">📊 📈 💹 💰 📉 🏦 📃 /span>
</div>
"""

# Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Base(), analytics_enabled=False) as interface:
    gr.HTML("<h1>Professional Finance Assistant</h1>")
    gr.HTML("<h3>Your AI-powered financial advisor and market analyst</h3>")
    
    gr.HTML(finance_icons)
    gr.HTML(ticker_html)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=4,
                placeholder="Ask me anything about finance (e.g., 'What is dividend investing?', 'Analyze Tesla stock', 'Latest market news on tech sector')",
                label="Your Financial Query",
                show_label=True,
                interactive=True
            )
            submit_btn = gr.Button("Submit", elem_classes=["submit-button"])
    
    output_text = gr.Textbox(label="Financial Analysis", lines=10)
    
    example_queries = [
        ["What is the difference between bull and bear markets?"],
        ["Analyze AAPL stock performance"],
        ["Latest news about cryptocurrency market"],
        ["Explain P/E ratio and its importance"]
    ]
    gr.Examples(example_queries, inputs=input_text, examples_per_page=5, label="Examples")
    
    # Event handlers
    submit_btn.click(fn=get_response, inputs=input_text, outputs=output_text)

# Launch the interface
interface.launch(share=False, inbrowser=True)