import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Page configuration
st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 30px;
        }
        .question-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .answer-box {
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
        }
        .source-box {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'> AI Research Paper Assistant</h1>", unsafe_allow_html=True)
st.markdown("""
This assistant answers your questions about **Artificial Intelligence, Machine Learning, and Large Language Models** 
by searching through a collection of research papers and leveraging a local LLM.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    model_name = st.text_input(
        "Model Name",
        value="llama3.2:1b",
        help="Ollama model to use for responses"
    )
    
    num_papers = st.slider(
        "Number of Research Papers to Retrieve",
        min_value=1,
        max_value=20,
        value=5,
        help="How many relevant papers to use for answering"
    )
    
    st.divider()
    st.subheader("About")
    st.write("""
    This project uses:
    - **LangChain** for orchestration
    - **Ollama** for local LLM inference
    - **Chroma** for vector database
    - **Streamlit** for the UI
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = OllamaLLM(model=model_name)

if "retriever_k" not in st.session_state:
    st.session_state.retriever_k = num_papers

# Update model if changed
if model_name != st.session_state.model.model:
    st.session_state.model = OllamaLLM(model=model_name)

# Chat history
st.subheader("Chat History")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")
    st.divider()

# Input section
st.subheader("Ask a Question")
question = st.text_area(
    "Ask about AI, ML, or LLMs:",
    placeholder="E.g., What are the latest advances in transformer models?",
    height=100,
    label_visibility="collapsed"
)

col1, col2 = st.columns([4, 1])

with col2:
    submit_button = st.button("Submit", use_container_width=True, type="primary")

# Process question
if submit_button and question.strip():
    with st.spinner("Searching research papers..."):
        # Retrieve relevant papers
        docs = retriever.invoke(question)
        
        research_papers = "\n\n".join([
            f"Title: {d.metadata.get('Date', 'Unknown Date')}\nContent: {d.page_content}" 
            for d in docs
        ])
    
    # Prepare prompt
    template = """
You are an AI research assistant with deep expertise in Artificial Intelligence, Machine Learning, and Large Language Models (LLMs).

You have access to a collection of research papers that cover a wide range of topics — from deep learning architectures and optimization methods to modern transformer-based models and diffusion models.

Use the context below to answer the user's question clearly and insightfully.

Research Paper Context:
{research_papers}

User's Question:
{question}

Guidelines:
- Write your answer directly and confidently.
- Do not say phrases like "Based on the research papers you've shared" or "According to the dataset".
- Explain as an expert researcher would, focusing on clarity, precision, and depth.
- Use bullet points or short paragraphs for readability.
- If relevant, reference concepts or models (e.g., BERT, GPT, ResNet) but avoid unnecessary repetition.
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | st.session_state.model
    
    # Generate response
    with st.spinner("Generating answer..."):
        response = chain.invoke({
            "research_papers": research_papers,
            "question": question
        })
    
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display the answer
    st.markdown("---")
    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
    st.markdown(f"**Assistant's Answer:**\n\n{response}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show sources
    with st.expander("📚 View Source Papers"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Paper {i}:**")
            st.write(f"**Date:** {doc.metadata.get('Date', 'Unknown')}")
            st.write(f"**Content:** {doc.page_content[:300]}...")
            st.divider()
    
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Build by Shreya Sheta
</div>
""", unsafe_allow_html=True)
