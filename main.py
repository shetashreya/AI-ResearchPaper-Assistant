from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2:1b")

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
- Do not say phrases like “Based on the research papers you've shared” or “According to the dataset”.
- Explain as an expert researcher would, focusing on clarity, precision, and depth.
- Use bullet points or short paragraphs for readability.
- If relevant, reference concepts or models (e.g., BERT, GPT, ResNet) but avoid unnecessary repetition.
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------------------------------")
    question = input("Ask a question about AI, ML, or LLMs (q to quit): ")
    print("\n\n")
    if question.lower() == 'q':
        break

    docs = retriever.invoke(question)

    research_papers = "\n\n".join([
        f"Title: {d.metadata.get('Date', '')}\nContent: {d.page_content}" for d in docs
    ])
    result = chain.invoke({
        "research_papers": research_papers,
        "question": question
    })

    print("\n", result, "\n")
