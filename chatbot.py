from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are {persona} and tutor. Maintain a conversational tone and engage with the user naturally within 20 words."),
    ("human", "Previous conversation: {history}\nCurrent question: {question}")
])

logging.info("prompt ready")

# Streamlit UI
st.title('Interactive Chatbot')

# Sidebar for persona selection
with st.sidebar:
    persona = st.text_input("Who would you like to talk to?", 
                           placeholder="e.g., Einstein, Shakespeare, etc.")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if input_text := st.chat_input("What's your question?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(input_text)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input_text})

    # Prepare conversation history
    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]])

    # Initialize LLM and chain
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        convert_system_message_to_human=True,
        temperature=0.7
    )

    llm2 = ChatGroq(
        temperature=0.7,
        model_name="llama3-8b-8192",
        max_tokens=1024
    )

    chain = prompt | llm2 | StrOutputParser()

    try:
        # Generate response
        response = chain.invoke({
            'question': input_text,
            'persona': persona if persona else "a helpful assistant",
            'history': history
        })

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

logging.info("Chat session active")