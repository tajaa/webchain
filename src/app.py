import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


def get_vectorstore_from_url(url):
    # get text in document forms
    loader = WebBaseLoader(url)
    document = loader.load()

    # split doucment into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, genarate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


# this function is creating a dox chain to take context and answer
def get_conversation_rag_chain(retriever_chain):
    """
    this function is creating a documents chain taht takes context and answers questions based on documents --that we pass the context, then its putting it together with the retreiver chain to create our final chain that will take our user_query, chat_history and it will return to us an answer based the etnire convo and the context the chain has found
    """
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    return "i don't know"


# app.config

st.set_page_config(page_title="Chat with websites", page_icon="🤖")

st.title("Chat with websites")

# if "chat_history" not in st.session_state:
#    st.session_state.chat_history = [
#        AIMessage(content="Hello, I'm a bot. How can I help you?")
#    ]

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("please insert a website url")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I'm a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

        # create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    conversation_rag_chain = get_conversation_rag_chain(retriever_chain)

    # user input
    user_query = st.chat_input("type message here...")
    if user_query is not None and user_query != "":

        # response = get_response(user_query)
        response = conversation_rag_chain.invoke(
            {"chat_history": st.session_state.chat_history, "input": user_query}
        )
        st.write(response)
        # st.session_state.chat_history.append(HumanMessage(content=user_query))
        # st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:

        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
