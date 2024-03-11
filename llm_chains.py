import chromadb
import yaml
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma

from prompt_templates import memory_prompt_template

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def create_llm(
    model_path=config["model_path"]["large"],
    model_type=config["model_type"],
    model_config=config["model_config"],
):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm


def create_embeddings(embeddings_path=config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)


def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(
        memory_key="history", chat_memory=chat_history, k=3
    )


def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)


def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)


def create_llm_chain_no_memory(llm, chat_prompt):
    return LLMChain(llm=llm, prompt=chat_prompt)


def load_normal_chain(chat_history):
    return chatChain(chat_history)


def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient("chroma_db")

    langchain_chroma = Chroma(
        client=persistent_client, collection_name="pdfs", embedding_function=embeddings
    )
    return langchain_chroma


class chatChain:
    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        return self.llm_chain.run(
            human_input=user_input,
            history=self.memory.chat_memory.messages,
            stop=["Human:"],
        )

    # def run(self, user_input):
    # print(user_input)
    # return self.llm_chain.invoke(
    #    input={"human_input": user_input, "history": self.chat_history},
    #    stop=["Human:"],
    # )["text"]
