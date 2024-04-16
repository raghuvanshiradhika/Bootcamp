import os

from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.agents import load_tools, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langfuse.client import Langfuse


class OpenAIChat:
    def __init__(self):
        # Langfuse
        self.langfuse = Langfuse(public_key="pk-lf-f00c4ebc-b458-4169-a83a-ad9644cbf0b1",
                                 secret_key="sk-lf-15acce6c-1fb6-4cc0-9b62-923b72f63be8",
                                 host="http://dev.monitoring.cybage.com",
                                 release="1.0.0",
                                 debug=True)

        self.trace = self.langfuse.trace(name="openai_function_agent", user_id="truptija@cybage.com")
        self.handler = self.trace.get_langchain_handler()
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0,
            callbacks=[self.handler],
        )
        self.vector_store = self.load_document()

    def load_document(self):
        loader = Docx2txtLoader('Product_Catalogue.docx')
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)
        print(len(all_splits))
        # Store splits
        return Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    def submit_prompt(self, question) -> str:
        from langchain_community.utilities import GoogleSerperAPIWrapper
        search = GoogleSerperAPIWrapper()

        tools = load_tools(["search", "document_search"])
        from langchain.agents import create_tool_calling_agent

        agent = create_tool_calling_agent(self.llm, tools, question)

        from langchain.agents import AgentExecutor

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        result = agent_executor.invoke({"input": question})
        # print(question)

        return result

    def Lang_fuse(self):
        pass
        # result = self.document_search(question)
        # self.trace.update(input={"query": "Give the output for {query_3}".format(query_3=question)},
        # output={"result": result})
        # self.handler.get_trace_url()
        # self.langfuse.flush()

    def document_search(self, question):
        from langchain.chains import RetrievalQA
        qa_chain = RetrievalQA.from_llm(llm=self.llm, retriever=self.vector_store.as_retriever())
        result = qa_chain.run(question)
        return result
