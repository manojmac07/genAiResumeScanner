import os
import openai
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch



os.environ["AZURE_OPENAI_ENDPOINT"] = "https://csscan.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_ENDPOINT"] = "https://csscan.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"

openai.api_type='azure'
openai.api_key=''
openai.api_version='2024-05-01-preview'


# Define the embedding model
embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")

index_name: str = "skmatchtraining"

vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint='https://resumescan.search.windows.net',
        azure_search_key='',
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        # Configure max retries for the Azure client
        additional_search_client_options={"retry_total": 4},
        )


retriever = vector_store.as_retriever()
llm = AzureChatOpenAI(model="gpt-4o", temperature=0, max_retries=4)

contextualize_q_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question.Find the unique names and unique skills"
    "If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "Now you are the project manager and you have the responsibility to build a team from the list of resumes given to you. "
    "Project Description: To build a trading application which can help users trade actively on Equity, futures. Key pointers on the project are as below:"
    "A strong Ui is a must which focus on user experience is vital. "
    "Customer journey and exhaustive list of use cases have to be derived for the development"
    "A strong data flow architecture and data storage is important to navigate through the application"
    "A business person who can guide on the functionality would be needed at all stages of development. "
    "Project is to be completed in 3 months time and is fixed bid project with no cap on number of people in the project. The project is an agile project with periodic reviews with the client."
    "You are the resourcing assistant for a project to build a retail ecommerce platform for selling Apparels."
    "The project has to be built from scratch and it is an agile IT project. "
    "From the list of resources, help identify the below resources to a build a dynamic team."
    "Task: Use your best judgement to build an agile team from the resumes available along with the selection criteria used to pick the resources. "
    "Restrictions: Don’t pick the same resume for two roles and don’t use any resumes outside the folder provided to you."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def process_user_query(query, chat_history):
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return result["answer"]