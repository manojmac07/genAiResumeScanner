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

index_name: str = "resumeindexnew"

vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint='https://resumescan.search.windows.net',
        azure_search_key='',
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        # Configure max retries for the Azure client
        additional_search_client_options={"retry_total": 4},
        )

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = vector_store.as_retriever()
# Create a ChatOpenAI model
llm = AzureChatOpenAI(model="gpt-4o",temperature=0,max_retries=4)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question.Find the unique names and unique skills"
    "If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
#user input "data" "exp"
#templates 

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
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

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Function to simulate a continual chat
def continual_chat():
    
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            
        # Display the AI's response
        print(f"AI: {result['answer']}")
    

        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))
    

# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()