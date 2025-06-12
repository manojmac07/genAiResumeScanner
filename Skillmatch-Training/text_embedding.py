import os
import openai
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import hashlib



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

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))


index_name: str = "skmatchtrainingv1"
vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint='https://resumescan.search.windows.net',
        azure_search_key='',
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        # Configure max retries for the Azure client
        additional_search_client_options={"retry_total": 4},
        )

search_client = SearchClient(endpoint='https://resumescan.search.windows.net', index_name=index_name, credential=AzureKeyCredential(''))





def generate_checksum(file_content):
    # Use SHA-256 or another hash function to compute file content checksum
    checksum = hashlib.sha256(file_content).hexdigest()
    return checksum

def generate_file_checksum(file_path):
    """Generate a checksum for the entire file content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buffer = f.read()
        hasher.update(buffer)
    return hasher.hexdigest()

def file_exists(checksum):
    # Query the index for existing content with the same checksum
    results = search_client.search(search_text="", filter=f"checksum eq '{checksum}'")
    return len(list(results)) > 0


def store_or_update_file(file_content, person_name, allocated, start_date, end_date):
    checksum = generate_checksum(file_content)

    # Info to upload
    vector_embeddings = your_language_model_function(file_content)  # Replace with your actual function for generating embeddings
    document = {
        "id": checksum,  # Use checksum as a unique identifier
        "file_content": file_content,
        "checksum": checksum,
        "vector_embeddings": vector_embeddings,
        "person_name": person_name,
        "allocated": allocated,
        "start_date": start_date,
        "end_date": end_date,
    }

    if not file_exists(checksum):
        # Data is new, so upload
        search_client.upload_documents(documents=[document])
        print("New file content uploaded successfully.")
    else:
        print("File content already exists in Azure Search.")

def vector_store_all():
    all_folder_path = os.path.join(current_dir, "docs")
    documents = []
    
    # Rename files to remove spaces from file names
    for filename in os.listdir(all_folder_path):
        if ' ' in filename:  # Check if there's a space in the file name
            new_filename = filename.replace(' ', '_')  # Replace spaces with underscores
            os.rename(os.path.join(all_folder_path, filename), os.path.join(all_folder_path, new_filename))
    
    # Load documents based on their file extensions
    for file in os.listdir(all_folder_path):
        print(file)
        file_path = os.path.join(all_folder_path, file)
        
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            continue  # Skip unsupported file types
        
        documents.extend(loader.load())  # Extend the documents list

    

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    

    vector_store.add_documents(documents=docs)

def store_file_vector(file_content, vector_embeddings, file_checksum, storage_client):
    vector_embeddings = your_language_model_function(file_content)  # Replace with your actual function for generating embeddings
    document = {
        "id": file_checksum,  # Use checksum as a unique identifier
        "file_content": file_content,
        "checksum": file_checksum,
        "vector_embeddings": vector_embeddings,
        "person_name": person_name,
        "allocated": 'allocated',
        "start_date": start_date,
        "end_date": end_date,
    }
def generate_vector_embeddings(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported file type")
    
    documents = loader.load()
    full_text = " ".join(doc['text'] for doc in documents)  # Assuming each `doc` has a `text` attribute
    return full_text


def process_single_file_upload(file, storage_client):
    """Process a single file from the frontend attachment or form upload."""
    file_path = file.save()  # Save the file in a temporary location or directly access its contents
    file_checksum = generate_file_checksum(file_path)
    
    if not file_exists(file_checksum, storage_client):
        # Load, vectorize, and store if it is a new file
        vector_embeddings = generate_vector_embeddings(file_path)
        store_file_vector(file_path, vector_embeddings, file_checksum, storage_client)
        print(f"Stored new file: {file_path}")
    else:
        print(f"File {file_path} already exists in storage.")
        
def process_bulk_uploads(files, storage_client):
    """Process multiple files from a bulk upload."""
    for file in files:
        process_single_file_upload(file, storage_client)
        
def handle_frontend_upload(request, storage_client):
    """Handle file uploads from the frontend."""
    if 'files' in request.files:  # Assuming Flask-like request object with file uploads
        # Bulk upload
        files = request.files.getlist('files')
        #process_bulk_uploads(files, storage_client)
        vector_store_all()
    elif 'file' in request.files:
        # Single attachment upload
        file = request.files['file']
        process_single_file_upload(file, storage_client)
    
    
# Main function to start the process
#if __name__ == "__main__":
#    vector_store_all()
    