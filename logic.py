from dotenv import load_dotenv
load_dotenv()
import base64
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from parser import parse, encodeImage  
vision_llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def get_image_summaries(image_paths: list):
    summaries = []

    
    prompt_base = """
You are an expert at analyzing images extracted from documents.
Describe this image in detail.
- If it's a chart or graph: Extract the title, axis labels, all data points, and describe the key trends.
- If it's a table: Extract all headers, rows, and columns as text, preserving the structure.
- If it's a diagram: Describe what it shows and the relationships between parts.
Provide the output as plain text (no JSON required).
"""

    for img_path in image_paths:
        print(f"Summarizing image: {img_path}")
        try:
            base64_image = encodeImage(img_path)
        except Exception as e:
            print(f"Warning: could not read image {img_path}: {e}")
            summaries.append(f"[Failed to read image: {img_path}]")
            continue

        
        message_content = [
            {
                "type": "text",
                "text": prompt_base  
            },
            {
                "type": "image_url",
                
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        ]

        
        try:
            msg = vision_llm.invoke([HumanMessage(content=message_content)])
           
            if hasattr(msg, "content"):
                content = msg.content
            else:
                content = str(msg)
            
            if not isinstance(content, str):
                content = str(content)
            summaries.append(content)
        except Exception as e:
            print(f"Error while summarizing image {img_path}: {e}")
            summaries.append(f"[Error summarizing image {os.path.basename(img_path)}: {e}]")

    return summaries


def create_vector_store(text_chunks: list, image_summaries: list, persist_directory: str = "./chroma_data"):
    print("Creating vector store...")

    
    vector_store = Chroma(
        collection_name="visual_doc_store",
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    combined_text = "\n\n".join(text_chunks) if text_chunks else ""
    split_texts = text_splitter.split_text(combined_text) if combined_text else []

    if split_texts:
        try:
            vector_store.add_texts(split_texts)
        except Exception as e:
            print(f"Warning: failed to add text chunks to vector store: {e}")

    
    if image_summaries:
        summary_metadatas = [{"source_type": "image_summary"} for _ in image_summaries]
        try:
            vector_store.add_texts(texts=image_summaries, metadatas=summary_metadatas)
        except Exception as e:
            print(f"Warning: failed to add image summaries to vector store: {e}")

    
    try:
        vector_store.persist()
    except Exception:
        
        pass

    print("Vector store created successfully.")
    return vector_store


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    template = """
Answer the user's question based ONLY on the following context, which may include text and descriptions of charts or tables.
If the answer isn't in the context, say so.

Context:
{context}

Question:
{question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


def get_answer_from_pdf(pdf_path: str, question: str):
    """
    This is the main function that ties everything together.
    It will be called by our MCP server.
    """

    
    text_chunks, image_paths = parse(pdf_path)
    image_summaries = get_image_summaries(image_paths)
    vector_store = create_vector_store(text_chunks, image_summaries)
    qa_chain = create_qa_chain(vector_store)
    try:
        answer = qa_chain.invoke(question)
    except Exception as e:
        print(f"Error during QA chain invocation: {e}")
        answer = f"An error occurred while answering the question: {e}"
    for img_path in image_paths:
        try:
            os.remove(img_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: could not remove {img_path}: {e}")
    try:
        if image_paths:
            parent_dir = os.path.dirname(image_paths[0])
            if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
    except Exception:
        pass

    return answer
