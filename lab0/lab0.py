from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()
documents = [Document(page_content=text)]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0
)
print("RAG система запущена")
print("Введи 'exit' для виходу\n")

while True:

    query = input("Запит: ")

    if query.lower() == "exit":
        print("Завершення роботи")
        break
    docs_found = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs_found])
    prompt = f"""
Ти AI-помічник факультету.

Відповідай ТІЛЬКИ використовуючи інформацію з контексту.

Якщо відповіді немає у контексті — напиши:
"У наданому тексті немає інформації про це."

Контекст:
{context}

Питання:
{query}
"""
    response = llm.invoke(prompt)

    print("\nВідповідь:")
    print(response.content)

    print("\n" + "-" * 50 + "\n")