from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-3.5-turbo")
loader = CSVLoader(file_path="./laptops.csv")
data = loader.load()
vector = FAISS.from_documents(data, OpenAIEmbeddings())

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert sales assistant at a laptop store. Your job is to recommend a few laptops to the 
    customer based on their requirements.
    If the user is asking something other than buying laptops, let them know that this is a laptop store and purchasing 
    laptops is the only thing that can be done here. 
    Try to fit in the customer to a user persona like student, teacher, gamer, 
    business professional, researcher, teacher, casual everyday user, content creator etc. to recommend laptops. You 
    don't have to stick to the above personas strictly but this is a good guide. It is very important that you do not 
    mention to the user that you're trying to fit them into these personas. It is extremely important to ask user 
    questions to get more information about user's requirements and work pattern if you do not have enough 
    information to make an informed decision.It is absolutely mandatory to use only the below laptop products with 
    their detailed specification as mentioned in the inventory. Context:

<context>
{context}
</context>
         """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retriever.search_kwargs = {'k': 10}

history_aware_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the "
             "conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, history_aware_prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

print("Hi! Welcome to our laptop store, how may I help you?\n")

chat_history = []
while True:
    user_question = input("\nUser: ")
    response = retrieval_chain.invoke({"input": f"{user_question}", "chat_history": chat_history})
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=response["answer"]))
    print("\nSales bot: ", response["answer"])
