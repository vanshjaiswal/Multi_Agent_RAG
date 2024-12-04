import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


class Data_prepration:

    #Setup the Astra DB and configure the access tokens
    def astra_db_connection(self, urls):
        ASTRA_DB_APPLICATION_TOKEN="<Enter the Astra DB Token>" # enter the "AstraCS:..." string found in in your Token JSON file"
        ASTRA_DB_ID="<Enter the DB ID>"
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

    #document to index

        # Load
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits

    def embedding_creation(self, embedding_model_name, doc_splits):
        
        #Creating the huggingface Embeddings
        embeddings=HuggingFaceEmbeddings(model_name=embedding_model_name)
        # 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

        #Storing the embeddings in Astra Vector store
        astra_vector_store=Cassandra(
            embedding=embeddings,
            table_name="qa_mini_demo",
            session=None,
            keyspace=None)


        #adding the documents in vector store
        astra_vector_store.add_documents(doc_splits)
        # print("Inserted %i headlines." % len(doc_splits))
        astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
        retriever=astra_vector_store.as_retriever()

        return
    

        

if __name__=="__main__":
    data_prep = Data_prepration()
    urls = [
        "https://www.investopedia.com/articles/fundamental/03/022603.asp",
    "https://groww.in/p/exchange-traded-funds",
    "https://www.investopedia.com/terms/e/etf.asp",
    "https://www.amfiindia.com/investor-corner/knowledge-center/what-are-mutual-funds-new.html",
    "https://cleartax.in/s/mutual-fund-types",
    "https://www.bajajfinserv.in/futures-and-options",
    "https://www.investopedia.com/ask/answers/difference-between-options-and-futures/",
    "https://www.hdfcbank.com/personal/resources/learning-centre/invest/difference-between-ETF-and-mutual-fund",
    "https://www.schwab.com/etfs/mutual-funds-vs-etfs",
    "https://www.nseindia.com/products-services/indices-index-funds",
    "https://mutualfund.adityabirlacapital.com/blog/what-is-an-index-funds",
    ]
    
    

    doc_splits = data_prep.astra_db_connection(urls)
    embedding_model_name = "all-MiniLM-L6-v2"

    write_flag = True

    if write_flag:
        data_prep.embedding_creation(embedding_model_name=embedding_model_name, doc_splits=doc_splits)
    




    

    

# 












































# urls =["https://www.investopedia.com/terms/c/creditcard.asp",
#    "https://groww.in/credit-card",
#    "https://www.hdfcbank.com/personal/resources/learning-centre/pay/what-is-credit-card-how-do-credit-cards-work",
#    "https://www.investopedia.com/how-do-credit-cards-work-5025119",
#    "https://www.investopedia.com/ask/answers/050415/what-are-differences-between-debit-cards-and-credit-cards.asp",
#    "https://www.hdfcbank.com/personal/resources/learning-centre/pay/debit-card-or-credit-card-which-is-better",
#    "https://www.mastercard.us/en-us/personal/find-a-card/standard-mastercard-credit.html",
#    "https://cardmaven.in/forum/threads/mastercard-world-elite-credit-cards-and-benefits-in-india.38/",
#    "https://www.mastercard.co.in/en-in/personal/find-a-card.html",
#    "https://www.bankbazaar.com/debit-card/mastercard-debit-card.html",
#    "https://www.investopedia.com/articles/markets/032615/how-mastercard-makes-its-money-ma.asp",
   
#    ]
