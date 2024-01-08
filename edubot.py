from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from config import *

class EduBotCreator:

    def __init__(self):
        self.prompt_temp = PROMPT_TEMPLATE
        self.input_variables = INP_VARS
        self.chain_type = CHAIN_TYPE
        self.search_kwargs = SEARCH_KWARGS
        self.embedder = EMBEDDER
        self.vector_db_path = VECTOR_DB_PATH
        self.model_ckpt = MODEL_CKPT
        self.model_type = MODEL_TYPE
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE

    def create_custom_prompt(self):
        """
        The function `create_custom_prompt` creates a custom prompt template using a given template and
        input variables.
        :return: a custom prompt template object.
        """
        custom_prompt_temp = PromptTemplate(template=self.prompt_temp,
                            input_variables=self.input_variables)
        return custom_prompt_temp
    
    def load_llm(self):
        """
        The function `load_llm` creates and returns an instance of the `CTransformers` class with specified
        parameters.
        :return: an instance of the CTransformers class.
        """
        llm = CTransformers(
                model = self.model_ckpt,
                model_type=self.model_type,
                max_new_tokens = self.max_new_tokens,
                temperature = self.temperature
            )
        return llm
    
    def load_vectordb(self):
        """
        The function `load_vectordb` loads a vector database using the HuggingFaceEmbeddings model and FAISS
        library.
        :return: the loaded vector database.
        """
        hfembeddings = HuggingFaceEmbeddings(
                            model_name=self.embedder, 
                            model_kwargs={'device': 'cpu'}
                        )

        vector_db = FAISS.load_local(self.vector_db_path, hfembeddings)
        return vector_db

    def create_bot(self, custom_prompt, vectordb, llm):
        """
        The `create_bot` function creates a retrieval-based question answering bot using a language model, a
        vector database, and a custom prompt.
        
        :param custom_prompt: The custom_prompt parameter is a string that represents the prompt or question
        that the bot will use to generate responses. It is the initial input that the bot will receive from
        the user
        :param vectordb: The `vectordb` parameter is an instance of a vector database. It is used as a
        retriever in the retrieval-based question answering system. The `as_retriever` method is called on
        the `vectordb` object to convert it into a retriever
        :param llm: The "llm" parameter is an instance of a language model. It is used in the retrieval
        question answering chain to generate responses based on the retrieved documents
        :return: The function `create_bot` returns a retrieval-based question answering (QA) chain.
        """
        retrieval_qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type=self.chain_type,
                                retriever=vectordb.as_retriever(search_kwargs=self.search_kwargs),
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": custom_prompt}
                            )
        return retrieval_qa_chain
    
    def create_edubot(self):
        """
        The function creates an "edubot" by initializing various components and returning the created bot.
        :return: the "bot" object.
        """
        self.custom_prompt = self.create_custom_prompt()
        self.vector_db = self.load_vectordb()
        self.llm = self.load_llm()
        self.bot = self.create_bot(self.custom_prompt, self.vector_db, self.llm)
        return self.bot