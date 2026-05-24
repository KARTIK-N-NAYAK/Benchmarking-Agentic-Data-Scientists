import warnings

from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agent.memory import MemKey
from agent.tools.base_tool import Tool

warnings.filterwarnings("ignore")


class RAG(Tool):
    requires_llm_prompt: bool = True

    def __init__(self, doc: str | None, doctype: str | None, use_rag: bool = True, tag=MemKey.RAG_RETRIEVAL):
        """
        Args:
            use_rag: whether to use RAG (True by default)
        """
        self.doc = doc
        self.doctype = doctype
        self.use_rag = use_rag
        self.tag = tag

    def __call__(self, agent_input: str) -> dict[str, str]:
        if not self.use_rag:
            return {}
        print("AGENT_INPUT", agent_input)
        rag = self.get_context(agent_input=agent_input)
        return {self.tag: rag}

    def doc_loader(self):
        if self.doctype not in ["pdf", "files", "html", "csv", "python_files"]:
            raise ValueError("Please enter a valid doctype from- 'pdf', 'files', 'html', 'csv', 'python_files'.")
        elif self.doctype == "pdf":
            loader = PyPDFLoader(self.doc)
        elif self.doctype == "html":
            loader = UnstructuredHTMLLoader(self.doc)
        elif self.doctype == "csv":
            loader = CSVLoader(file_path=self.doc)
        elif self.doctype == "files":
            try:
                loader = DirectoryLoader(self.data, glob="**/*")
            except Exception:
                print("The file types in your directory may not be supported.")
                raise
        elif self.doctype == "python_files":
            loader = GenericLoader.from_filesystem(
                self.doc,
                glob="**/*",
                suffixes=[".py"],
                parser=LanguageParser(language=Language.PYTHON.value, parser_threshold=500),
            )
        else:
            raise ValueError(self.doctype)

        doc = loader.load()

        if self.doctype == "python_files":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
            )
            texts = splitter.split_documents(doc)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, is_separator_regex=False)
            texts = splitter.split_documents(doc)

        return texts

    def db_loader(self):
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        texts = self.doc_loader()
        db = FAISS.from_documents(texts, embed)
        # "krlvi/sentence-t5-base-nlpl-code_search_net"
        # "sentence-transformers/all-mpnet-base-v2"
        # db.save_local("faiss_index")
        return db

    def get_context(self, agent_input: str) -> str:
        retriever = self.db_loader().as_retriever(search_kwargs={"k": 2})
        info = retriever.get_relevant_documents(agent_input)
        context = info[0].page_content
        return context


class DSRAG(RAG):
    requires_llm_prompt = False

    def __init__(self, use_rag: bool, tag: str):
        super().__init__(doc=None, doctype=None, use_rag=use_rag, tag=tag)

    def db_loader(self):
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = FAISS.load_local("faiss_index", embed)
        return db


if __name__ == "__main__":
    test = DSRAG()("How to code XGBoost?")
    print(test)
