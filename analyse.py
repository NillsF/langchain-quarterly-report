import os
import dotenv
import logging

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# Load environment variables from .env file
dotenv.load_dotenv(dotenv_path='./.env')

# Create an instance of the AzureChatOpenAI class using Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    openai_api_version="2023-05-15",
    verbose=True)

# Testing chat llm  
res = llm([HumanMessage(content="Tell me a joke about a penguin sitting on a fridge.")])
print(res)

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

docs_folder = "docs-to-analyse"
for filename in os.listdir(docs_folder):
    logger.info(f"Analyzing file: {filename}")
    loader = PyPDFLoader(os.path.join(docs_folder, filename))
    pages = loader.load_and_split()

    prompt_template = """
    You're an analyst reviewing a document. You have 3 tasks:
    Task 1: Generate Document Summary. When writing the summary you prefer completeness over conciseness and include specific terminology used in the document.
    Task 2: Include all sentences that include Microsoft, OpenAI, Amazon, AWS, Google and/or GCP. You will include the exact sentence mentioning these companies in the document summary.
    Task 3: Detect Investments and releases in Technology, Data Centers, or Cloud and include them in the document summary.
    Task 4: Detect Investments and releases in innovative technology, artificial intelligence, or machine learning and include them in the document summary.
    Task 5: Major initiatives at the company driving either cost savings or revenue growth and include them in the document summary.
    {text}

    Result of all 5 tasks IN ENGLISH:"""
    MAPPROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])


    #chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=MAPPROMPT,combine_prompt=PROMPT)
    #print(chain({"input_documents": pages}, return_only_outputs=True))
    
    chain = load_summarize_chain(llm , chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
    print(chain({"input_documents": pages}, return_only_outputs=True))

    


