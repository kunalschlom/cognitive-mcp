import langchain
import langgraph
import langchain_core
import langchain_huggingface 
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint, HuggingFaceEmbeddings
import os 

from dotenv import load_dotenv
load_dotenv()
def create_model():
    
   hf_token=os.getenv("HF_TOKEN")
   if not hf_token:
        raise ValueError("the hf token is not available")   
   repo_id="Qwen/Qwen2.5-7B-Instruct"     
   llm=HuggingFaceEndpoint(
       repo_id=repo_id,
       huggingfacehub_api_token=hf_token,
       task="conversational"
   ) 
   model=ChatHuggingFace(llm=llm)

   return model 

