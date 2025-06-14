import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY not found in .env file")

os.environ["OPENAI_API_KEY"] = api_key

db_path = "sqlite:///data/inference.db"
engine = create_engine(db_path)
db = SQLDatabase(engine)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False)

if len(sys.argv) < 2:
    sys.exit("Usage: python sql_agent.py \"<your question>\"")

question = sys.argv[1]
response = db_chain.invoke({"query": question})
print(response)
