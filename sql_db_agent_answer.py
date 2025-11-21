import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
import httpx
from sqlalchemy import create_engine

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Load env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
http_client = httpx.Client(verify=False)

# LLM (your endpoint)
model = ChatOpenAI(
    api_key=openai_key,
    model="openai/gpt-4.1",
    base_url="https://models.github.ai/inference",
    http_client=http_client,
)

# ----------------------------
# Build SQLite database
# ----------------------------
database_file_path = "./db/salary.db"
engine = create_engine(f"sqlite:///{database_file_path}")

file_url = "./data/salaries_2023.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)

df = pd.read_csv(file_url).fillna(value=0)
df.to_sql("salaries_2023", con=engine, if_exists="replace", index=False)

# ----------------------------
# Connect DB for LangChain
# ----------------------------
db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
query_tool = QuerySQLDataBaseTool(db=db)

# ----------------------------
# OLD SQL-AGENT PROMPT (Your Original)
# ----------------------------
MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.

## Instructions:
- Given an input question, create a syntactically correct SQL query to run,
  then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples, ALWAYS limit to 30 rows.
- ONLY select relevant columns, never SELECT *.
- If a query fails, rewrite it and try again.
- DO NOT execute INSERT, UPDATE, DELETE, DROP, or other destructive commands.
- NEVER hallucinate table names — only use the tables that exist in the database.
- NEVER add markdown backticks inside the SQL query itself.
- Final answer must always contain:
  - The SQL query used.
  - The SQL result.
  - An Explanation section showing how you arrived at the answer.
- Always provide your SQL inside ```sql blocks in the final answer (but NOT during execution).
- If the question is not related to the database, answer "I don't know".

Your task: Convert the user's question into a SQL query that answers the question.
"""

# ----------------------------
# Updated Prompt (Includes OLD logic)
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", MSSQL_AGENT_PREFIX),
    ("human", "{question}")
])

# ----------------------------
# Modern SQL Agent (Runnable)
# ----------------------------
def call_sql_tool(question):
    return query_tool.run(question)

sql_agent = RunnableSequence(
    prompt | model,
)

def clean_sql(sql_text: str) -> str:
    return (
        sql_text.replace("```sql", "")
                .replace("```", "")
                .strip()
    )

# ----------------------------
# Ask a question
# ----------------------------
QUESTION = "What is the highest average salary by department?"

# 1. Convert NL → SQL
sql_query = sql_agent.invoke({"question": f"{QUESTION}"})
print("\nGenerated SQL query:\n", sql_query.content)

# 2. Clean & Execute SQL
sql_query_clean = clean_sql(sql_query.content)
sql_result = call_sql_tool(sql_query_clean)
print("\nSQL Result:\n", sql_result)

# 3. Final Answer with SQL + results + explanation
final_prompt = f"""
Use the query and results below to produce the final answer.

### SQL Query
```sql
{sql_query_clean}
``` 
### SQL Results
{sql_result}
### Final Answer
final_answer = model.invoke(final_prompt)
print("\nFinal Answer:\n", final_answer.content)
final_answer = model.invoke(
    f"Here are the SQL results: {sql_result}. Provide the final answer."
)
print("\nFinal Answer:\n", final_answer.content)
"""

