from fastmcp import FastMCP
import psycopg2
import asyncpg
from datetime import datetime
import json
import re
import langchain

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



mcp = FastMCP(name="Cognitive MCP")

# --------------------------
# Database Initialization
# --------------------------
def initialise_db():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )
    cur = conn.cursor()

    cur.execute("""
    CREATE SCHEMA IF NOT EXISTS cognitive;
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cognitive.cognitive_inputs (
        id SERIAL PRIMARY KEY,
        active_tasks INTEGER NOT NULL,
        urgent_tasks INTEGER NOT NULL,
        context_switches_last_hour INTEGER NOT NULL,
        focus_minutes_today INTEGER NOT NULL,
        sleep_hours_last_night NUMERIC(3,1) NOT NULL,
        self_reported_stress INTEGER NOT NULL CHECK (self_reported_stress BETWEEN 1 AND 10),
        date DATE DEFAULT current_date
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cognitive.cognitive_assessments (
        id SERIAL PRIMARY KEY,
        input_id INTEGER NOT NULL REFERENCES cognitive.cognitive_inputs(id) ON DELETE CASCADE,
        state VARCHAR(50) NOT NULL,
        action VARCHAR(50) NOT NULL,
        confidence NUMERIC(3,2) CHECK (confidence BETWEEN 0 AND 1),
        date DATE DEFAULT current_date
    );
    """)

    conn.commit()
    cur.close()
    conn.close()


# --------------------------
# Insert incoming cognitive data
# --------------------------
@mcp.tool
async def add_data(
    active_tasks: int,
    urgent_tasks: int,
    context_swithes_last_hour: int,
    focus_minutes_today: int,
    sleep_hours_last_night: float,
    self_reported_stress: int,
    date: str
):
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )
    date=datetime.strptime(date,"%Y-%m-%d").date()

    await conn.execute(
        """
        INSERT INTO cognitive.cognitive_inputs
        (active_tasks, urgent_tasks, context_switches_last_hour,
         focus_minutes_today, sleep_hours_last_night,
         self_reported_stress, date)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        """,
        active_tasks,
        urgent_tasks,
        context_swithes_last_hour,
        focus_minutes_today,
        sleep_hours_last_night,
        self_reported_stress,
        date
    )

    await conn.close()
    return {"message": "data inserted into cognitive_inputs"}


# --------------------------
# Generate cognitive signal
# --------------------------
@mcp.tool
async def cognitive_signal_(date: str):
    date_obj = datetime.strptime(date, "%Y-%m-%d").date()

    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )

    row = await conn.fetchrow(
        """
        SELECT *
        FROM cognitive.cognitive_inputs
        WHERE date = $1
        ORDER BY id DESC
        LIMIT 1
        """,
        date_obj
    )

    if not row:
        await conn.close()
        return {"date": date_obj, "signal": "no_data"}

    data = dict(row)
 

    input_id = data["id"]

    model = create_model()

    prompt = f"""
You are a cognitive analyzer.

Return ONLY valid JSON with keys:
- state
- action
- confidence

Allowed states:
- overloaded
- normal
- underutilized
- fatigued

Allowed actions:
- reduce_task_load
- maintain
- increase_focus
- rest

Confidence:
- number between 0 and 1

Input metrics:
{data}
"""

    response = model.invoke(prompt)
    text = response.content

    clean = re.sub(r"```json|```", "", text).strip()
    output = json.loads(clean)

    await conn.execute(
        """
        INSERT INTO cognitive.cognitive_assessments
        (input_id, state, action, confidence, date)
        VALUES ($1,$2,$3,$4,$5)
        """,
        input_id,
        output["state"],
        output["action"],
        output["confidence"],
        date_obj
    )

    await conn.close()

    return {
        "date": date_obj,
        "signal": output
    }


if __name__=="__main__":
    initialise_db()

    mcp.run(transport='http',port=8001,host='0.0.0.0')
    
