from fastmcp import FastMCP

DATABASE_URL ="postgresql://neondb_owner:npg_RPjCGwAZ2Wz5@ep-frosty-frog-a1ectlpt-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
mcp = FastMCP(name="CognitiveMCP")



async def get_conn():
    import asyncpg
    return await asyncpg.connect(
        DATABASE_URL,
        ssl="require",
    )



def create_model():
   import langchain_huggingface 
   from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
    
   hf_token="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"
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
    from datetime import datetime
    conn = await get_conn()
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
    from datetime import datetime
    import re 
    import json
    date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    model=create_model()
    conn = await get_conn()

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

    response = await model.ainvoke(prompt)
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
    

    mcp.run(transport='http',port=8001,host='127.0.0.1')
    
