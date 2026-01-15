from fastmcp import FastMCP
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("CognitiveMCP")



mcp = FastMCP(name="CognitiveMCP")
 
DATABASE_URL="postgresql://neondb_owner:npg_RPjCGwAZ2Wz5@ep-frosty-frog-a1ectlpt-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

        
async def get_conn():
    import asyncpg

    logger.info("Attempting DB connection")
    try:
        conn = await asyncpg.connect(
            DATABASE_URL,
            ssl="require",
        )
        logger.info("DB connection established")
        return conn
    except Exception as e:
        logger.exception("DB connection failed")
        raise

def safe_parse_json(text: str) -> dict | None:
    import json, re

    logger.info("Attempting to parse model output")
    logger.debug(f"Raw model output: {text}")

    if not isinstance(text, str):
        logger.error("Model output is not a string")
        return None

    text = re.sub(r"```(?:json)?|```", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            logger.info("JSON parsed successfully (direct)")
            return obj
    except json.JSONDecodeError:
        logger.warning("Direct JSON parse failed")

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                logger.info("JSON parsed successfully (regex fallback)")
                return obj
        except json.JSONDecodeError:
            logger.warning("Regex JSON parse failed")

    logger.error("JSON parsing failed completely")
    return None



def create_model():
    from huggingface_hub import InferenceClient
    import os
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Creating inference client")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not set")
        raise RuntimeError("HF_TOKEN not set")

    client = InferenceClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        token=hf_token,
        timeout=30,
    )

    logger.info("Inference client created successfully")
    return client



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


@mcp.tool
async def cognitive_signal_(date: str):
    from datetime import datetime
    import asyncio, json

    logger.info(f"cognitive_signal_ called with date={date}")

    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        logger.exception("Date parsing failed")
        return {"error": "invalid_date"}

    try:
        model = create_model()
    except Exception:
        logger.exception("Model creation failed")
        return {"error": "model_init_failed"}

    try:
        conn = await get_conn()
    except Exception:
        return {"error": "db_connection_failed"}

    logger.info("Fetching cognitive input row")
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
        logger.warning("No cognitive input found for date")
        await conn.close()
        return {"date": str(date_obj), "signal": "no_data"}

    data = dict(row)
    input_id = data["id"]
    logger.info(f"Fetched input_id={input_id}")

    prompt = f"""
Return ONLY a valid JSON object.

Rules:
- Output must be strict JSON
- No markdown
- No explanation
- No extra text
- No extra keys

JSON schema:
{{
  "state": "overloaded | normal | underutilized | fatigued",
  "action": "reduce_task_load | maintain | increase_focus | rest",
  "confidence": 0.0 <= number <= 1.0
}}

Input:
{data}
"""

    logger.info("Calling HF inference model")
    try:
        response = await asyncio.to_thread(
            model.chat_completion,
            messages=[{"role": "user", "content": prompt}],
           
        )
    except Exception:
        logger.exception("Model inference failed")
        await conn.close()
        return {"error": "model_call_failed"}

    try:
        text = response.choices[0].message.content
        logger.info("Model response received")
        logger.debug(f"Model raw text: {text}")
    except Exception:
        logger.exception("Failed to extract model response")
        await conn.close()
        return {"error": "invalid_model_response"}

    output = safe_parse_json(text)

    if not output:
        logger.error("Model output is not valid JSON")
        await conn.close()
        return {
            "date": str(date_obj),
            "signal": "invalid_model_output",
            "raw_output": text[:300],
        }

    logger.info("Inserting cognitive assessment into DB")
    try:
        await conn.execute(
            """
            INSERT INTO cognitive.cognitive_assessments
            (input_id, state, action, confidence, date)
            VALUES ($1,$2,$3,$4,$5)
            """,
            input_id,
            output["state"],
            output["action"],
            float(output["confidence"]),
            date_obj
        )
    except Exception:
        logger.exception("DB insert failed")
        await conn.close()
        return {"error": "db_insert_failed"}

    await conn.close()
    logger.info("cognitive_signal_ completed successfully")

    return {
        "date": str(date_obj),
        "signal": output
    }







if __name__=="__main__":
    

    mcp.run(transport='http',port=8001,host='127.0.0.1')
    
