import asyncio

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from src.centroids import safe_print
from src.schemas import ClassifyRequest, ClassifyResponse
from src.service import run_classification

app = FastAPI()
pipeline_lock = asyncio.Lock()

_HOME = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <title>Classify</title>
</head>
<body>
  <h1>طبقه‌بندی موضوع</h1>
  <p><a href="/docs">Swagger /docs</a></p>
  <label>عنوان</label><br/>
  <input id="title" style="width:80%"/><br/><br/>
  <label>بدنه</label><br/>
  <textarea id="body" rows="12" style="width:80%"></textarea><br/><br/>
  <label>post_id</label><br/>
  <input id="post_id" value="1"/><br/><br/>
  <button id="go">ارسال</button>
  <pre id="out"></pre>
  <script>
    document.getElementById("go").onclick = async () => {
      const out = document.getElementById("out");
      out.textContent = "در حال اجرا...";
      const res = await fetch("/classify", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          title: document.getElementById("title").value,
          body: document.getElementById("body").value,
          post_id: document.getElementById("post_id").value
        })
      });
      out.textContent = JSON.stringify(await res.json(), null, 2);
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return _HOME


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    async with pipeline_lock:
        result = await run_in_threadpool(
            run_classification,
            request.body,
            request.title,
            request.post_id,
        )
        safe_print(result.model_dump_json(ensure_ascii=False))
        return result

