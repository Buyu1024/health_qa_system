# 导入 FastAPI 相关模块，用于构建 API 和 WebSocket
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
# 导入 FastAPI 响应类型，用于流式响应和文件服务
from fastapi.responses import StreamingResponse, FileResponse
# 导入 CORS 中间件，支持跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 导入静态文件服务模块
from fastapi.staticfiles import StaticFiles
# 导入 WebSocket 断开异常
from starlette.websockets import WebSocketDisconnect
# 导入系统操作模块，用于文件目录管理
import os
# 导入 Pydantic 模型，用于请求验证
from pydantic import BaseModel
# 导入异步事件循环模块
import asyncio
# 导入 JSON 处理模块
import json
# 导入 UUID 模块，生成唯一会话 ID
import uuid
# 导入类型注解模块
from typing import Optional, List, Dict, Any
# 导入时间模块，记录处理时间
import time
# 导入正则表达式模块，用于匹配日常问候
import re

from contextlib import asynccontextmanager
# 导入问答系统
from main import IntergrateQASystem

qa_system = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_system
    try:
        print("正在初始化集成问答系统...")
        qa_system = IntergrateQASystem()
        print("集成问答系统初始化完成！")
    except Exception as e:
        print(f"集成问答系统初始化失败：{e}")
        raise e
    yield
    if qa_system and hasattr(qa_system, 'mysql_client'):
        qa_system.mysql_client.close()
        print("已关闭MySQL数据库连接")

app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("static", exist_ok=True)

GREEING_PATTERNS = [
    {
        "pattern": r"你好|hi|hello",
        "response": "你好，我是膳食与慢病食养知识问答系统，你可以向我提问任何问题。"
    }
]

class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = qa_system.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败：{e}")

@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    result = qa_system.clear_session_history(session_id)
    if result:
        return {"message": "会话历史已清空"}
    else:
        raise HTTPException(status_code=500, detail="清空会话历史失败")

def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern in GREEING_PATTERNS:
        match = re.match(pattern["pattern"], query, re.IGNORECASE)
        if match:
            return pattern["response"]
    return None

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            source_filter = request_data.get("source_filter")
            query = request_data.get("query")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id,
                })
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id,
                    })
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time,
                    })
                break
            collected_answer = ""
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                collected_answer += token
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id,
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                        })
                    break
                await asyncio.sleep(0.01)
    except WebSocketDisconnect as e:
        print(f"WebSocket连接断开：code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"WebSocket error：{e}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.close()
        except Exception as e:
            print(f"WebSocket close error：{e}")

@app.get("/heath")
async def heath_check():
    return {"status": "healthy"}

@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)