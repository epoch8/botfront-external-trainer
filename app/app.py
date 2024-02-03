from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated, Union

DEV = os.getenv('DEV')
logging.basicConfig(level=logging.DEBUG if DEV else logging.INFO)

from fastapi import FastAPI, UploadFile, Request, Response, HTTPException, Form
from fastapi.responses import PlainTextResponse, StreamingResponse, FileResponse

from job_manager import K8sJobManager

AUTH_TOKEN = os.getenv("AUTH_TOKEN")


@dataclass
class Ctx:
    job_manager: K8sJobManager | None = None


_ctx = Ctx()


def get_job_manager() -> K8sJobManager:
    assert _ctx.job_manager
    return _ctx.job_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ctx.job_manager = K8sJobManager()
    yield


app = FastAPI(lifespan=lifespan)


def verify_token(request: Request) -> None:
    token = request.headers.get("Authorization")
    if token != AUTH_TOKEN:
        raise HTTPException(401)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/train")
def train(
    request: Request,
    project_id: Annotated[str, Form()],
    image: Annotated[str, Form()],
    training_data: UploadFile,
    rasa_extra_args: str | None = Form(None),
    node: str | None = Form(None),
    use_cache: bool = Form(True),
    is_rasa_for_botfront: bool = Form(False),
):
    verify_token(request)
    job_id, created = get_job_manager().train(
        project_id,
        image,
        training_data.file.read().decode(),
        rasa_extra_args=rasa_extra_args,
        node=node,
        use_cache=use_cache,
        is_rasa_for_botfront=is_rasa_for_botfront,
    )
    return {'job_id': job_id, 'created': created}


@app.get("/status")
def status(request: Request, job_id: str):
    verify_token(request)
    res = get_job_manager().status(job_id)
    return {'status': res}


@app.post("/cancel")
def cancel(request: Request, job_id: str):
    verify_token(request)
    res = get_job_manager().cancel(job_id)
    return {'cancelled': res}


@app.get("/logs", response_class=PlainTextResponse)
def logs(request: Request, job_id: str):
    verify_token(request)
    res = get_job_manager().logs(job_id)
    return res or ''


@app.get("/result")
def result(request: Request, job_id: str):
    verify_token(request)
    result = get_job_manager().result(job_id)
    if result is None:
        raise HTTPException(404)
    media_type = 'application/tar+gzip'
    if isinstance(result, Path):
        return FileResponse(result, media_type=media_type)
    return StreamingResponse(result, media_type=media_type)
