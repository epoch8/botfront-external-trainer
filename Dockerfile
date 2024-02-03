FROM python:3.10

WORKDIR /app

RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.2.2 python3 -
ENV PATH "/root/.local/bin:${PATH}"

COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev

COPY /app /app

CMD poetry run uvicorn --host 0.0.0.0 app:app
