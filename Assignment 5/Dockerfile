FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked

COPY main.py run_server.py pipeline_v1.bin ./

EXPOSE 8000

CMD ["uv", "run", "python", "run_server.py"]