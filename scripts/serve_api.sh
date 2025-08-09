#!/usr/bin/env bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
