import os
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
import os
from markdown import markdown
from llm.llm_factory import LLMFactory, NVIDIA
import pdfkit
import uuid
import threading
import gradio as gr
from prometheus_client import Gauge, start_http_server, Counter
#from dotenv import load_dotenv
from utils import config_loader
import llm.query_helper as QueryHelper
from scheduler.round_robin import RoundRobinScheduler
import pandas as pd
from utils.callback import QueueCallback

que = Queue()

os.environ["REQUESTS_CA_BUNDLE"] = ""
# initialization
#load_dotenv()
config_loader.init_config()
llm_factory = LLMFactory()
llm_factory.init_providers(config_loader.config)
print("Call AskLLM")

# Gradio implementation
def ask_llm(provider_model, model_input, chain_without_llm):
    que = Queue()
    callback = QueueCallback(que)
    session_id = str(uuid.uuid4())
    provider_id, model_id = get_provider_model(provider_model)
    llm = llm_factory.get_llm(provider_id, model_id, callback)
    chain = chain_without_llm(llm)

    for next_token, content in stream(chain, que, model_input, session_id, model_id):
        # Generate the download link HTML
        download_link_html = f' <input type="hidden" id="pdf_file" name="pdf_file" value="/file={get_pdf_file(session_id)}" />'
        yield content, download_link_html