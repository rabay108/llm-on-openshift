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
#from prometheus_client import Gauge, start_http_server, Counter
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
# Example usage
provider_model = "your_OpenShift AI (vLLM): ibm-granite-instruct"  
query = "Red Hat Summit and Ansiblefest2025"  # Replace with your actual query text



def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata["source"] not in unique_list:
            unique_list.append(item.metadata["source"])
    return unique_list

lock = threading.Lock()
def stream(chain, que, model_input: dict, session_id, model_id) -> Generator:
    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        ##MODEL_USAGE_COUNTER.labels(model_id=model_id).inc()
        # Call this function at the start of your application
        ##initialize_feedback_counters(model_id)
        with lock:
            start_time = (
                time.perf_counter()
            )  # start and end time to get the precise timing of the request
            try:
                resp = chain.invoke(input=model_input)
                end_time = time.perf_counter()
                sources = remove_source_duplicates(resp["source_documents"])
                ##REQUEST_TIME.labels(model_id=model_id).set(end_time - start_time)
                print(resp["result"])
                #create_pdf(resp["result"], session_id)
                if len(sources) != 0:
                    que.put("\n*Sources:* \n")
                    for source in sources:
                        que.put("* " + str(source) + "\n")
            except Exception as e:
                print(e)
                que.put("Error executing request. Contact the administrator.")

            que.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = que.get(True, timeout=100)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue
def get_provider_model(provider_model):
    if provider_model is None:
        return "", ""
    s = provider_model.split(": ")
    return s[0], s[1]

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

def generate_response(provider_model, query):
    chain_without_llm = QueryHelper.get_qa_chain

    query = f" '{query}'"
    model_input = {'query': query}
    
    for content, download_link_html in ask_llm(provider_model, model_input, chain_without_llm):
        yield content, download_link_html
# Call the method
for content, download_link_html in generate_response(provider_model, query):
    print(f"Content: {content}")
    print(f"Download Link HTML: {download_link_html}")