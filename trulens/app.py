from dotenv import load_dotenv

load_dotenv()

import time
import os
import logging

file_path = os.getenv('TRULENS_RESULT_FILE')
app_id = os.getenv('TRULENS_APP_ID')
file_polling_interval = int(os.getenv('TRULENS_FILE_POLLING_INTERVAL'))

def monitor_file_changes(callback):
    last_modified = -1

    while True:
        if os.path.exists(file_path):
            current_modified = os.path.getmtime(file_path)
            if current_modified > last_modified:
                logging.info(f"File has been modified. Reloading...")
                last_modified = current_modified
                callback()
        else:
            logging.info(f"File not found: {file_path}")

        if last_modified == -1: # first run
            logging.info(f"Initializing dashboard...")
            last_modified = 0
            callback()
        time.sleep(file_polling_interval)

def load_json_data():
    import json
    import logging

    json_data = {}

    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        logging.info(f"Data: {json_data}")
        return json_data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON file: {file_path}")
    
    return None

from trulens_eval import Select
retriever_component = Select.RecordCalls.retriever

virtual_app_data = dict(
    llm=dict(
        modelname="GPT-3.5-turbo"
    ),
    template="RAG App Evaluation",
    debug=""
)

def load_rec():
    from trulens_eval.tru_virtual import VirtualRecord

    json_data = load_json_data()

    if not json_data:
        return None
    
    context = json_data['context'] if 'context' in json_data else dict()

    rec = VirtualRecord(
        main_input=json_data['prompt'],
        main_output=json_data['response'],
        calls=
            {
                context_call: dict(
                    args=[json_data['prompt']],
                    rets=[context]
                )
            },
        )
    return rec

from trulens_eval.feedback.provider import OpenAI
from trulens_eval.feedback.feedback import Feedback

# Initialize provider class
openai = OpenAI()

# The selector for a presumed context retrieval component's call to
# `get_context`. The names are arbitrary but may be useful for readability on
# your end.
context_call = retriever_component.get_context

# Select context to be used in feedback. We select the return values of the
# virtual `get_context` call in the virtual `retriever` component. Names are
# arbitrary except for `rets`.
context = context_call.rets[:]

from trulens_eval.feedback import Groundedness
import numpy as np
grounded = Groundedness(groundedness_provider=OpenAI())
# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(context.collect()) # collect context chunks into a list
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance, name="Answer Relevance").on_input_output()
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(openai.qs_relevance, name="Context Relevance")
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

def load_app():
    from trulens_eval import Select
    from trulens_eval.tru_virtual import VirtualApp

    virtual_app = VirtualApp(virtual_app_data) # can start with the prior dictionary
    virtual_app[Select.RecordCalls.llm.maxtokens] = 1024

    virtual_app[retriever_component] = "this is the retriever component"

    return virtual_app

def load_virtual_recorder():
    virtual_app = load_app()

    from trulens_eval.tru_virtual import TruVirtual

    virtual_recorder = TruVirtual(
        app_id=app_id,
        app=virtual_app,
        feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance],
        initial_app_loader=load_app
    )

    rec = load_rec()

    if rec is not None:
        virtual_recorder.add_record(rec)

def run_dashboard():
    load_virtual_recorder()

    from trulens_eval import Tru

    tru = Tru()
    
    if tru._dashboard_proc is None:
        tru.reset_database()

    if tru._dashboard_proc is not None:
        tru.stop_dashboard(force=True)
    
    tru.get_leaderboard(app_ids=[app_id])

    tru.run_dashboard()

monitor_file_changes(run_dashboard)
