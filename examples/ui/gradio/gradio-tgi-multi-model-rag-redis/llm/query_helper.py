from langchain.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
from vector_db.db_provider_factory import FAISS, DBFactory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnableLambda
############################
# LLM chain implementation #
############################

db_factory = DBFactory()

GENERATE_PROPOSAL_THEMPLATE = """
### [INST]
Instructions:
- You are a helpful assistant in writing a project proposal for products owned by Red Hat for the company name provided in the question below.
- Your job is to look at the  question and create the project proposal addressed to the company mentioned in question.
- Base your answer on the provided context and question and not on prior knowledge.
- The proposal should contain headings and sub-headings and each heading and sub-heading should be in bold.
- Generate the project proposal in markdown language.
- Each section in the proposal should contain only three items.
- Proposal should be minimum of 500 lines.

Here is context to help:
{context}

### QUESTION:
Question: {question}

[/INST]
"""

UPDATE_PROPOSAL_TEMPLATE = """
### [INST]
Instructions:
- You are a helpful assistant tasked with updating a project proposal for products owned by Red Hat.
- Update the old proposal based on the user query, using the provided old proposal, context, and question.
- Do not rely on prior knowledge; base your response solely on the provided information.
- Update the proposal in markdown format.
- Each section of the proposal should contain only three items.
- Ensure the proposal is a minimum of 500 lines.
- Modify only the content based on the user's request, while keeping everything else the same, but you can renumber sections.

Context:
{context}

Old Proposal:
{old_proposal}

### User Query:
{user_query}
[/INST]
"""

QUERY_UPDATE_PROPOSAL_TEMPLATE = """"
### [INST]
Old Proposal:
{old_proposal}

### User Query:
{user_query}
[/INST]
"""

def _get_retriever():
    try:
        type = os.getenv("DB_TYPE") if os.getenv("DB_TYPE") else "REDIS"
        if type is None:
            raise ValueError("DB_TYPE is not specified")
        print(f"Retriever DB: {type}")
        retriever = db_factory.get_retriever(type)
    except Exception as e:
        print(e)
        print(
            f"{type} server is unavailable. Project proposal will be generated without RAG content."
        )
        retriever = db_factory.get_retriever(FAISS)  
    return retriever

def get_qa_chain(llm):
    generate_proposal_prompt = PromptTemplate.from_template(GENERATE_PROPOSAL_THEMPLATE)
    retriever = _get_retriever()

    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": generate_proposal_prompt},
        return_source_documents=True,
    )

def get_update_proposal_chain(llm):
    update_proposal_prompt = PromptTemplate.from_template(UPDATE_PROPOSAL_TEMPLATE)
    query_update_proposal_prompt = PromptTemplate.from_template(QUERY_UPDATE_PROPOSAL_TEMPLATE)
    retriever = _get_retriever()
    combine_docs_chain = create_stuff_documents_chain(llm, update_proposal_prompt)

    return RunnableParallel({'context': query_update_proposal_prompt| RunnableLambda(lambda x: x.text)  | retriever, 'old_proposal': lambda x:x['old_proposal'], 'user_query': lambda x: x['user_query']}) | RunnableParallel({"source_documents": lambda x: x['context'], 'result': combine_docs_chain})