import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import os
import openai



def is_unable_to_answer(response):
 
    analysis = openai.Completion.create(
    engine="text-davinci-002",
    prompt=response,
    max_tokens=1,
    temperature=0.0,
    n=1,
    stop=None,
    presence_penalty=0.0,
    frequency_penalty=0.0
)
    score = analysis.choices[0].text.strip().replace("'", "").replace(".", "")
    if not score.isdigit():   
        return True
    threshold = 0.6
    if float(score) > threshold:
        return False
    else:
        return True


def row_to_string(row):
    return ",".join([str(val) for val in row])


def prompt_document(row):
    template = """
    There is a csv.
    \n
    disease,Symptom,Medical Tests,Medications
    {row}
    \n\n
    Convert the csv to natural language. Output a string sentence.
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["row"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(row = row)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(row = row,\
                                                        text={})

    response_document_row = chat(chat_prompt_with_values.to_messages()).content

    return response_document_row


os.environ['OPENAI_API_KEY']="YOUR_OPENAI_KEY"
OPENAI_API_KEY="YOUR_OPENAI_KEY"
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

current_path = os.path.abspath(os.getcwd())

df = pd.read_csv("disease_database_mini.csv", sep=",", encoding="utf-8")


for index, row in df.iterrows():
    row = row.transpose()
    print(row)

    filename = str(row[0]) + ".txt"
    text = ",".join([str(val) for val in row[1:]])
    document_value = prompt_document(text)
    if is_unable_to_answer(document_value):
        document_value = prompt_document(text)

        
    print(document_value)


    with open(os.path.join(current_path, "document", filename), "w", encoding="utf-8") as f:
        f.write(document_value)
