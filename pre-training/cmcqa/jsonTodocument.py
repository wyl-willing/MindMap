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
import json


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
    以下是一个表格.
    \n
    疾病\t症状\t诊断测试\t推荐药物
    {row}
    \n\n
    将csv转换为自然语言。输出一个字符串句子。
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


for data in open('./data/cmcqa/medical.json'):
    count += 1
    data_json = json.loads(data)
    disease = data_json['name']
    symptom = data_json['symptom']
    test = data_json['check']
    try:
        medical = data_json['common_drug']
    except:
        medical = []

    input_str = '\t'.join([str(disease),','.join(symptom),','.join(test),','.join(medical)])
    
    filename = str(count) + ".txt"

    document_value = prompt_document(input_str)
    if is_unable_to_answer(document_value):
        document_value = prompt_document(input_str)

        
    print(document_value)

    with open(os.path.join(current_path, "document", filename), "w", encoding="utf-8") as f:
        f.write(document_value)
