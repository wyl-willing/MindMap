# This is a Chinese version of MindMap.
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
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from langchain.llms import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec



def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    
    global exist_entity
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "OPTIONAL MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            if path is not None:
                for i in range(len(path.nodes)):
                    node = path.nodes[i]
                    entity_name = node["name"]
                    entities.append(entity_name)
                    if i < len(path.relationships):
                        relationship = path.relationships[i]
                        relation_type = relationship.type
                        relations.append(relation_type)
           
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i]
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i]
                    path_str += "->" + relations[i] + "->"
            
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}

        if short_path == 0:
            exist_entity = {}

        if len(paths) > 5: 
                   
            paths = sorted(paths, key=len)[:4]

        # print('short_path',short_path)
        # print('paths',paths)
        # print('exist_entity',exist_entity)
        return paths,exist_entity


def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results


def get_entity_neighbors(entity_name):
   
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        
     
        neighbors = record["neighbor_entities"]
        
        
        neighbor_list.append([entity_name, rel_type, 
                            ','.join([x for x in neighbors])
                            ])

    return neighbor_list

def prompt_path_finding(path_input):
    template = """
    以下是一些知识图谱路径，遵循“实体->关系->实体”的格式。
    \n\n
    {Path}
    \n\n
    使用以上知识图谱路径知识，分别将它们翻译为自然语言总结描述。用单引号标注实体名和关系名。并将它们命名为路径证据1, 路径证据2....\n输出尽量精简，减少token数，但不能丢失事实。\n\n

    输出:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})

    response_of_KG_path = chat(chat_prompt_with_values.to_messages()).content
    return response_of_KG_path

def prompt_neighbor(neighbor):
    template = """
    以下是一些知识图谱路径，遵循“实体->关系->实体”的格式。
    \n\n
    {neighbor}
    \n\n
    使用以上知识图谱路径知识，分别将它们翻译为自然语言总结描述。用单引号标注实体名和关系名。 并将它们命名为邻居证据1, 邻居证据2...\n\n

    输出:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

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

def prompt_comparation(reference,output1,output2):
    template = """
    Reference: {reference}
    \n\n
    output1: {output1}
    \n\n
    output2: {output2}
    \n\n
    According to the reference output, which output is better. If the answer is output1, output '1'. If the answer is output2, output '2'.
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["reference","output1","output2"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(reference = reference,
                                 output1 = output1,
                                 output2 = output2)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(reference = reference,\
                                                        output1 = output1,\
                                                        output2 = output2,\
                                                        text={})

    response_of_comparation = chat(chat_prompt_with_values.to_messages()).content

    return response_of_comparation

def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def final_answer(str,response_of_KG_list_path,response_of_KG_neighbor):
    messages  = [
                SystemMessage(content="你是一名优秀的AI医生,你可以根据对话中的症状诊断疾病并推荐药物和提出治疗与检查方案。"),
                HumanMessage(content="患者输入:"+ input_text),
                AIMessage(content="你拥有以下医学证据知识:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
                HumanMessage(content="参考提供的路径证据和邻居证据知识，根据患者输入的症状描述，请问患者患有什么疾病?确认疾病需要什么检查来诊断?推荐的治疗疾病的药物和食物是什么?忌吃什么?一步步思考。\n\n\n"
                            + "输出1：回答应包括疾病和检查已经推荐的药物和食物。\n\n"
                             +"输出2：展示推理过程，即从哪个路径证据或邻居证据中提取什么知识，最终推断出什么结果。 \n 将推理过程转化为以下格式:\n 路径证据标号('实体名'->'关系名'->...)->路径证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->结果标号('实体名')->路径证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->结果标号('实体名')->... \n\n"
                             +"输出3：画一个决策树。在输出2的的推理过程中，单引号中的实体或关系与用括号包围的证据来源一起作为一个节点。\n\n"
                             + "以下是一个样例，参考其中的格式:\n"
                             + """
输出1：
根据所描述的症状，患者可能患有喉炎，这是声带的炎症。为了确认诊断，患者应该接受喉咙的身体检查，可能还需要喉镜检查，这是一种使用镜检查声带的检查。治疗喉炎的推荐药物包括抗炎药物，如布洛芬，以及减少炎症的类固醇。还建议让声音休息，避免吸烟和刺激物。

输出2：
路径证据1(“患者”->“症状”->“声音沙哑”)->路径证据2(“声音沙哑”->“可能疾病”->“喉炎”)->邻居证据1(“喉咙体检”->“可能包括”->“喉镜检查”)->邻居证据2(“喉咙体检”->“可能疾病”->“喉炎”)->路径证据3(“喉炎”->“推荐药物”->“消炎药和类固醇”)-邻居证据3(“消炎药和类固醇”->“注意事项”->“休息声音和避免刺激”)。

输出3:：
患者(路径证据1)
└── 症状(路径证据1)
    └── 声音沙哑(路径证据1)(路径证据2)
        └── 可能疾病(路径证据2)
            └── 喉炎(路径证据2)(邻居证据1)
                ├── 需要(邻居证据1)
                │   └── 喉咙体检(邻居证据1)(邻居证据2)
                │       └── 可能包括(邻居证据2)
                │           └── 喉炎(邻居证据2)(结果1)(路径证据3)
                ├── 推荐药物(路径证据3)
                │   └── 消炎药和类固醇(路径证据3)(结果2)(邻居证据3)
                └── 注意事项(邻居证据3)
                    └── 休息声音和避免刺激(邻居证据3)
                                    \n\n\n"""
                            + "参考以上样例的格式得到针对患者输入的输出。\n并分别命名为“输出1”，“输出2”，”输出3“。"
                             )

                                   ]
        
    result = chat(messages)
    output_all = result.content
    return output_all

def prompt_document(question,instruction):
    template = """
    你是一个优秀的AI医生，你可以根据对话中的症状诊断疾病并推荐药物。\n\n
    患者输入:\n
    {question}
    \n\n
    以下是您的一些医学知识信息:
    {instruction}
    \n\n
    病人得了什么病?患者需要做哪些检查来确诊?推荐哪些药物可以治愈这种疾病?
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content

    return response_document_bm25


def comparation_score(response_of_comparation,compare_name):
    count = 0
    countTrue = 0
    if response_of_comparation == '1' or 'output1':
        response_of_comparation = 'MindMap'
        count += 1
        countTrue += 1
    else:
        response_of_comparation = compare_name
        count += 1
    score = countTrue/count
    return response_of_comparation

if __name__ == "__main__":
    

    os.environ['OPENAI_API_KEY']="YOUR_OPRNAI_KEY"

    # 1. build neo4j knowledge graph datasets
    uri = "YOUR_URL"
    username = "neo4j"     
    password = "YOUR_PASSWORD"     

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()


    session.run("MATCH (n) DETACH DELETE n")# clean all

    # read triples
    df = pd.read_csv('./data/kg_triples_small.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])


    for index, row in df.iterrows():
        head_name = row['head']
        tail_name = row['tail']
        relation_name = row['relation']
       
        try:
            query = (
                "MERGE (h:Entity { name: $head_name }) "
                "MERGE (t:Entity { name: $tail_name }) "
                "MERGE (h)-[r:`" + relation_name + "`]->(t)"
            )
            session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)

        except:
            continue
        
# # 2. OpenAI API based keyword extraction and match entities

    OPENAI_API_KEY="YOUR_OPENAI_KEY"
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)


    # with open('./dataset2_output/output_dataset2.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Question', 'Label', 'MindMap_output1','MindMap_output2','MindMap_output3'])


    with open('./data/cmcqa/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./data/cmcqa/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)


    with open("/home/willing/MindMap-KG/dataset2_output/remind_dataset2.json", "r") as f:
        for line in f.readlines()[:]:
            flag_openai = 1
            count = 0
            x = json.loads(line)
            input_text = x["question"][0]
                        
            if input_text == []:
                continue
            print(input_text)

            
            output_text = x["answer"]
            print(output_text)
  
            question_kg = x["question_kg"]
            question_kg = question_kg.split(",")
            if len(question_kg) == 0:
                print("<Warning> no entities found", input)
                continue
            # print("question_kg",question_kg)

            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
        

            for kg_entity in question_kg:
            
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]

                max_index = cos_similarities.argmax()
                if cos_similarities.argmax() < 0.5:
                    continue
                        
                match_kg_i = entity_embeddings["entities"][max_index]

                while match_kg_i in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i)
            print('match_kg',match_kg)

    #       # # 4. neo4j knowledge graph path finding 

            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]
                
                result_path_list = []
                while 1:
                    flag = 0
                    paths_list = []
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)
                        paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                      
                    path_list = []
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                paths_list.append(path_list)
                        
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
    
                    result_path = combine_lists(*paths_list)                
                
                    if result_path != []:
                        result_path_list.extend(result_path)
                
                    if flag == 1:
                        continue
                    else:
                        break
                    
                
                # if len(result_path_list)> 5:
                #     result_path = result_path_list[:5]
                start_tmp = []
                for path_new in result_path_list:
                
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                if len(start_tmp) == 0:
                        result_path = {}
                        kg_retrieval = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]

                    else:
                        result_path = []
                        
                            
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
                            count_tmp = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                count_tmp += 1

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    try:
                        kg_retrieval = result_path_list[0]
                    except:
                        kg_retrieval = result_path_list
                    
            else:
                result_path = {}
                kg_retrieval = {}
            
            print('result_path',result_path)

        
            

    # # 5. neo4j knowledge graph neighbor entities
            neighbor_list = []
        
            for match_entity in match_kg:
                disease_flag = 0
                neighbors = get_entity_neighbors(match_entity)
                neighbor_list.extend(neighbors)

            if result_path != {}:
                if len(neighbor_list) > 5:
                    new_neighbor = []
                    for neighbor_new in neighbor_list:
                        if "疾病" not in neighbor_new[1] and "症状" not in neighbor_new[1]:# 更改图谱后修改这里
                            new_neighbor.append(neighbor_new)

                    neighbor_list = new_neighbor

                if len(neighbor_list) > 5:
                    neighbor_list_tmp = []
                    for neighbor in neighbor_list:
                        if neighbor[1] == '常用药品':
                            neighbor_list_tmp.append(neighbor)
                        if len(neighbor_list_tmp) >= 5:
                            break
                    if len(neighbor_list_tmp) < 5:
                        for neighbor in neighbor_list:
                            if neighbor[1] == '诊断检查':
                                neighbor_list_tmp.append(neighbor)
                            if len(neighbor_list_tmp) >= 5:
                                break

                    if len(neighbor_list_tmp) < 5:
                        for neighbor in neighbor_list:
                            neighbor_list_tmp.append(neighbor)
                            if len(neighbor_list_tmp) >= 5:
                                break
                        

                    neighbor_list = neighbor_list_tmp

            print("neighbor_list",neighbor_list)


    #         # 6. knowledge gragh path based prompt generation
            
            if len(match_kg) != 1 and len(match_kg) != 0 and result_path != {}:
                response_of_KG_list_path = []
                if result_path == [] or {}:
                    response_of_KG_list_path = '{}'
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    
                    try:
                        path = "\n".join(result_new_path[:3])
                        response_of_KG_list_path = prompt_path_finding(path)
                        if is_unable_to_answer(response_of_KG_list_path):
                            response_of_KG_list_path = prompt_path_finding(path)
                        print("response_of_KG_list_path",response_of_KG_list_path)
                    except:
                        path = "\n".join(result_new_path)[:4]
                        response_of_KG_list_path = prompt_path_finding(path)
                        if is_unable_to_answer(response_of_KG_list_path):
                            response_of_KG_list_path = prompt_path_finding(path)
                        print("response_of_KG_list_path",response_of_KG_list_path)

                    
            else:
                response_of_KG_list_path = '{}'

            response_kg_retrieval = prompt_path_finding(kg_retrieval)
            if is_unable_to_answer(response_kg_retrieval):
                response_kg_retrieval = prompt_path_finding(kg_retrieval)

    #         # # 7. knowledge gragh neighbor entities based prompt generation   
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

            

            try:
                neighbor_input = "\n".join(neighbor_new_list)
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            
                if is_unable_to_answer(response_of_KG_neighbor):
                    response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            except:
                neighbor_input = neighbor_new_list[:2]
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)
                
                if is_unable_to_answer(response_of_KG_neighbor):
                    response_of_KG_neighbor = prompt_neighbor(neighbor_input)

                

            print("response_of_KG_neighbor",response_of_KG_neighbor)


    #         # # 8. prompt-based medical diaglogue answer generation

            
            output_all = final_answer(input_text,response_of_KG_list_path,response_of_KG_neighbor)
            if is_unable_to_answer(output_all):
                output_all = final_answer(input_text,response_of_KG_list_path,response_of_KG_neighbor)

            print('output_all',output_all)
        

            
            re4 = r"输出1：(.*?)输出2："
            re5 = r"输出2：(.*?)输出3："

            flag_wrong = 0
            output1 = re.findall(re4, output_all, flags=re.DOTALL)
            if len(output1) > 0:
                output1 = output1[0]
            else:
                flag_wrong = 1
            
            output2 = re.findall(re5, output_all, flags=re.DOTALL)
            if len(output2) > 0:
                output2 = output2[0]
            else:
                flag_wrong = 1
                
            output3_index = output_all.find("输出3：")
            if output3_index != -1:
                output3 = output_all[output3_index + len("输出3："):].strip()

            if flag_wrong == 1:
                with open('./dataset2_output/train_wrong.json',"a+",encoding='utf-8') as f7:
                  
                    test_dict = {
            'question':input_text,
            'qustion_kg': question_kg,
            'answer':output_text
            
        } 
                    json_str = json.dumps(test_dict,ensure_ascii=False)
                    f7.write(json_str+'\n')

                continue

            try:
                with open('./dataset2_output/output_dataset2.csv', 'a+', newline='') as f6:
                    writer = csv.writer(f6)
                
                    writer.writerow([input_text, output_text,output1,output2,output3])
                    f6.flush()
            except:
                with open('./dataset2_output/train_wrong.json',"a+",encoding='utf-8') as f7:
                                      
                    test_dict = {
            'question':input_text,
            'qustion_kg': question_kg,
            'answer':output_text
            
        } 
                    json_str = json.dumps(test_dict,ensure_ascii=False)
                    f7.write(json_str+'\n')

                
                            
    

