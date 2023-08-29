import os
import openai
import json
import sys
from time import sleep
import re
import numpy as np
import ahocorasick

openai.api_key = "YOUR_OPRNAI_KEY"


def chat(prompt):
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": prompt}
  ])
  return completion.choices[0].message.content


merged_triplets = {}


wordlist =[]
with open('./data/explainpe/entity.txt', 'r') as entity_file:
  for line in entity_file:
    wordlist.append(line.rstrip('\n'))


actree = ahocorasick.Automaton()
for index, word in enumerate(wordlist):
  actree.add_word(word,(index, word))
actree.make_automaton()




with open('./data/explainpe/train_ner.json',"a+") as outFile:
  with open(r"./data/explainpe/train_afterprocess.json", 'r', errors='ignore') as file:
    lines = file.readlines()
        
    #623886
    selected_lines = lines[471:]
    
    for line in selected_lines:
        dataset = json.loads(line)


        region_wds = []

        wd = ""
        for j in actree.iter(dataset['input']):
            wd = j[1][1]

        

        PROMPT_DICT = {
            "prompt_question_input":(
                        "这是一个提取实体的范例："
                        "\n\n"
                        "### 指令:\n'学习从以下医学问题中提取实体。'\n\n### 输入:\n"
                        "<CLS>患者，男，80岁，因发热、咳嗽、咳痰2天就诊，诊断为社区获得性肺炎，给予左氧氟沙星、甘草合剂、氨溴索、对乙酰氨基酚、维生素C等治疗。患者用药3天后出现失眠、烦躁不安等症状，最可能引起该症状的药物是\nA: 氨溴索\nB: 维生素C\nC: 甘草合剂\nD: 左氧氟沙星\nE: 对乙酰氨基酚\n<SEP>提取的实体是：\n\n ### 输出:"
                        '<CLS>患者，男，80岁，因发热、咳嗽、咳痰2天就诊，诊断为社区获得性肺炎，给予左氧氟沙星、甘草合剂、氨溴索、对乙酰氨基酚、维生素C等治疗。患者用药3天后出现失眠、烦躁不安等症状，最可能引起该症状的药物是\nA: 氨溴索\nB: 维生素C\nC: 甘草合剂\nD: 左氧氟沙星\nE: 对乙酰氨基酚\n<SEP>提取的实体是：发热，咳嗽，咳痰，社区获得性肺炎，左氧氟沙星，甘草合剂，氨溴索，对乙酰氨基酚，维生素C，失眠，烦躁不安<EOS>'
                        "\n\n"
                        "遵守以上格式，根据输入得到输出:\n\n" 
                        "### 指令:\n'学习从以下医学问题中提取实体。'\n\n### 输入:\n"
                        "<CLS>"+ dataset['input']+"<SEP>提取的实体是：\n\n ### 输出:"
            ),
        }
        
        
        re1 = r'<SEP>提取的实体是：(.*?)<EOS>'
        

        prompt_question_input = PROMPT_DICT["prompt_question_input"]
        prompt_question = [
            prompt_question_input.format_map(example)
            for example in [dataset]
        ]  

        question_kg = []
        
        try:
            str1 = chat(str(prompt_question))
        except:
            sleep(10)
            str1 = chat(str(prompt_question))
        
        
        # print(str1)
        

        
        question_kg = re.findall(re1,str1)
        if question_kg == []:
            try:
                str1 = chat(str(prompt_question))
            except:
                sleep(10)
                str1 = chat(str(prompt_question))
        
        question_kg = re.findall(re1,str1)
        
        if question_kg == []:
            continue
        else:
            question_kg = question_kg[0].split('\uFF0C')

        
        
        
        if wd not in question_kg or '<CLS>' in question_kg[0]:
            try:
                str1 = chat(str(prompt_question))
            except:
                continue
            question_kg = re.findall(re1,str1)
            if question_kg == []:
                continue
            else:
                question_kg = question_kg[0].split('\uFF0C')
            
            if "咳痰" in question_kg and "社区获得性肺炎" in question_kg:
               continue
               
            
            print(question_kg)
        
        
        
        test_dict = {
            'question':dataset['input'],
            'qustion_kg': ",".join(question_kg),
            'answer':dataset['answer']
            
        } 

        json_str = json.dumps(test_dict,ensure_ascii=False)
        outFile.write(json_str+'\n')
            
  file.close()
outFile.close()