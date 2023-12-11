import openai
import csv
from time import sleep


openai.api_key = "<OPENAI-KEY>"

def prompt_comparation(reference,output1,output2):
    template = """
    Reference: {reference}
    \n\n
    output1: {output1}
    \n\n
    output2: {output2}
    \n\n
    According to the disease diagnosis result in reference output, which output is better match and complete. If the output1 is better match, output '1'. If the output2 is better match, output '0'. If they are same match, output '2'. 
    """

    prompt = template.format(reference=reference, output1=output1, output2=output2)


  # 将 prompt 作为输入传递给 GPT-4 模型进行生成
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
          {"role": "system", "content": "You are an excellent AI doctor."},
          {"role": "user", "content": prompt}
      ]
  ) 
    response_of_comparation = response.choices[0].message.content

    return response_of_comparation

input_file = '/home/willing/LLMDoctor-KG/dataset2_output/output_ranking_add3.5_final_remind.csv'
output_file = '/home/willing/LLMDoctor-KG/dataset2_output/output_gpt4_comparation_disease.csv'

with open(input_file,'r',newline="") as f_input, open(output_file, 'a+', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    header = next(reader)
    header.extend(["output2_winrate","output3_winrate","output4_winrate","output5_winrate","output6_winrate"])
    writer.writerow(header)

    for row in reader:
        output1_text = [row[2].strip("\n")]
        output2_text = [row[5].strip("\n")]
        output3_text = [row[6].strip("\n")]
        output4_text = [row[7].strip("\n")]
        output5_text = [row[8].strip("\n")]
        output6_text = [row[9].strip("\n")]


        references = [row[1].strip("\n")]

        flag = 0
        while flag == 0:
            try:
                response_of_comparation1 = prompt_comparation(references,output1_text,output2_text)
                try:
                    response_of_comparation1 = int(response_of_comparation1)
                    sleep(1)
                except:
                    response_of_comparation1 = prompt_comparation(references,output1_text,output2_text)
                    response_of_comparation1 = int(response_of_comparation1)

                

                response_of_comparation2 = prompt_comparation(references,output1_text,output3_text)
                try:
                    response_of_comparation2 = int(response_of_comparation2)
                    sleep(1)
                except:
                    response_of_comparation2 = prompt_comparation(references,output1_text,output3_text)
                    response_of_comparation2 = int(response_of_comparation2)

                    

                response_of_comparation3 = prompt_comparation(references,output1_text,output4_text)
                try:
                    response_of_comparation3 = int(response_of_comparation3)
                    sleep(1)
                except:
                    response_of_comparation3 = prompt_comparation(references,output1_text,output4_text)
                    response_of_comparation3 = int(response_of_comparation3)
                    
                    
                response_of_comparation4 = prompt_comparation(references,output1_text,output5_text)
                try:
                    response_of_comparation4 = int(response_of_comparation4)
                    sleep(1)
                except:
                    response_of_comparation4 = prompt_comparation(references,output1_text,output5_text)
                    response_of_comparation4 = int(response_of_comparation4)
                    

                response_of_comparation5 = prompt_comparation(references,output1_text,output6_text)
                try:
                    response_of_comparation5 = int(response_of_comparation5)
                    sleep(1)
                except:
                    response_of_comparation5 = prompt_comparation(references,output1_text,output6_text)
                    response_of_comparation5 = int(response_of_comparation5)
            

                flag = 1
            except:
                sleep(10)
                flag = 0


        print([response_of_comparation1,response_of_comparation2,response_of_comparation3,response_of_comparation4,response_of_comparation5])

        row.extend([response_of_comparation1,response_of_comparation2,response_of_comparation3,response_of_comparation4,response_of_comparation5])
        writer.writerow(row)


