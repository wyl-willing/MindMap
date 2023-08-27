import pandas as pd 

df_database = pd.read_csv("./disease_database_mini.csv") 

for index, row in df_database.iterrows():
    with open("./train.txt","a") as f:   
        for symptom in row["Symptom"].strip('[]').split(","):
            f.write(row["disease"].replace(" ","_"))
            f.write("\t")
            f.write("has_symptom")
            f.write("\t")
            f.write(symptom.strip(" ''").replace(" ","_"))
            f.write("\n")

            f.write(symptom.strip(" ''").replace(" ","_"))
            f.write("\t")
            f.write("possible_disease")
            f.write("\t")
            f.write(row["disease"].replace(" ","_"))
            f.write("\n")

        for medical_test in row["Medical Tests"].strip('[]').split(","):
            f.write(row["disease"].replace(" ","_"))
            f.write("\t")
            f.write("need_medical_test")
            f.write("\t")
            f.write(medical_test.strip(" ''").replace(" ","_"))
            f.write("\n")

            f.write(medical_test.strip(" ''").replace(" ","_"))
            f.write("\t")
            f.write("can_check_disease")
            f.write("\t")
            f.write(row["disease"].replace(" ","_"))
            f.write("\n")

        for medication in row["Medications"].strip('[]').split(","):
            f.write(row["disease"].replace(" ","_"))
            f.write("\t")
            f.write("need_medication")
            f.write("\t")
            f.write(medication.strip(" ''").replace(" ","_"))
            f.write("\n")

            f.write(medication.strip(" ''").replace(" ","_"))
            f.write("\t")
            f.write("possible_cure_disease")
            f.write("\t")
            f.write(row["disease"].replace(" ","_"))
            f.write("\n")
f.close()

