import os 
import os.path
import sys
import pandas as pd 

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

# global variable 
agent = None

def prompt():
    global agent
    cmd = "init"
    while(cmd != ""):
        cmd = input("prompt[`enter` for exit]:")
        if cmd == "":
            break 
        agent.run(cmd)
        
def select_data():
    import glob
    
    global agent
    path = "./data/*.csv"
    file_list = glob.glob(path)
    for idx, filename in enumerate(file_list):
        print(f"{idx}:{filename}")
    num = input(f"select data [0-{len(file_list)-1}] or `enter` for exit: ")
    try:
        data_idx = int(num)
    except:
        print (f"thank you for using.")
        sys.exit(0)
    
    data_path = file_list[data_idx]
    print(f"using {data_path}")
    agent = create_csv_agent(OpenAI(temperature=0), data_path, verbose=True)
    print("== agent info ==")
    print(agent)
    print("================")
    #print (">> 프롬프트 템플릿(prompt template)")
    #print(agent.agent.llm_chain.prompt.template)
    agent.run("데이터에 몇 열이 있어?")
    
    prompt()
    

# 여기에 openai api key를 넣습니다. 
os.environ["OPENAI_API_KEY"] = ""


# 근린공원과 소공원의 분포는?
# 운동시설의 분포는?

# 먼지 농도와 습도와의 상관관계는?
# 미세먼지가 가장 심했던 날과 지역은?
# 압축 천연가스를 가장 많이 보유한 회사는?

# 면적대비 시가표준액이 낮은 상위 1%는?
# 과세년도 2021년에 시가총액이 가장높은 것은?

while True:
    select_data()
    

