import os 
import pandas as pd 

print(">> set api key")
# 여기에 openai api key를 넣습니다. 
os.environ["OPENAI_API_KEY"] = ""

print (">> read tain data 'train.csv'")
df = pd.read_csv("./data/train.csv")

print (">> head of data frame")
print(df.head())



print (">> run langchain agent ")
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

agent = create_csv_agent(OpenAI(temperature=0), './data/train.csv', verbose=True)

print(agent) 

print (">> 프롬프트 템플릿(prompt template)")
print(agent.agent.llm_chain.prompt.template)


agent.run("데이터에 몇 열이 있어?")


agent.run("여자는 몇명이나 있구매횟수의 분포가 가장 높은 여자들의 분포는?지?")

agent.run("현재 도시에서 3년 넘게 지내고 있는  사람은 몇명?")

agent.run("현재 도시에서 3년 넘게 지내고 있는  사람 중 여자는 몇명?")

agent.run("구매횟수의 분포가 가장 높은 여자들의 분포는?")

agent.run("구매횟수의 분포가 가장 높은 여자들의 분포를 그래프로 그리면?")


prompt = "init"
while(prompt != ""):
    prompt = input("chat:")
    agent.run(prompt)
    

