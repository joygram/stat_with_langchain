# chatGPT를 만난 통계 분석

chatGPT를 연계한 솔류션은 여러가지가 있습니다. 어떤 것이 우리에게 도움이 될지 부딪혀보며 경혐해봐야 하는 시기가 아닌가 생각해봅니다.

chatGPT를 통해 우리가 깨닫고 있는 것은

- 전문적인 IT지식이 없이도 사용할 수 있다.
- 지식의 확산 속도가 매우 빨라서 도움받을 수 있는 부분이 많다.
- IT도구의 경계를 따지지 않는다. R이냐 Python이냐가 중요하지 않다.

과거에 IT기술 진입점에서 이탈하여 본연의 목표를 이루지 못했다면 chatGPT의 기술 덕분에 본연의 목표에 가까이 다가갈 수 있습니다.

본 강의에서는 가장 기본적으로 python으로 패키지를 설치하고 간단한 문법정도를 이해하고 있다는 가정하에 진행합니다. 하지만 그런 지식이 없어도 전혀 걱정하지 마세요.

chatGPT를 통해 도움을 얻어 python설치 및 환경 구성이 가능 할테니까요.

## 프롬프트 기술

ai가 효과적으로 정보처리를 하게 돕는 것이야말로 미래사회의 인간이 해야할 역할이라고 봐도 과언이 아닐 것입니다.

여러 강연에서 많이 다루어 졌지만 다시한번 강조해도 부족하지 않을 것이 `프롬프트` 기술입니다.

chatGPT에게 올바르게 문제를 해석하고 결과를 만들어낼 수 있도록 나 자신이 원하는 목표를 명확하기 정의하는 것이 필요합니다.

인간의 입장에서는 이말이나 저말이나 비슷해도 chatGPT의 ai입장에서는 처리가 곤란할 수 있기 때문입니다. 좀 더 미래에는 더 인식을 잘해겠지만 ...

## langchain

langchain과 python을 통해 데이터 분석을 위한 pandas등을 전혀 모르는 상태에서 자연어로 요구사항에 대한 답을 얻어내는 것을 시도해보겠습니다.

순서는 다음과 같습니다.

- openAI api키 발급
- langchain 라이브러리 설치
- 샘플 코드를 확장하여 excel/csv데이터를 활용한 통계 분석시도

## openAI api키 발급받기

chatGPT기술을 직접사용하기 위해서는 openAI의 api를 반드시 얻어야 합니다. 이와 비슷하게 ai기술의 도움을 받기위해서 비슷한 패턴으로 api를 받아야 할 수 있기 때문에 방식을 알아두면 여러모로 편리합니다.

openAI의 api 키를 발급받는 방법을 chatGPT에 물어본 결과입니다.  아래의 절차대로 회원가입을 하고 키를 생성하면 됩니다. 키를 잊어도 걱정하지 마세요 다시 발급하면 되니까요.


`프롬프트: OpenAI API Key를 발급받으려면?`
```
OpenAI API Key를 발급받으려면 OpenAI 웹사이트에서 API 액세스를 신청해야 합니다. 
API 액세스를 신청하려면 OpenAI 계정이 필요합니다.

API 액세스를 신청하려면 아래와 같은 단계를 따르면 됩니다.

- OpenAI 웹사이트 (https://openai.com/)에 접속합니다.

- 홈페이지 상단의 "Get started for free" 버튼을 클릭합니다.

- "Create an account" 버튼을 클릭하여 OpenAI 계정을 생성합니다. 계정을 생성하는 과정에서 요구되는 정보를 입력합니다.

- 계정 생성이 완료되면, "API" 메뉴에서 "API Documentation"을 선택합니다.

- "API Access" 섹션에서 "Request Access" 버튼을 클릭합니다.

- API 액세스 신청서를 작성하고, 제출합니다.

- API 액세스를 신청하면 일정 시간 이내에 OpenAI 측에서 신청서를 검토하고 승인 여부를 알려줍니다. 승인이 되면 Access Key와 Secret Key가 발급됩니다.

Access Key와 Secret Key는 API 요청을 보낼 때 사용되는 인증 정보로, 발급된 키를 사용하여 API에 액세스할 수 있습니다. 
발급된 키는 안전하게 보관되어야 하며, 다른 사람에게 노출되지 않도록 주의해야 합니다.
```

## langchain 라이브러리

langchain은 chatGPT와 wolfram Alpha(울프람 알파)를 결합하여 보다 정확하고 자세한 답변을 얻는 방법입니다.

- langchain은 AI에게 Wolfram Alpha(계산적 사고)와 chatGPT(언어적 사고)로 작업하는 방법을 가르치는 시도를 합니다.

- 사용자의 질문을 판단하여 Alpha나 GPT에 API를 호출합니다.

- langchain은 수학 물리등 복잡한 처리를 보다 잘 해낼 수 있습니다.

오늘 제공하는 실습을 위해서

- 파이썬3.10 이상을 설치합니다. 

- langchain, openai, pandas라이브러리를 설치합니다.

```

pip install langchain openai pdandas
```

너무 걱정하지 마세요 `파이썬 코딩`을 직접 수행하지는 않을 것이니까요. 

## langchain 간단 설명

langchain에서 핵심적으로 사용하는 개념을 크게 두가지 입니다. 그외 자세한 것은 

https://github.com/hwchase17/langchain 을 참고하도록 합니다. 

### LLMs (Large Language Models)

langchain에는 LLMs를 위해 ` 프롬프트` 관리, 최적화, 일반적인 인터페이스를 통합한 유틸리티를 제공합니다.

### Agent

agent는 어떤 행동을 할지 결정하여 행동하고 관찰합니다. 그리고 완료될 때까지 반복하는 LLM을 포함합니다.

## 구현해보기 

csv기반의 데이터 분석을 위해서 필요한 코드는 다음과 같습니다. 


```python
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-발급받은키"

data_path = './data/train.csv'
agent = create_csv_agent(OpenAI(temperature=0), data_path, verbose=True)
agent.run("데이터에 몇 열이 있어?")
```

실제 파이썬 코딩 스킬을 사용했다고 보기 어려울 정도입니다.

물론, 가장 기초적인 지식은 있으면 좋겠지요^^


## 실습해보기

기본 환경이 설정된 이후에는 편안하게 원하는 바를 타이핑하면 됩니다.

준비한 간단한 데이터셋을 통해 프롬프팅 해보겠습니다. 



## 마무리하며 
얼핏보면 chatGPT에 바로 프롬프팅 하는 것과 큰 차이가 없어보일 수 있습니다. 

좀 들여다보면 

- ai에게 문제를 풀수 있도록 다양한 시도를 하는 점 
- 코드 생성의 전반도 시도해볼 수 있는점 
- 데이터와 직접 바인딩하여 해당 데이터를 코딩하지 않고 바로 처리하는 점 

등은 설명하지 않아도 어떻게 활용할지에 대해 상상하실 수 있을 것입니다. 

익숙하지 않은 학습 허들을 지나서 본연의 목표에 집중하는 것 

그것이 ai시대의 방향성이 아닌가 싶습니다. 
