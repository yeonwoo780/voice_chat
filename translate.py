import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
# API KEY 정보로드
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def _llm_translate(question: str, language: str):
    llm = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06")

    SystemPrompt = """You are a helpful assistant in translation.
    translate language list only ["korean", "japanese", "english", "arabic"]
    you only answer output. don't show input
    Below is an example.

    korean translate Example:
    - input: generative model은 주어진 training data를 학습하고 training data의 분포를 따르는 유사 data를 생성하는 모델입니다.
    - output: 생성 모델은 주어진 학습 데이터를 학습하고 학습 데이터의 분포를 따르는 유사 데이터를 생성하는 모델입니다.

    japanese translate Example:
    - input: generative model은 주어진 training data를 학습하고 training data의 분포를 따르는 유사 data를 생성하는 모델입니다.
    - output: 生成モデルは、与えられたトレーニングデータを学習し、トレーニングデータの分布に従う類似データを生成するモデルです。

    english translate Example:
    - input: generative model은 주어진 training data를 학습하고 training data의 분포를 따르는 유사 data를 생성하는 모델입니다.
    - output: A generative model is a model that learns given training data and generates similar data that follows the distribution of the training data.

    arabic translate Example:
    - input: generative model은 주어진 training data를 학습하고 training data의 분포를 따르는 유사 data를 생성하는 모델입니다.
    - output: النموذج التوليدي هو نموذج يتعلم بيانات التدريب المعطاة ويولد بيانات مماثلة تتبع توزيع بيانات التدريب.

    Now, look at the Question and Language and translate the Question into Language.

    Question: {question}
    Language: {language}
    """
    prompt = ChatPromptTemplate.from_template(
        SystemPrompt
    )

    chain = (
        {"question": RunnablePassthrough(), "language": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"question":question, "language": language})