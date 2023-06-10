
import os
from apikey import cohere_apikey

import streamlit as st
# from langchain.llms import OpenAI
# from langchain.llms import GPT4All
 
from langchain.llms import Cohere
# from langchain import PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['COHERE_API_KEY'] = cohere_apikey

# Prompt
st.title("🦜️🔗 LangChain Banu")
prompt = st.text_input("Wasssssuuuppppp Mate?")

# Prompt Templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template="Write me a youtube video script based on this title TITLE: {title} \
    while leveraging this wikipedia research: {wikipedia_research}"
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs
llm = Cohere(cohere_api_key=cohere_apikey)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
# sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True) # order matters here. 
# sequential_chain = SequentialChain(chains=[title_chain, script_chain], 
#                                    input_variables=['topic'], 
#                                    output_variables=['title','script'], 
#                                    verbose=True) # order matters here. 


wiki = WikipediaAPIWrapper()

if prompt:
    # response = sequential_chain.run(prompt)
    # response = sequential_chain({'topic':prompt})

    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)


    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wiki research'):
        st.info(wiki_research)
















# template = """Question: {question}

# Answer: Let's think step by step."""



# prompt = PromptTemplate(template=template, input_variables=["question"])






# llm_chain = LLMChain(prompt=prompt, llm=llm)
# question = input()#"What NFL team won the Super Bowl in the year Justin Beiber was born?"
# # lm_chain.run(question)

# # # print(f"ENV: {os.environ['OPENAI_API_KEY']}")

# # # # App framework
# # # # LLM
# # # llm = OpenAI(model_name='text-davinci-003', temperature=0.9)

# # if prompt:
# #     response = llm_chain.run(question)
# #     st.write(response)