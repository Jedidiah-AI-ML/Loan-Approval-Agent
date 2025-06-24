#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install openai')


# ## Building an ai agent to answer questios based on loan approvals 

# In[2]:


from openai import OpenAI


# In[3]:


client = OpenAI(api_key='your api key')


# In[4]:


import pandas as pd


# In[5]:


df = pd.read_csv("../ai agent/trainn.csv")


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


# creating a function to summarize the data
def create_data_summary(df):
    summary = f"The dataset has {df.shape[0]} columns.\n"
    summary += "Columns:\n"
    for col in df.columns:
        summary += f"-{col} (type: {df[col].dtype})\n"
    return summary


# In[9]:


# this summary helps the aget unnderstand the structure without actually loading the entire data into prompt(which will exceed toke limits)


# ### **Buildig the AI agent function**

# lets define the agent that will hadle user queries ased on the data summary

# In[10]:


def ai_agent(user_query, df):
    data_context = create_data_summary(df)

    prompt = f"""
You are a data expert AI agent.

You have ee provided with this dataset summary:
{data_context}

Now, based on the user's question:
'{user_query}'

Think step-by-step. Assume you ca access ad analyze the dataset like an Expert Data Scientist would using Pandas.

Give a clear, final answer. And do not hallucinate.
    """

    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [{"role":"user", "content": prompt}],
        temperature = 0.2,
        max_tokens = 500
    )

    answer = response.choices[0].message.content
    return answer


# We defied a function ai_agent that takes a user query and the dataset, summarizes the dataset structure, and creates a prompt combining both the context and the question. This prompt is then sent to OpemAI's GPT=4o model using the client.chat.completions.create() method, and the model's step-by-step,natural-language response is returned to the user.
# 
# Now let's create an interactive loop where users can ask questions to the Ai Agent:

# In[ ]:


print("Welcome to Loan Review AI Agent!")
print("You can ask aything about the loan applicants data.")
print("Type 'exit' to quit.")

while True:
    user_input= input("\nYour Question:")
    if user_input.lower() == "exit":
        break
    response = ai_agent(user_input, df)
    print("\nAI Agent Response:")
    print(response)


# we just created a simple iteractive loop that continuously prompts the user to ask questions. when the user inputs a query, it is passed to the ai_agent function which processes it and returns a natural language answer based on the sataset, if the user types "exit", the loop breaks and the program ends

# In[ ]:




