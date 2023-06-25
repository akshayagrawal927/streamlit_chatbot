import openai
import streamlit as st
from streamlit_chat import message
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Setting page title and header
st.set_page_config(page_title="Collector", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Dataframe explorer chatbot</h1>", unsafe_allow_html=True)

# Pass openai API key here
openai.api_key = "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"


# Upload file here
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    schema = df.dtypes.to_string()
    query = None

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'Graph' not in st.session_state:
    st.session_state['Graph'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

#function for evaluation the prompt output
def create_plot_from_response(response_text):
    [eval(statement) for statement in response_text.split("\n")]

# generate a response
def generate_response(query):
    st.session_state['messages'].append({"role": "user", "content": query})
    prompt = f"""You are a matplotlib and seaborn expert.
    You answer questions related to data
    You also write code for creating visualization charts based on input query.
    Image name is always 'plot.png'
    You have a pandas DataFrame called df that contains data with the following schema:
    {schema}

    Use appropriate figure size
    Do not add imports
    Do not return anything but code
    If plot is required then follow these steps
        To create charts you return in the similar format:
        plt.figure(figsize=(n,n))
        sns.<plot_kind>()
        plt.xlabel(<categories>)
        plt.ylabel(<values>)
        plt.title(<suitable-chart-name>)
        plt.savefig(<image-name>)
    Else give the answer to the query in code

    Query: """

    end_prompt = "\nResult: "

    full_prompt = prompt + query + end_prompt

    completion_params = {"temperature": 0.7, "max_tokens": 250, "model": "text-davinci-003"}

    completion = openai.Completion.create(prompt=full_prompt, **completion_params)

    global response

    response = completion["choices"][0]["text"].strip(" \n")

    if 'plt' in response:
        create_plot_from_response(response)
        st.session_state['Graph'].append(Image.open(r"C:\Users\Msi\Desktop\streamlit\plot.png"))
        response = "Chart is shown below"
    else:
        st.session_state['Graph'].append("no image")
        response = response
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=70)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            #img = Image.open(r"C:\Users\akshayagrawal\Desktop\streamlit-chatgpt-ui\plot.png")
            try:
                st.image(st.session_state['Graph'][i],width =500)
            except:
                pass
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
