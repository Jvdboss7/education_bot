from edubot import EduBotCreator
from config import *
import streamlit as st
from streamlit_chat import message

@st.cache_resource(show_spinner=True)
def create_edubot():
    """
    The function creates an instance of EduBot using the EduBotCreator class.
    :return: an instance of the EduBot class, which is created by the EduBotCreator class.
    """
    edubotcreator = EduBotCreator()
    edubot = edubotcreator.create_edubot()
    return edubot
edubot = create_edubot()

def infer_edubot(prompt):
    """
    The function `infer_edubot` takes a prompt as input, uses the `edubot` model to generate a response,
    and returns the generated answer.
    
    :param prompt: The prompt is the input given to the model, which is used to generate a response. It
    can be a question, statement, or any text that you want the model to generate a response for
    :return: The answer returned by the `infer_edubot` function.
    """
    model_out = edubot(prompt)
    answer = model_out['result']
    return answer

def display_conversation(history):
    """
    The function `display_conversation` takes a conversation history as input and displays the user and
    assistant messages in the conversation.
    
    :param history: The `history` parameter is a dictionary that contains the conversation history. It
    has two keys: "user" and "assistant". The value of each key is a list of messages exchanged between
    the user and the assistant
    """
    for i in range(len(history["assistant"])):
        message(history["user"][i], is_user=True, key=str(i) + "_user")
        message(history["assistant"][i],key=str(i))

def main():
    """
    The `main()` function creates a chatbot interface using Streamlit and Langchain to provide answers
    to user queries.
    """

    st.title("Edubot: Your Smart Education Sidekick 📚🤖")
    st.subheader("A bot created using Langchain 🦜 to run on cpu making your learning process easier")

    user_input = st.text_input("Enter your query")

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ["I am ready to help you"]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hey there!"]
                
    if st.button("Answer"):
        answer = infer_edubot({'query': user_input})
        st.session_state["user"].append(user_input)
        st.session_state["assistant"].append(answer)

        if st.session_state["assistant"]:
            display_conversation(st.session_state)

if __name__ == "__main__":
    main()