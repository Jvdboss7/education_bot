from config import *
import streamlit as st
from streamlit_chat import message
from edubot import EduBotCreator

@st.cache_resource(show_spinner=True)
def create_edubot():
    edubotcreator = EduBotCreator()
    edubot = edubotcreator.create_edubot()
    return edubot

edubot = create_edubot()

def main():
    st.title("Edubot: Your Smart Education Sidekick 📚🤖")
    st.subheader("A bot created using Langchain 🦜 to run on cpu making your learning process easier")

    user_input = st.text_input("Enter your query")

    


if __name__ == "__main__":
    pass