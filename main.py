import os

import streamlit as st
import yaml
from langchain.memory import StreamlitChatMessageHistory
from numpy import transpose
from streamlit_mic_recorder import mic_recorder

from audio_handler import transcribe_audio
from llm_chains import load_normal_chain
from utils import get_timestamp, load_chat_history_json, save_chat_history_json

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_chain(chat_history):
    return load_normal_chain(chat_history)


def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""


def set_send_input():
    st.session_state.send_input = True
    clear_input_field()


def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key


def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            # this makes the new name of session the timestamp
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(
                st.session_state.history,
                config["chat_history_path"] + st.session_state.new_session_key,
            )
        else:
            save_chat_history_json(
                st.session_state.history,
                config["chat_history_path"] + st.session_state.session_key,
            )


def main():
    st.title("Multi Modal local chat app")

    # just a div in html we can access later on
    chat_container = st.container()
    st.sidebar.title("chat sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])
    # print(chat_sessions)

    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"

    if (
        st.session_state.session_key == "new_session"
        and st.session_state.new_session_key != None
    ):
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox(
        "Select a chat sesssion",
        chat_sessions,
        key="session_key",
        index=index,
        on_change=track_index,
    )

    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(
            config["chat_history_path"] + st.session_state.session_key
        )
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input(
        "type msg here", key="user_input", on_change=set_send_input
    )

    # voice recording
    voice_recording_column, send_button_column = st.columns(2)
    with voice_recording_column:
        voice_recording = mic_recorder(
            start_prompt="Start recording", stop_prompt="Stop recording", just_once=True
        )

    with send_button_column:
        send_button = st.button("send", key="send_button", on_click=clear_input_field)

        # uplaod and transcribe messages
    uploaded_audio = st.file_uploader(
        "upload an audio file", type=["wav", "mp3", "ogg"]
    )
    if uploaded_audio:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain.run(transcribed_audio)

    # print(voice_recording)
    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        # print(transcribed_audio)
        llm_chain.run(transcribed_audio)

    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            # because user_input becomes user_question b4 it gets here
            llm_response = llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ""

        if chat_history.messages != []:
            with chat_container:
                st.write("Chat history")
                for message in chat_history.messages:
                    st.chat_message(message.type).write(message.content)

        # print(chat_history.messages[0].dict())

        save_chat_history()


if __name__ == "__main__":
    main()
