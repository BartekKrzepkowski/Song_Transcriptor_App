import streamlit as st

from streamlit_compose import intro, func1, func2, func3, func4, func5, func6, func7
PAGES = {
    'Dashboard': intro,
    'Find a song title': func1,
    'Extract audio from youtube video': func2,
    'Cut out audio': func3,
    'Make a transcription': func4,
    'Ask about few interesting facts': func5,
    'Translate lyrics': func6,
    'Ask for a interpretation': func7,
}


def main_app():
    # Creating an object of prediction service
    api_key = st.sidebar.text_input("OpenAI API Key:", type="password", value='XXX')

    with st.sidebar:
        selection = st.radio("", list(PAGES.keys()))

    if api_key:
        PAGES[selection](api_key)
    else:
        st.error("🔑 API Key Not Found!")
        st.info("💡 Copy paste your OpenAI API key that you can find in User -> API Keys section once you log in to the OpenAI API Playground")


if __name__ == '__main__':
    main_app()

