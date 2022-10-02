import os
from io import BufferedReader

import streamlit as st
from pydub import AudioSegment

from model_training_service import Code


pred = Code()


@st.cache
def process_prompt(completion_kwargs, topic):
    return pred.model_prediction(completion_kwargs=completion_kwargs, topic=topic)

def intro(api_key):
    # Setting up the Title
    st.title("Song transcriptor app powered by GPT-3 & Whisper")

    st.write("Get lyrics of your favorite band's song right now thanks to the text (GPT-3) and voice (Whisper) "
             "language models provided by the OpenAI research lab.")

    st.image("screen.jpg", use_column_width=True, caption='Image generated by Stable Diffustion algorithm.')

    st.write('''
    In this app you can:
    1. Find a song title based on hints you enter.
    2. Download audio from a selected video on youtube.
    3. Cut out a chunk of a chosen song whose text you want to get.
    4. Make a transcription of a chosen song (audio -> text).
    5. Ask about few interesting facts about chosen track.
    6. Translate lyrics of a chosen song into a language of your choice.
    7. Ask for a interpretation of the indicated song.
    ''')


    st.write("---")

    st.write(f"""
    ### Disclaimer
    Models executing indicated commands do not have access to Internet, they only use knowledge they have
    obtained from the data on which they learned to recognize which parts of the utterance co-occur
    with others. Therefore, the creator of this application does not take responsibility for any mistakes that these
    models may make. Even if these mistakes can be astonishing.
    """)

    st.write('''
    You can read about these models in the links provided:
    
    GPT-3:
    https://en.wikipedia.org/wiki/GPT-3
    
    Whisper:
    https://openai.com/blog/whisper/
    ''')

    st.write("---")

    st.info("Unfortunately, since the cloud on which the application is deployed does not have a GPU,"
            " GPU transcription is only possible when the application is run on a computer that has access to the GPU."
            " Only CPU transcription is possible on the cloud.")


def func1(api_key):
    with st.form("form1"):
            st.subheader("1. Find a song title based on the hint you enter.")

            song_hint = st.text_input('Provide some hint:',
                                      value='the song that was one of the first remixes in black rap, by areosmith',
                                      help='think that you are describing this song to someone who knows history of music,'
                                           ' or someone who knows the lyrics')
            engine = st.radio('Engine',
                ('text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))
            temp = st.slider('Temperature:', min_value=0., max_value=1., step=0.05)
            max_len = st.slider('Max tokens:', min_value=0, max_value=4096 if 'davinci' in engine else 2048,
                                step=8, value=64)
            if st.form_submit_button('Find the title'):
                completion_kwargs = {
                    "song_hint": (song_hint,),
                    "engine": engine,
                    "temperature": temp,
                    "max_tokens": max_len,
                    "api_key": api_key
                }
                st.write('**Name of the song**')
                st.write(f"""---""")
                with st.spinner(text='In progress'):
                    report_text = process_prompt(completion_kwargs, "song_hint")
                    st.markdown(report_text)
                    st.success('Done')


ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'ffmpeg-location': './',
    'outtmpl': "./%(id)s.%(ext)s",
}


def func2(api_key):
    st.subheader("2. Download audio from a selected video on youtube.")
    url = st.text_input('Provide a URL link to a youtube video:', value='https://youtu.be/bzUPG8olnO0')
    if st.button('Extract audio'):
        st.video(url)
        st.info('It may take few seconds.')
        with st.spinner(text='In progress'):
            import youtube_dl
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                meta = ydl.extract_info(url)
            newAudio = AudioSegment.from_mp3(f'{meta["id"]}.mp3')
            st.download_button(
                        label="Download extracted audio",
                        data=BufferedReader(newAudio.export(format="mp3")),
                        file_name=f'{meta["title"]}.mp3'
            )
        os.remove(f'{meta["id"]}.mp3')


def func3(api_key):
    st.subheader("3. Cut out a chunk of a chosen song whose text you want to get.")
    audio_file = st.file_uploader("Choose a song file in a mp3 format", type=['mp3'])
    t1 = st.number_input('Start second:', value=0)
    t2 = st.number_input('End second:', min_value=t1, value=t1)
    if st.button('Get the chunk'):
        audio_bytes = audio_file.getvalue()
        st.audio(audio_bytes, format='audio/mp3')

        if t1 < t2:
            st.info('It may take few seconds.')
            newAudio = AudioSegment.from_mp3(audio_file)
            newAudio = newAudio[t1 * 1000: t2 * 1000]
            st.download_button(
                label="Download chosen chunk of chosen song",
                data=BufferedReader(newAudio.export(format="mp3")),
                file_name='song_chunk.mp3',
            )
        else:
            st.warning("Enter seconds as int so that the first is less than the second.")


def func4(api_key):
    st.subheader("4. Make a transcription of a chosen song (audio -> text).")
    audio_file = st.file_uploader("Choose a song file in mp3 that you wanna transcript", type=['mp3'])
    st.info("GPU works only locally.")
    device = st.radio('Device', ('CPU', 'GPU'))
    model_size = st.radio('Whisper model size',('tiny', 'base', 'small', 'medium', 'large'),
                          help="Take into account that music may make it difficult to"
                               " transcribe lyrics of a song. Not all languages are"
                               " recognized to the same degree, so I advise you to"
                               " choose larger and larger models in case of failure.")
    if st.button('Transcribe'):
        newAudio = AudioSegment.from_mp3(audio_file)
        newAudio.export("song.mp3", format="mp3")

        with st.spinner(text='In progress'):
            st.write('Loading model and transcribing..')
            os.system(f'whisper song.mp3 --model {model_size} --fp16 {device=="GPU"} --device {device} > text.txt')
            os.remove('song.mp3')
            st.success('Done')

        # print the recognized text
        with open('text.txt', 'r') as f:
            text = f.read()
            text_download = '\n'.join(text.split('\n')[2:])
            text = '\n'.join(text.split('\n')[1:])
            text = text.replace('\n', '  \n')

        st.download_button(
            label="Download lyrics of the song",
            data=text_download,
            file_name='lyrics.txt',
            mime="text/plain"
        )
        os.remove('text.txt')
        os.remove('song.mp3.txt')
        os.remove('song.mp3.vtt')
        st.write(text)


def func5(api_key):
    with st.form("form5"):
        st.subheader("5. Ask about a few interesting facts about chosen track.")

        num = st.number_input('Provide a number of facts:', value=6, min_value=1)
        song_title = st.text_input('Provide the title of song:', value='Rock You Like A Hurricane by Scorpions')
        engine = st.radio('Engine (5)',
            ('text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))
        temp = st.slider('Temperature (5):', min_value=0., max_value=1., step=0.05, value=0.4)
        max_len = st.slider('Max tokens (5):', min_value=0, max_value=4096 if 'davinci' in engine else 2048,
                            step=8, value=512)
        if st.form_submit_button('Get facts'):
            completion_kwargs = {
                "facts": (num, song_title),
                "engine": engine,
                "temperature": temp,
                "max_tokens": max_len,
                "api_key": api_key
            }
            st.write('**Facts**')
            st.write(f"""---""")
            with st.spinner(text='In progress'):
                report_text = process_prompt(completion_kwargs, "facts")
                st.markdown(report_text)
                st.success('Done')

def func6(api_key):
    with st.form("form6"):
        st.subheader("6. Translate lyrics of a chosen song into the language of your choice.")

        lyrics = st.file_uploader(label='Provide lyrics to translation:', type=['.txt'])
        lang = st.text_input('Provide the language you want to translate into:', value='polish')
        engine = st.radio('Engine (6)',
            ('text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))
        temp = st.slider('Temperature (6):', min_value=0., max_value=1., step=0.05)
        max_len = st.slider('Max tokens (6):', min_value=0, max_value=4096 if 'davinci' in engine else 2048,
                            step=8, value=2048)
        if st.form_submit_button('Translate'):
            completion_kwargs = {
                "trans": (lang, lyrics.getvalue()),
                "engine": engine,
                "temperature": temp,
                "max_tokens": max_len,
                "api_key": api_key
            }
            st.write('**Translation**')
            st.write(f"""---""")
            with st.spinner(text='In progress'):
                report_text = process_prompt(completion_kwargs, "trans")
                report_text = report_text.replace(r'\n', '  \n')
                st.write(report_text)
                st.success('Done')

def func7(api_key):
    with st.form("form7"):
        st.subheader("7. Ask for interpretation of a indicated song.")

        lyrics = st.file_uploader(label='Provide lyrics to interpretation:', type=['.txt'])
        engine = st.radio('Engine (7)',
            ('text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))
        temp = st.slider('Temperature (7):', min_value=0., max_value=1., step=0.05, value=0.6)
        max_len = st.slider('Max tokens (7):', min_value=0, max_value=4096 if 'davinci' in engine else 2048,
                            step=8, value=2048)
        if st.form_submit_button('Interpret'):
            completion_kwargs = {
                "interp": (lyrics.getvalue(),),
                "engine": engine,
                "temperature": temp,
                "max_tokens": max_len,
                "api_key": api_key
            }
            st.write('**Interpretation**')
            st.write(f"""---""")
            with st.spinner(text='In progress'):
                report_text = process_prompt(completion_kwargs, "interp")
                st.markdown(report_text)
                st.success('Done')
