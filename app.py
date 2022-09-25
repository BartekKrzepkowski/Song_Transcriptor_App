import streamlit as st
from model_training_service import Code

def main_app():
        
    # Creating an object of prediction service
    pred = Code()

    api_key = st.sidebar.text_input("OpenAI API Key:", type="password")

    # Using the streamlit cache 
    @st.cache
    def process_prompt(completion_kwargs, topic):
        return pred.model_prediction(completion_kwargs=completion_kwargs, topic=topic)

    if api_key:
            
        # Setting up the Title
        st.title("Song transcriptor app powered by GPT-3 & Whisper")

        st.write("Get the lyrics of your favorite band's song right now thanks to the text and voice "
                 "language models provided by the OpenAI research lab.")

        st.image("song.jpg", use_column_width=True)

        st.write('''
        In this app you can:
        1. Find the song title based on the hint you enter.
        2. Cut out a chunk of a chosen song whose text you want to get.
        3. Make a transcription of a chosen song (audio -> text).
        4. Ask about few interesting facts about chosen track.
        5. Translate the lyrics of a chosen song into the language of your choice.
        6. Ask for interpretation of the indicated song.
        ''')
        
        st.write("---")

        st.write(f"""
        ### Disclaimer
        Models executing the indicated commands do not have access to the Internet, they only use the knowledge they have
        obtained from the data on which they learned to recognize which parts of the utterance coexist
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

        st.write(f"""---""")

        st.subheader("1. Find the song title based on the hint you enter.")

        song_hint = st.text_input('Provide some hint:',
                                  value='the song that was one of the first remixes in black rap, by areosmith',
                                  help='think that you are describing this song to someone who knows music,'
                                       ' or someone who knows the lyrics')
        engine = st.radio('Engine',
            ('text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))
        temp = st.slider('Temperature:', min_value=0., max_value=1., step=0.05)
        max_len = st.slider('Max tokens:', min_value=0, max_value=4096 if 'davinci' in engine else 2048,
                            step=8, value=64)
        if st.button('Find the title'):
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

        st.write(f"""---""")

        st.subheader("2. Cut out a chunk of a chosen song whose text you want to get.")
        audio_file = st.file_uploader("Choose a song file in a mp3 format", type=['mp3'])
        t1 = st.text_input('Start second:')
        t2 = st.text_input('End second:')
        if st.button('Get a chunk'):
            audio_bytes = audio_file.getvalue()
            st.audio(audio_bytes, format='audio/mp3')

            if t1 != None and t2 != None:
                from pydub import AudioSegment
                from io import BufferedReader
                newAudio = AudioSegment.from_mp3(audio_file)
                newAudio = newAudio[int(t1)*1000:int(t2)*1000]
                st.download_button(
                    label="Download chosen chunk of chosen song:",
                    data=BufferedReader(newAudio.export(format="mp3")),
                    file_name='song_chunk.mp3',
                )
        st.write(f"""---""")

        st.subheader("3. Make a transcription of a chosen song (audio -> text).")
        audio_file = st.file_uploader("Choose a song file in mp3 that you wanna trascript", type=['mp3'])
        model_size = st.radio('Whisper model size',
            ('tiny', 'base', 'small', 'medium', 'large'))
        if st.button('Transcribe'):
            # audio_bytes = audio_file.getvalue()
            import whisper
            model = whisper.load_model(model_size)
            audio = whisper.load_audio("song_chunk.mp3")
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)
            print(f"Detected language: {max(probs, key=probs.get)}")

            # decode the audio
            options = whisper.DecodingOptions()
            with st.spinner(text='In progress'):
                result = whisper.decode(model, mel, options)
                st.success('Done')

            # print the recognized text
            st.markdown(result.text)

        st.write(f"""---""")

        st.subheader("4. Ask about a few interesting facts about chosen track.")

        num = st.text_input('Provide a number of facts:', value=6)
        song_title = st.text_input('Provide the title of song:', value='Rock You Like A Hurricane by Scorpions')
        engine = st.radio('Engine4',
            ('text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))
        temp = st.slider('Temperature4:', min_value=0., max_value=1., step=0.05, value=0.4)
        max_len = st.slider('Max tokens4:', min_value=0, max_value=4096 if 'davinci' in engine else 2048,
                            step=8, value=512)
        if st.button('Get facts'):
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

        st.write(f"""---""")

        st.subheader("5. Translate the lyrics of a chosen song into the language of your choice.")

        lyrics = st.text_input('Provide the lyrics to translation:')
        lang = st.text_input('Provide the language you want to translate into:', value='german')
        engine = st.radio('Engine (5)',
            ('text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))
        temp = st.slider('Temperature (5):', min_value=0., max_value=1., step=0.05)
        max_len = st.slider('Max tokens (5):', min_value=0, max_value=4096 if 'davinci' in engine else 2048,
                            step=8, value=2048)
        if st.button('Translate'):
            completion_kwargs = {
                "trans": (lang, lyrics),
                "engine": engine,
                "temperature": temp,
                "max_tokens": max_len,
                "api_key": api_key
            }
            st.write('**Translation**')
            st.write(f"""---""")
            with st.spinner(text='In progress'):
                report_text = process_prompt(completion_kwargs, "trans")
                st.markdown(report_text)
                st.success('Done')

        st.write(f"""---""")

        st.subheader("6. Ask for interpretation of the indicated song.")

        lyrics = st.text_input('Provide the lyrics to interpretation:')
        engine = st.radio('Engine (6)',
            ('text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))
        temp = st.slider('Temperature (6):', min_value=0., max_value=1., step=0.05, value=0.6)
        max_len = st.slider('Max tokens (6):', min_value=0, max_value=4096 if 'davinci' in engine else 2048,
                            step=8, value=2048)
        if st.button('Interpret'):
            completion_kwargs = {
                "interp": (lyrics,),
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

    else:
        st.error("🔑 API Key Not Found!")
        st.info("💡 Copy paste your OpenAI API key that you can find in User -> API Keys section once you log in to the OpenAI API Playground")


if __name__ == '__main__':
    main_app()
