import streamlit as st
import openai

key = "unique-key"

# Get a GPT-3 API key from OpenAI
openai.api_key = "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"


def get_email_response(text):

    response = openai.Completion.create(

        engine="davinci",

        max_tokens=500,

        temperature=0.7,

        top_p=0.9,

        n=5,

        stream=text,

        stop="\n",

        logprobs=None,

        frequency_penalty=0.2,

        presence_penalty=0.6

    )

    return response

def main():

    st.title("Email Response Generator")

    text = st.text_area("Enter email text")

    if st.button("Submit"):

        response = get_email_response(text)

        st.write(response.choices[0]['text'])

if __name__ == '__main__':

    main()
