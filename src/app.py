import streamlit as st
from langchain.llms import OpenAI
#from test_update_paperqa_tool import update_answer
from PIL import Image
import pickle
import re



def main():
    st.title("Few-shot learning for question answering with LLM")


    # Add content to the first section/column
    with st.container():
        st.markdown("<style>h1 { margin-top: -50px; }</style>", unsafe_allow_html=True)  # Reduce margin for h1 tag
        col1, col2, col3 = st.columns([7, 7, 15])
        
        with col1:
            st.markdown("<h2 style='font-size: 20px;'>Potential incorrectness</h2>", unsafe_allow_html=True)
            st.write("Please be aware that myGPT can occasionally generate incorrect information.")
    
        with col2:
            st.markdown("<h2 style='font-size: 20px;'>Limited knowledge after 2021</h2>", unsafe_allow_html=True)
            st.write("Please be aware that myGPT has limited knowledge after 2021.")

        with col3:
            st.markdown("<h2 style='font-size: 20px;'>Instructions</h2>", unsafe_allow_html=True)
            st.write("1. Upload a dataset containing questions and answers.")
            st.write("2. Enter the test question.")
            st.write("3. Click the button Submit to generate the response .")
       
    with st.container():
        col1, col2, = st.columns([7, 7])
        with col1:
            st.markdown("<h2 style='font-size: 20px;'>Step 1: Upload the data</h2>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("")        
        with col2:
            st.markdown("<h2 style='font-size: 20px;'>Step 2: Enter test question</h2>", unsafe_allow_html=True)
            question = st.text_input('', '')
            submit_question = col2.button("Submit Question", key='submit_question')

    
if __name__ == "__main__":
    main()
