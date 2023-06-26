import streamlit as st
from langchain.llms import OpenAI
#from test_update_paperqa_tool import update_answer
from PIL import Image
import pickle
import re


def update_answer(question, steps_feedback=None):
    print(f'RUNNING update_answer. Steps Feedback: {steps_feedback}')
    if steps_feedback:
        return 'AI ANSWER - ' + steps_feedback, 'smth'
    else:
        return 'AI ANSWER - ' + question, 'smth'

def process_llm_output(llm_output):
    cleaned_entire_output = re.sub(r'\([^)]*\)', '', llm_output)
    return '<br>'.join(cleaned_entire_output.split('\n'))

def main():
    st.title("Streamlit Application")
    
    if 'submit_question' not in st.session_state:
        st.session_state.disabled = True

    # Add content to the first section/column
    with st.container():
        col1, col2 = st.columns(2)
        st.markdown("<style>h1 { margin-top: -50px; }</style>", unsafe_allow_html=True)  # Reduce margin for h1 tag
        with col1:
            st.title("My Streamlit App")
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.markdown("<h2 style='font-size: 20px;'>Potential incorrectness</h2>", unsafe_allow_html=True)
                st.write("Please be aware that myGPT can occasionally generate incorrect information.")
        
            with subcol2:
                st.markdown("<h2 style='font-size: 20px;'>Limited knowledge after 2021</h2>", unsafe_allow_html=True)
                st.write("Please be aware that myGPT has limited knowledge after 2021.")

        with col2:
            image_url = "agent.jpg"  # Replace with the URL of your image
            image = Image.open(image_url)

            # Reduce the width and height of the image
            image_width = 300
            image_height = 200

            # Resize the image using PIL
            resized_image = image.resize((image_width, image_height))

            # Display the resized image using st.image
            st.image(resized_image)
    
    # Add a text input field and button
    container = st.container()
    # Add text input and button in parallel
    col1, col2 = container.columns([2, 1])
    col2.markdown(
        """
        <style>
        .stButton > button {
            margin-top: 10px;  /* Adjust top margin */
            margin-down: 20px; /* Adjust left margin */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with col1:
        user_question = st.text_input('Enter the question', '')

        # with st.form(key="form_question"):
        #     user_question = st.text_input('Enter the question', '')
        #     submit_question = st.form_submit_button(label="Submit Question")
                
    with col2:
        submit_question = col2.button("Submit Question", key='submit_question')
        if submit_question:
            st.session_state.disabled = False
    with st.container():
        # Create a container with three parallel columns
        col1, col2, col3, col4, col5 = st.columns([4, 7, 7, 7, 6])
        
        # Add content to each column
        with col1:
            st.markdown("<h2 style='font-size: 20px;'>Show step</h2>", unsafe_allow_html=True)
            show_step1 = col1.button("Click here to show step 1!", key='step1', disabled=st.session_state.disabled)
            show_step2 = col1.button("Click here to show step 2!", key='step2', disabled=st.session_state.disabled)

        with col2:
            st.markdown("<h2 style='font-size: 20px;'>Step</h2>", unsafe_allow_html=True)

        with col3:
            st.markdown("<h2 style='font-size: 20px;'>AI Output</h2>", unsafe_allow_html=True)
        with col4:
            st.markdown("<h2 style='font-size: 20px;'>AI Output with Feedback</h2>", unsafe_allow_html=True)
            
            #col2.write(process_llm_output(entire_output), unsafe_allow_html=True)
        
        with col5:
            st.markdown("<h2 style='font-size: 20px;'>Rate this Explanation!</h2>", unsafe_allow_html=True)
            
            #     # st.write('Suggest a better answer for Step 2!')
            #     user_answer_step2 = st.text_input('Answer Feedback', '')
            #     submit_feedback = st.form_submit_button(label="Submit Feedback")

            submit_feedback = col5.button("Submit Feedback", key='submit_feedback', disabled=st.session_state.disabled)
            feedback_text = col5.text_input('Feedback Answer', '', disabled=st.session_state.disabled)

            #no_submit_feedback = col5.button("No Feedback/Reset", key='none', disabled=st.session_state.disabled)

    if submit_feedback:
        st.session_state.disabled = True

    # if no_submit_feedback:
    #     st.session_state.disabled = True

    if show_step1:
        st.session_state.disabled = True

    if show_step2:
        st.session_state.disabled = True

    
    if submit_question:
        ai_answer, steps_output = update_answer(question=user_question)
        col3.write(ai_answer)
        with open("ai_answer.txt", "w") as file:
            file.write(ai_answer)

    if "upload1_state" not in st.session_state:
        st.session_state.upload1_state = False

    if (show_step1 or st.session_state.upload1_state):
        st.session_state.upload1_state = True

        print('show step 1')
        col5.write('Is Step 1 reasonable? (1 to 5)') 
        col2.write('This is Output of STEP 1', unsafe_allow_html=True)
        col5.write('This is Output of STEP 1', unsafe_allow_html=True)
        selected_option_1 = col5.radio("Rate Step 1", [1, 2, 3, 4, 5], horizontal=True)
        col5.write(selected_option_1)
        col5.write('Suggest a better answer for Step 1!')
        #feedback_step1 = col5.text_input('Feedback 1', '')
        feedback_step2 = None 
        st.session_state.disabled = False

        st.session_state.upload1_state=False
        
    if "upload2_state" not in st.session_state:
        st.session_state.upload2_state = False

    if (show_step2 or st.session_state.upload2_state):
        st.session_state.upload2_state = True
        
        print('show step 2')
        col5.write('Is Step 2 reasonable? (1 to 5)') 
        col2.write('This is Output of STEP 2', unsafe_allow_html=True)
        col5.write('This is Output of STEP 2', unsafe_allow_html=True)
        selected_option_2 = col5.radio("Rate Step 2", [1, 2, 3, 4, 5], horizontal=True)
        col5.write(selected_option_2)
        col5.write('Suggest a better answer for Step 2!')
        #feedback_step2 = col5.text_input('Feedback 2', '')
        feedback_step1 = None
        st.session_state.disabled = False

        st.session_state.upload2_state=False

    # if no_submit_feedback:
    #     st.session_state.upload1_state=False
    #     st.session_state.upload2_state=False
    #     with open("ai_answer.txt", "r") as file:
    #         ai_response= file.read().strip()
    #         col3.write(ai_response)

    if submit_feedback:
        st.session_state.upload1_state=False
        st.session_state.upload2_state=False
        # selected_steps = {key: value for key, value in steps_output.items() if key in [1,2]}
        # selected_steps[2]['subanswer'] = user_answer_step2
        # feedback_answer, feedback_steps_output = update_answer(question=user_question, steps_feedback=selected_steps)
        # col3.write(process_llm_output(feedback_answer), unsafe_allow_html=True)
        with open("ai_answer.txt", "r") as file:
            ai_response= file.read().strip()
            col3.write(ai_response)

        # if feedback_step1:
        #     feedback_answer, feedback_steps_output = update_answer(question=user_question, steps_feedback=feedback_step1)
        # elif feedback_step2:
        #     feedback_answer, feedback_steps_output = update_answer(question=user_question, steps_feedback=feedback_step2)
        if feedback_text:
            feedback_answer, feedback_steps_output = update_answer(question=user_question, steps_feedback=feedback_text)
        col4.write(feedback_answer)        

        st.session_state.disabled = False
if __name__ == "__main__":
    main()
