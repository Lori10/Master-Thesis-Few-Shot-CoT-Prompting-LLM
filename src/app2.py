import streamlit as st
from langchain.llms import OpenAI
from test_paperqa_tool import answer
from PIL import Image

def main():
    st.title("Streamlit Application")
    
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
        user_question = col1.text_input('Enter the question', '')
    with col2:
        submit_button = col2.button("Submit")
    
    with st.container():
        # Create a container with three parallel columns
        col1, col2, col3 = st.columns([3, 2.5, 3])
        

        # Add content to each column
        with col1:
            st.markdown("<h2 style='font-size: 20px;'>Show step</h2>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h2 style='font-size: 20px;'>AI Output</h2>", unsafe_allow_html=True)
            if submit_button:
                #entire_output = answer(question=user_question)
                entire_output = 'ENTIRE OUTPUT'
                col2_content = entire_output
                col2.write(col2_content)

        with col3:
            st.markdown("<h2 style='font-size: 20px;'>Rate this Explanation!</h2>", unsafe_allow_html=True)
            st.write("Content for column 3")

        if col1.button("Step 1\nClick here to show step 1!"):
            col2.write("Output of step 1!")

        if col1.button("Step 2\nClick here to show step 2!"):
            col2.write("Output of step 2!")

        if col1.button("Step 3\nClick here to show step 3!"):
            col2.write("Output of step 3!")


if __name__ == "__main__":
    main()
