import streamlit as st
import numpy as np

text = st.text_input("Put text here")
x = st.slider("Pick a number", 0, 10)
y = st.slider("Pick a number", 0, 5)

if st.button("Click me"):
    st.info(f"Message: {text}")

    with st.status("Loading..") as status:
        for i in range(1_000_000):
            print(i)
    
    
    with st.chat_message(f"user"):
        st.write(f"The number is {x} and its type is {type(x)}")
        st.line_chart(np.random.randn(30, 3))


    