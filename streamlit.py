import streamlit as st 
from helpers import *
from langgraph.graph import END, StateGraph, START




st.set_page_config(page_title = "BlackRock AI Agent", page_icon = "public/blackrock_logo.png", layout = "wide")
st.image("public/blackRock_logo.png", width = 250)
# https://cdn.prod.website-files.com/617960145ff34f911afe7243/662aa1fb10438a4e4fda0dc8_AI%20pick%20the%20stock.jpg

# https://img.freepik.com/free-photo/futuristic-robot-interacting-with-money_23-2151612697.jpg?t=st=1732675577~exp=1732679177~hmac=523677211117face180cf4f628f38003a3f95c2bc1a55761aebb24cd0a1fc665&w=1800


st.markdown(f'<p style="background-color:black;color:white;font-size:16px;border-radius:0%;">DISCLAIMER: This application is built as a POC for the technical interview of BlackRock by Vansh Jaiswal </p>', unsafe_allow_html=True)
# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://investingnews.com/media-library/robotic-hand-and-human-hand-reaching-out-to-touch-glowing-brain.jpg?id=51482476&width=1200&height=800&quality=80&coordinates=0%2C0%2C0%2C1");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
    background-opacity: 0.5;
}
</style>
"""



st.markdown(background_image, unsafe_allow_html=True)



with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.text("")
        st.text("")
        original_title = '<h1 style="text-align: centre; font-family: serif; color:white; font-size: 32px;">BlackRock Multi-Agent AI</h1>'
        st.markdown(original_title, unsafe_allow_html=True)
    with col3:
        pass
    

    st.text("")
    st.text("")
    col1, col2, col3 = st.columns((0.05, 0.9, 0.05))
    with col1:
        pass
    with col2:
        prompt = st.text_input("Please Enter Your Input", placeholder="Enter the query")
        button = st.button("Submit")
        if button:
            workflow = StateGraph(GraphState)
            # Define the nodes
            workflow.add_node("wiki_search", wiki_search)  # web search
            workflow.add_node("retrieve", retrieve)  # retrieve

            # Build graph
            workflow.add_conditional_edges(
                START,
                route_question,
                {
                    "wiki_search": "wiki_search",
                    "vectorstore": "retrieve",
                },
            )
            workflow.add_edge( "retrieve", END)
            workflow.add_edge( "wiki_search", END)
            # Compile
            app = workflow.compile()

            # Run
            inputs = {
                "question": prompt
            }
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Node
                    pprint(f"Node '{key}':")
                    # Optional: print full state at each node
                    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                pprint("\n---\n")

            # Final generation
            # pprint(value)
            # print("**********************************", value)
            # st.info(value)
            value=str(value["documents"].content)

            st.markdown(f'<p style="background-color:black;color:white;font-size:16px;border-radius:0%;">{value}</p>', unsafe_allow_html=True)
    with col3:
        pass

    








    

    # input_style = """
    # <style>
    # input[type="text"] {
    #     background-color: transparent;
    #     color: #a19eae;  // This changes the text color inside the input box
    # }
    # div[data-baseweb="base-input"] {
    #     background-color: transparent !important;
    # }
    # [data-testid="stAppViewContainer"] {
    #     background-color: transparent !important;
    # }
    # </style>
    # """
    # st.markdown(input_style, unsafe_allow_html=True)







