import streamlit as st 
from dotenv import load_dotenv
from src.utility import process_uploaded_doc
import traceback
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utility import get_context

load_dotenv()

model=ChatOllama(
    model="ministral-3:8b",
    temperature=0.1
)

parser=StrOutputParser()

st.set_page_config(
    page_title="PDFusion:PDFs combined with AI-powered search",
    page_icon="📚"
)

st.header("PDFusion 📚")
st.write("your own question answering tool ")
st.header("upload your PDF to begin with conversation:")

uploaded_pdf=st.file_uploader("upload pdf",type=['pdf'])

if uploaded_pdf is not None:
    st.success("pdf uploaded successfully ✅")
    try:
        with st.spinner(
            text="processing in progress....",
            show_time=True,
            width="content"
        ):
            process_uploaded_doc(uploaded_pdf)
        st.success("processing successful ✅")
    except Exception as e:
        st.error(f"X error {e}")
        st.code(traceback.format_exc())

    if "chat_started" not in st.session_state:
        st.session_state.chat_started=False

    if st.button("start conversation"):
        st.session_state.chat_started=True

    if st.session_state.chat_started:
        st.write("Ask you question....")
        user_que=st.chat_input("Message:")
        if user_que :
            with st.spinner(text="generating response please wait.......",show_time=True):

                    docs=get_context(user_que)
                    context_text="\n\n".join(doc.page_content for doc in docs)
                    prompt_template=PromptTemplate(
                        template="""

                    You are a helpful assistant. Based on the given context and query,
        answer the question appropriately.

        Use only the provided context. If the answer is not available
        in the context, say that you cannot find the answer in the provided context.

        ------ Context ------
        {context}

        ------ Query ------
        {query}
        """,
        input_variables=['context','query']
                    )
                    chain=prompt_template | model | parser
                    res=chain.invoke({'query':user_que,'context':context_text})
                    st.markdown(res)