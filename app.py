%%writefile app.py

import streamlit as st
import fitz
import re
import pandas as pd

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

st.set_page_config(page_title="PDF RAG 챗봇", layout="wide")
st.title("📄 보험 PDF 하이브리드 RAG 챗봇")


uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"])


# ======================
# 보험료 추출
# ======================
def extract_premium_info(pdf_doc):
    data = []

    for page in pdf_doc:
        text = page.get_text()
        matches = re.findall(r'(\d{1,3}(?:,\d{3})+|\d+)원', text)

        for m in matches:
            data.append({
                "value": int(m.replace(",", "")),
                "raw": m + "원",
                "context": text[:200]
            })

    return pd.DataFrame(data)


if uploaded_file is not None:

    with st.spinner("PDF 분석 중..."):

        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

        df = extract_premium_info(pdf)

        text = ""
        for page in pdf:
            text += page.get_text() + "\n\n"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("PDF 분석 완료")


    # ======================
    # 사용자 입력 UI (중요 변경)
    # ======================
    query = st.text_input("질문을 입력하세요")


    ask = st.button("질문하기")


    # ======================
    # 정답 추출 함수
    # ======================
    def find_exact_premium(df, query):
        if df is None or df.empty:
            return None

        if ("암보험" in query or "보험" in query) and "보험료" in query:
            return df["value"].iloc[0]

        return None


    # ======================
    # 실행 로직
    # ======================
    if ask and query:

        exact = find_exact_premium(df, query)

        if exact:
            st.subheader("📌 문서 기반 정답")
            st.success(f"{exact:,}원")

        else:
            docs = vectorstore.similarity_search(query, k=5)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = ChatPromptTemplate.from_template("""
문서 기반으로만 답하세요.

[문서]
{context}

질문: {question}

답변:
""")

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            chain = prompt | llm | StrOutputParser()

            answer = chain.invoke({
                "context": context,
                "question": query
            })

            st.subheader("📄 AI 답변")
            st.write(answer)


        with st.expander("참고 문서"):
            docs = vectorstore.similarity_search(query, k=5)
            for i, doc in enumerate(docs, 1):
                st.markdown(f"### 문서 {i}")
                st.write(doc.page_content)