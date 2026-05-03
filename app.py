import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import chat

st.set_page_config(
    page_title="HR Talent Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 HR Talent Search Agent")
st.caption("Find the right candidates using AI-powered resume search")

# sidebar
with st.sidebar:
    st.header("Search Options")
    st.markdown("**Available Categories:**")
    categories = [
        "INFORMATION-TECHNOLOGY", "BUSINESS-DEVELOPMENT", "FINANCE",
        "ADVOCATE", "ACCOUNTANT", "ENGINEERING", "CHEF", "AVIATION",
        "FITNESS", "SALES", "BANKING", "HEALTHCARE", "CONSULTANT",
        "CONSTRUCTION", "PUBLIC-RELATIONS", "HR", "DESIGNER",
        "ARTS", "TEACHER", "APPAREL", "DIGITAL-MEDIA",
        "AGRICULTURE", "AUTOMOBILE", "BPO"
    ]
    for cat in categories:
        st.markdown(f"• {cat}")
    
    st.divider()
    st.markdown("**Session Stats**")
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    st.metric("Total Tokens Used", st.session_state.total_tokens)

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# tampilkan chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "tool_used" in message:
            st.caption(f"Tool used: `{message['tool_used']}`")
        if "tokens" in message:
            st.caption(f"Tokens: {message['tokens']}")

# input user
if prompt := st.chat_input("Search for candidates... e.g. 'find data scientist with Python skills'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching candidates..."):
          st.session_state.history.append(HumanMessage(content=prompt))
          result = graph.invoke({"messages": st.session_state.history})
          st.session_state.history = result["messages"]

          # Ambil response terakhir
          last_message = result["messages"][-1]
          response = last_message.content

          # Deteksi tool yang dipakai
          tool_used = None
          tokens = None
          for msg in result["messages"]:
              if hasattr(msg, 'tool_calls') and msg.tool_calls:
                  tool_used = msg.tool_calls[0]['name']
              if hasattr (msg, 'response_metadata'):
                  usage = msg.response_metadata.get('token_usage', {})
                  if usage:
                      tokens = usage.get('total_tokens', 0)
                      st.session_state.total_tokens += tokens

        st.markdown(response)
        if tool_used:
            st.caption(f"Tool used: `{tool_used}`")
        if tokens:
            st.caption(f"Tokens this response: {tokens}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "tool_used": tool_used,
        "tokens": tokens
    })