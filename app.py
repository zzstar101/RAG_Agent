import time

import streamlit as st
from agent.react_agent import ReactAgent

TYPEWRITER_DELAY_SECONDS = 0.003

st.title("智扫通智能客服")
st.divider()

typewriter_enabled = st.sidebar.toggle(
    "打字机效果",
    value=False,
    help="关闭后可显著降低长回答渲染延迟",
)

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
    
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])
    
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    response_messages = []
    with st.spinner("智能客服正在思考..."):
        res_stream = st.session_state["agent"].excute_stream(prompt)
        
        def capture(generator, cache_list, enable_typewriter: bool):
            previous_full_text = ""
            for chunk in generator:
                if not chunk:
                    continue

                chunk_text = str(chunk)
                if chunk_text.startswith(previous_full_text):
                    delta_text = chunk_text[len(previous_full_text):]
                    previous_full_text = chunk_text
                else:
                    delta_text = chunk_text
                    previous_full_text += chunk_text

                cache_list.append(delta_text)
                
                if enable_typewriter:
                    for char in delta_text:
                        time.sleep(TYPEWRITER_DELAY_SECONDS)
                        yield char
                else:
                    yield delta_text
                
        st.chat_message("assistant").write(capture(res_stream, response_messages, typewriter_enabled))
        full_response = "".join(response_messages).strip()
        if full_response:
            st.session_state["messages"].append({"role": "assistant", "content": full_response})
        st.rerun()