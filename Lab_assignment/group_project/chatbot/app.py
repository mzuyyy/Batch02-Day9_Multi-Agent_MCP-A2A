import streamlit as st
import sys
from pathlib import Path

# Thêm thư mục src vào sys.path để import các module RAG cá nhân
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from Lab_assignment.src.task10_generation import generate_with_citation

st.set_page_config(page_title="RAG Chatbot - Pháp luật Ma Tuý", page_icon="⚖️", layout="wide")

st.title(" Chatbot Tư Vấn Pháp Luật Phòng Chống Ma Tuý")
st.markdown("Hệ thống RAG Pipeline tìm kiếm và giải đáp dựa trên dữ liệu pháp luật và tin tức thực tế.")

# Sidebar Settings
st.sidebar.title("⚙️ Cài đặt hệ thống")
use_hyde = st.sidebar.checkbox("Bật HyDE (Bonus)", value=False, help="Sử dụng LLM để tạo ra tài liệu giả định, cải thiện kết quả tìm kiếm ngữ nghĩa.")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Form nhập tin nhắn
if prompt := st.chat_input("Nhập câu hỏi của bạn (ví dụ: Hình phạt tàng trữ ma tuý là gì?)"):
    # Lưu tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot trả lời
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Đang suy nghĩ và tìm kiếm tài liệu...")
        
        try:
            # Lấy lịch sử nhưng bỏ qua câu prompt vừa thêm vào
            history = st.session_state.messages[:-1]
            
            # Gọi RAG pipeline (Task 10)
            result = generate_with_citation(prompt, top_k=5)
            
            answer = result["answer"]
            sources = result["sources"]
            retrieval_source = result.get("retrieval_source", "unknown")
            
            # Hiển thị câu trả lời
            message_placeholder.markdown(answer)
            
            # Hiển thị sources
            if sources:
                with st.expander(f"📚 Xem các tài liệu tham khảo ({len(sources)} sources - {retrieval_source})"):
                    for i, doc in enumerate(sources, 1):
                        meta = doc.get("metadata", {})
                        source_name = meta.get("source", f"Tài liệu {i}")
                        st.markdown(f"**[{i}] {source_name}** (Score: {doc.get('score', 0):.3f})")
                        st.caption(doc["content"][:200] + "...")
                        
        except Exception as e:
            answer = f"⚠️ Đã có lỗi xảy ra: {str(e)}\n\nVui lòng kiểm tra lại cấu hình API key."
            message_placeholder.error(answer)

        # Lưu tin nhắn bot
        st.session_state.messages.append({"role": "assistant", "content": answer})
