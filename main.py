import streamlit as st
import os
import base64
import io
from dotenv import load_dotenv
from PIL import Image
import PyPDF2
import google.generativeai as genai
from openai import OpenAI
import cohere

load_dotenv()

st.set_page_config(
    page_title="AI Document Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    div[data-testid="stChatInput"] {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 0rem;
        padding-right: 0rem;
    }

    div[data-testid="stChatInput"] textarea {
        background-color: #f0f2f6;
        border-radius: 10px;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ===== HÀM ĐỌC API KEY (MỚI) =====
def get_api_key(key_name):
    """Đọc API key từ Streamlit secrets (cloud) hoặc .env (local)"""
    try:
        # Ưu tiên đọc từ Streamlit secrets khi deploy
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    # Nếu không có, đọc từ .env khi chạy local
    return os.getenv(key_name)


def get_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return None


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def chat_gemini(api_key, prompt, context_data, is_image=False):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        if is_image:
            response = model.generate_content([prompt, context_data])
        else:
            response = model.generate_content(f"Context:\n{context_data}\n\nQuestion: {prompt}")
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


def chat_gpt(api_key, prompt, context_data, is_image=False):
    client = OpenAI(api_key=api_key)
    messages = []
    if is_image:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{context_data}"}}
            ]
        }]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant analyzing documents."},
            {"role": "user", "content": f"Document content:\n{context_data}\n\nQuestion: {prompt}"}
        ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def chat_perplexity(api_key, prompt, context_data, is_image=False):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.perplexity.ai"
    )

    if is_image:
        return "Perplexity API hiện chưa hỗ trợ phân tích ảnh. Vui lòng chọn model khác hoặc upload file PDF."

    messages = [
        {"role": "system", "content": "You are a helpful assistant analyzing documents."},
        {"role": "user", "content": f"Document content:\n{context_data}\n\nQuestion: {prompt}"}
    ]

    try:
        response = client.chat.completions.create(
            model="sonar",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def chat_cohere(api_key, prompt, context_data, is_image=False):
    co = cohere.ClientV2(api_key=api_key)

    if is_image:
        return "Cohere API không hỗ trợ phân tích ảnh. Vui lòng chọn model khác hoặc upload file PDF."

    try:
        full_message = f"Document content:\n{context_data}\n\nQuestion: {prompt}"

        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "user",
                    "content": full_message
                }
            ]
        )

        return response.message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"


with st.sidebar:
    st.title("Cấu Hình & Tài Liệu")

    model_choice = st.selectbox(
        "Chọn AI Model",
        [
            "Google Gemini",
            "OpenAI GPT-4o",
            "Perplexity Sonar",
            "Cohere Command-A"
        ],
        index=0
    )

    # ===== ĐỌC API KEYS (ĐÃ SỬA) =====
    gemini_key = get_api_key("GOOGLE_API_KEY")
    openai_key = get_api_key("OPENAI_API_KEY")
    perplexity_key = get_api_key("PERPLEXITY_API_KEY")
    cohere_key = get_api_key("COHERE_API_KEY")

    # Kiểm tra API key
    if model_choice == "Google Gemini":
        if gemini_key:
            st.success("Gemini Key đã tải")
        else:
            st.error("Thiếu GOOGLE_API_KEY")
    elif model_choice == "OpenAI GPT-4o":
        if openai_key:
            st.success("OpenAI Key đã tải")
        else:
            st.error("Thiếu OPENAI_API_KEY")
    elif model_choice == "Perplexity Sonar":
        if perplexity_key:
            st.success("Perplexity Key đã tải")
        else:
            st.error("Thiếu PERPLEXITY_API_KEY")
        st.info("Perplexity chỉ hỗ trợ file PDF")
    else:  # Cohere
        if cohere_key:
            st.success("Cohere Key đã tải")
        else:
            st.error("Thiếu COHERE_API_KEY")
        st.info("Cohere chỉ hỗ trợ file PDF")

    st.divider()

    uploaded_file = st.file_uploader("Tải lên tài liệu", type=['pdf', 'jpg', 'png', 'jpeg'])

    if uploaded_file:
        file_type = uploaded_file.type
        if "image" in file_type:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh xem trước", use_container_width=True)
            if model_choice in ["Perplexity Sonar", "Cohere Command-A"]:
                st.warning("Model này không hỗ trợ ảnh")
        elif "pdf" in file_type:
            st.info(f"File PDF: {uploaded_file.name}")

    st.divider()
    if st.button("Xóa lịch sử chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

st.title("Phân Tích Tài Liệu")
st.caption("Upload tài liệu bên trái và bắt đầu hỏi.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    if not uploaded_file:
        st.toast("Vui lòng upload tài liệu trước!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang phân tích..."):
                response_text = ""
                file_type = uploaded_file.type

                # Chọn API key tương ứng
                if model_choice == "Google Gemini":
                    active_key = gemini_key
                elif model_choice == "OpenAI GPT-4o":
                    active_key = openai_key
                elif model_choice == "Perplexity Sonar":
                    active_key = perplexity_key
                else:  # Cohere
                    active_key = cohere_key

                if not active_key:
                    response_text = "Lỗi: Không tìm thấy API Key"
                else:
                    if "image" in file_type:
                        img_obj = Image.open(uploaded_file)

                        if model_choice == "Google Gemini":
                            response_text = chat_gemini(active_key, prompt, img_obj, is_image=True)
                        elif model_choice == "OpenAI GPT-4o":
                            base64_img = image_to_base64(img_obj)
                            response_text = chat_gpt(active_key, prompt, base64_img, is_image=True)
                        elif model_choice == "Perplexity Sonar":
                            response_text = chat_perplexity(active_key, prompt, None, is_image=True)
                        else:  # Cohere
                            response_text = chat_cohere(active_key, prompt, None, is_image=True)

                    elif "pdf" in file_type:
                        pdf_txt = get_pdf_text(uploaded_file)
                        if pdf_txt:
                            if model_choice == "Google Gemini":
                                response_text = chat_gemini(active_key, prompt, pdf_txt, is_image=False)
                            elif model_choice == "OpenAI GPT-4o":
                                response_text = chat_gpt(active_key, prompt, pdf_txt, is_image=False)
                            elif model_choice == "Perplexity Sonar":
                                response_text = chat_perplexity(active_key, prompt, pdf_txt, is_image=False)
                            else:  # Cohere
                                response_text = chat_cohere(active_key, prompt, pdf_txt, is_image=False)
                        else:
                            response_text = "Không đọc được text từ PDF (có thể là file scan)."

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
