from langchain.memory import ConversationBufferMemory

# 세션별 memory 저장소
session_memories = {}

def get_memory(session_id: str):
    """세션별 memory 반환 (없으면 새로 생성)"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return session_memories[session_id]
