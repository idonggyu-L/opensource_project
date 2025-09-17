import React, { useState } from "react";

function App() {
  const [input, setInput] = useState("");
  const [answer, setAnswer] = useState(null);

  const ask = async () => {
    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: "test",   // 임시 세션 ID
          user_input: input,    // 사용자가 입력한 질문
        }),
      });

      if (!res.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await res.json();
      setAnswer(data); // 결과 저장
    } catch (error) {
      console.error("Error:", error);
      setAnswer({ error: "요청 실패" });
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>경제 뉴스 검색</h1>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="질문 입력..."
      />
      <button onClick={ask}>검색</button>

      {answer && (
        <div style={{ marginTop: "20px" }}>
          <h2>응답 결과</h2>
          <pre>{JSON.stringify(answer, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
