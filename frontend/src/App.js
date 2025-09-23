import React, { useState } from "react";

function App() {
  const [input, setInput] = useState("");
  const [answer, setAnswer] = useState(null);

  const ask = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: "test",
          user_input: input,
        }),
      });

      if (!res.ok) throw new Error("Network response was not ok");
      const data = await res.json();
      setAnswer(data);
    } catch (error) {
      console.error("Error:", error);
      setAnswer({ error: "Request failed" });
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "flex-start",
        alignItems: "center",
        height: "100vh",
        overflowY: "auto",
        textAlign: "center",
        padding: "20px",
        backgroundColor: "#f9f9f9", // 연한 배경
      }}
    >
      {/* 제목 */}
      <h1 style={{ fontSize: "36px", fontWeight: "bold", marginBottom: "40px" }}>
        Economic News & Term Search
      </h1>

      {/* 검색창 */}
      <div style={{ display: "flex" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="검색어를 입력하세요..."
          style={{
            width: "400px",
            padding: "10px",
            fontSize: "16px",
            border: "1px solid #ccc",
            borderRadius: "24px 0 0 24px",
            outline: "none",
          }}
        />
        <button
          onClick={ask}
          style={{
            padding: "10px 20px",
            fontSize: "16px",
            border: "1px solid #ccc",
            borderLeft: "none",
            borderRadius: "0 24px 24px 0",
            backgroundColor: "#4285F4",
            color: "white",
            cursor: "pointer",
          }}
        >
          Search
        </button>
      </div>

      {/* 검색 결과 */}
      {answer && (
        <div
          style={{
            marginTop: "40px",
            maxWidth: "700px",
            width: "100%",
          }}
        >
          {/* 뉴스 요약 박스 */}
          {answer["뉴스 요약"] && (
            <div
              style={{
                backgroundColor: "white",
                padding: "20px",
                borderRadius: "12px",
                boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
                marginBottom: "20px",
                textAlign: "left",
              }}
            >
              <h2 style={{ fontSize: "22px", fontWeight: "bold" }}>뉴스 요약</h2>
              <pre style={{ whiteSpace: "pre-wrap", lineHeight: "1.6" }}>
                {answer["뉴스 요약"]}
              </pre>
            </div>
          )}

          {/* 관련 용어 박스 */}
          {answer["관련 용어"] &&
            Object.keys(answer["관련 용어"]).length > 0 && (
              <div
                style={{
                  backgroundColor: "white",
                  padding: "20px",
                  borderRadius: "12px",
                  boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
                  marginBottom: "20px",
                  textAlign: "left",
                }}
              >
                <h2 style={{ fontSize: "22px", fontWeight: "bold" }}>관련 용어</h2>
                <ul>
                  {Object.entries(answer["관련 용어"]).map(([term, explanation]) => (
                    <li key={term} style={{ marginBottom: "8px" }}>
                      <strong>{term}:</strong> {explanation}
                    </li>
                  ))}
                </ul>
              </div>
            )}

          {/* 출처 박스 */}
          {answer["출처"] && answer["출처"].length > 0 && (
            <div
              style={{
                backgroundColor: "white",
                padding: "20px",
                borderRadius: "12px",
                boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
                textAlign: "left",
              }}
            >
              <h2 style={{ fontSize: "22px", fontWeight: "bold" }}>출처</h2>
              <p>
                {answer["출처"].map((link, idx) => (
                  <a
                    key={idx}
                    href={link}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ marginRight: "10px" }}
                  >
                    [{idx + 1}]
                  </a>
                ))}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
