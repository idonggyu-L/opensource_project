import React, { useState } from "react";

function App() {
  const [input, setInput] = useState("");
  const [answer, setAnswer] = useState(null);

  const [stockCode, setStockCode] = useState("");
  const [stockResult, setStockResult] = useState(null);

  // 뉴스 검색
  const ask = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: input }),
      });
      if (!res.ok) throw new Error("Network response was not ok");
      const data = await res.json();
      setAnswer(data);
    } catch (error) {
      console.error("Error:", error);
      setAnswer({ error: "Request failed" });
    }
  };

  // 주가 예측
  const predictStock = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/predict_stock", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker_code: stockCode }),
      });
      if (!res.ok) throw new Error("Network response was not ok");
      const data = await res.json();
      setStockResult(data);
    } catch (error) {
      console.error("Error:", error);
      setStockResult({ error: "Request failed" });
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "20px",
        backgroundColor: "#f9f9f9",
        minHeight: "100vh",
      }}
    >
      {/* 제목 */}
      <h1 style={{ fontSize: "36px", fontWeight: "bold", marginBottom: "40px" }}>
        Economic News & Stock Prediction
      </h1>

      {/* 뉴스 검색 */}
      <div style={{ display: "flex", marginBottom: "20px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="뉴스 검색어를 입력하세요..."
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

      {/* 뉴스 결과 */}
      {answer && (
        <div style={{ marginTop: "20px", maxWidth: "700px", width: "100%" }}>
      
          {/* 뉴스 요약 */}
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

          {/* 관련 용어 */}
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

          {/* 출처 */}
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

          {/* 주가 예측: 뉴스 검색 결과가 있을 때만 */}
          <div
            style={{
              marginTop: "40px",
              maxWidth: "700px",
              width: "100%",
              textAlign: "center",
            }}
          >
            <div style={{ display: "flex", justifyContent: "center", marginBottom: "20px" }}>
              <input
                type="text"
                value={stockCode}
                onChange={(e) => setStockCode(e.target.value)}
                placeholder="종목 코드 입력 (예: 005930)"
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
                onClick={predictStock}
                style={{
                  padding: "10px 20px",
                  fontSize: "16px",
                  border: "1px solid #ccc",
                  borderLeft: "none",
                  borderRadius: "0 24px 24px 0",
                  backgroundColor: "#34A853",
                  color: "white",
                  cursor: "pointer",
                }}
              >
                Predict
              </button>
            </div>

            {stockResult && (
              <div
                style={{
                  backgroundColor: "white",
                  padding: "20px",
                  borderRadius: "12px",
                  boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
                  textAlign: "left",
                  margin: "0 auto",
                }}
              >
                <h2 style={{ fontSize: "22px", fontWeight: "bold" }}>주가 예측</h2>
                {stockResult.error ? (
                  <p style={{ color: "red" }}>{stockResult.error}</p>
                ) : (
                  <>
                    <p><strong>종목:</strong> {stockResult.종목}</p>
                    <p><strong>예측:</strong> {stockResult.예측}</p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
