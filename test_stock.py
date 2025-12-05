from main import predict_stock_action

if __name__ == "__main__":
    code = input("종목 코드를 입력하세요 (예: 005930): ")
    result = predict_stock_action(code)
    print(result)
