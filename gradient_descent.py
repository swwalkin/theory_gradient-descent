# 기본설정
import random
###################################################################################
# 변수
X = random.randint(-10, 10)

# 하이퍼파라미터
learning_rate = 0.1

# 함수
func_len = 3
coeff = [random.randint(-10, 10) for _ in range(func_len)]; coeff[-1] = abs(coeff[-1]) if abs(coeff[-1]) > 0 else 1
func = lambda x: sum(coeff[i] * x**i for i in range(func_len))
func_grad = lambda x: sum(i * coeff[i] * x**(i-1) for i in range(func_len))
###################################################################################
# 경사하강법
temp = func(X)
while True:
    grad = func_grad(X)
    X -= grad * learning_rate

    if abs(func(X) - temp) < 0.001:
        break
    temp = func(X)
#########################################
# 결과 출력
print(" ".join(map(str, reversed(coeff))))
print("경사하강법")
print(f"({X:.2f}, {func(X):.2f})")
# 실제값 계산
print("실제값")
print(f"({-coeff[1]/coeff[2]/2:.2f}, {func(-coeff[1]/coeff[2]/2):.2f})")
