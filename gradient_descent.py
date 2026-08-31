# 기본설정
import random
###################################################################################
# 변수

# 하이퍼 파라미터
learning_rate = 0.1

# 함수
coeff = [random.randint(-10, 10) for _ in range(3)]; coeff[-1] = abs(coeff[-1]) if coeff[-1] != 0 else 1
func = lambda x: sum(coeff[i] * x**i for i in range(3))
func_grad = lambda x: sum(i * coeff[i] * x**(i-1) for i in range(3))
X = random.randint(-10, 10)
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
txt = f"{coeff[2]}x^2"
if coeff[1] > 0:
    txt += f" + {coeff[1]}x"
elif coeff[1] < 0:
    txt += f" - {abs(coeff[1])}x"
if coeff[2] > 0:
    txt += f" + {coeff[2]}"
elif coeff[2] < 0:
    txt += f" - {abs(coeff[2])}"

print(txt)
print("경사하강법")
print(f"({X:.2f}, {func(X):.2f})")
print("실제값")
print(f"({-coeff[1]/coeff[2]/2:.2f}, {func(-coeff[1]/coeff[2]/2):.2f})")
