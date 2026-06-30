import torch
# Задание 1.1: Создание тензоров

print("=== Задание 1.1: Создание тензоров ===")
t_rand = torch.rand(3, 4)
print("Случайный тензор 3x4:\n", t_rand)
t_zeros = torch.zeros(2, 3, 4)
print("\nТензор из нулей 2x3x4:\n", t_zeros)
t_ones = torch.ones(5, 5)
print("\nТензор из единиц 5x5:\n", t_ones)
t_arange = torch.arange(16).reshape(4, 4)
print("\nТензор 4x4 (0-15):\n", t_arange)

# Задание 1.2: Операции с тензорами
print("\n=== Задание 1.2: Операции с тензорами ===")
A = torch.rand(3, 4)
B = torch.rand(4, 3)

# Транспонирование тензора A
A_t = A.t()  # или A.T
print("Размер транспонированного A:", A_t.shape)

# Матричное умножение A и B
matmul_res = torch.matmul(A, B)
print("Результат матричного умножения A @ B:\n", matmul_res)

# Поэлементное умножение A и транспонированного B
elementwise_res = A * B.t()
print("Результат поэлементного умножения A * B.T:\n", elementwise_res)

# Вычислите сумму всех элементов тензора A
sum_A = A.sum()
print("Сумма всех элементов тензора A:", sum_A.item())

# Задание 1.3: Индексация и срезы
print("\n=== Задание 1.3: Индексация и срезы ===")
C = torch.arange(125).reshape(5, 5, 5)
first_row = C[0, 0, :]
print("Первая строка первого среза:\n", first_row)
last_col = C[0, :, -1]
print("\nПоследний столбец первого среза:\n", last_col)

center_submatrix = C[2, 1:3, 1:3]
print("\nЦентральная подматрица 2x2 (из центрального среза):\n", center_submatrix)

# Все элементы с четными индексами
even_indices = C[::2, ::2, ::2]
print("\nЭлементы с четными индексами (шаг 2):\n", even_indices)

# Задание 1.4: Работа с формами
print("\n=== Задание 1.4: Работа с формами ===")
D = torch.arange(24)
print("Исходная форма D:", D.shape)

print("Форма 2x12:", D.reshape(2, 12).shape)
print("Форма 3x8:", D.reshape(3, 8).shape)
print("Форма 4x6:", D.reshape(4, 6).shape)
print("Форма 2x3x4:", D.reshape(2, 3, 4).shape)
print("Форма 2x2x2x3:", D.reshape(2, 2, 2, 3).shape)