import torch

# Задание 2.1: Простые вычисления с градиентами
print("Задание 2.1: Простые вычисления с градиентами")

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)
f = x**2 + y**2 + z**2 + 2*x*y*z

# Вычисляем градиенты (обратный проход)
f.backward()

print(f"Значение функции f(x, y, z) = {f.item()}")
print(f"Градиент df/dx: {x.grad.item()}")
print(f"Градиент df/dy: {y.grad.item()}")
print(f"Градиент df/dz: {z.grad.item()}")

grad_x_analyt = 2 * x.item() + 2 * y.item() * z.item()
grad_y_analyt = 2 * y.item() + 2 * x.item() * z.item()
grad_z_analyt = 2 * z.item() + 2 * x.item() * y.item()

print(f"Аналитический df/dx: {grad_x_analyt} (проверка: {x.grad.item() == grad_x_analyt})")
print(f"Аналитический df/dy: {grad_y_analyt} (проверка: {y.grad.item() == grad_y_analyt})")
print(f"Аналитический df/dz: {grad_z_analyt} (проверка: {z.grad.item() == grad_z_analyt})")

# Задание 2.2: Градиент функции потерь
print("\nЗадание 2.2: Градиент функции потерь ")
x_data = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([3.0, 5.0, 7.0])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# Линейная функция
y_pred = w * x_data + b

mse_loss = torch.mean((y_pred - y_true)**2)

mse_loss.backward()

print(f"Значение функции потерь MSE: {mse_loss.item():.4f}")
print(f"Градиент по весу w: {w.grad.item():.4f}")
print(f"Градиент по смещению b: {b.grad.item():.4f}")

# Задание 2.3: Цепное правило
print("\nЗадание 2.3: Цепное правило")

# f(x) = sin(x^2 + 1)
x_val = torch.tensor(1.5, requires_grad=True)
f_val = torch.sin(x_val**2 + 1)

grad_pytorch = torch.autograd.grad(f_val, x_val)[0]

grad_analytical = torch.cos(x_val**2 + 1) * 2 * x_val

print(f"df/dx (через torch.autograd.grad): {grad_pytorch.item():.6f}")
print(f"df/dx (аналитически):{grad_analytical.item():.6f}")
print(f"Проверка совпадения:{torch.allclose(grad_pytorch, grad_analytical)}")