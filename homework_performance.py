import torch
import time

# определяем устройство для расчетаа
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

print("Задание 3.1: Подготовка данных ")
# создаем большие матрицы указанных размеров со случайными числами
A_cpu = torch.rand(64, 1024, 1024, device=device_cpu)
B_cpu = torch.rand(128, 512, 512, device=device_cpu)
C_cpu = torch.rand(256, 256, 256, device=device_cpu)

print(f"Создана матрица A (CPU): {A_cpu.shape}")
print(f"Создана матрица B (CPU): {B_cpu.shape}")
print(f"Создана матрица C (CPU): {C_cpu.shape}")

# перенос данных на GPU
if torch.cuda.is_available():
    A_gpu = A_cpu.to(device_gpu)
    B_gpu = B_cpu.to(device_gpu)
    C_gpu = C_cpu.to(device_gpu)
    torch.cuda.synchronize()  # ожидаем окончания переноса
    print("Данные успешно перенесены на GPU.")
else:
    A_gpu = None
    B_gpu = None
    C_gpu = None
    print("GPU недоступен, расчеты будут проводиться только на CPU.")

# Задание 3.2: Функция измерения времени

def measure_time(device, op_func, *args):
    for _ in range(2):
        op_func(*args)

    if device.type == "cuda":
        # Используем специальные события КУДА для точного замера на ГПУ
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        op_func(*args)
        end_event.record()

        torch.cuda.synchronize()  # ждем окончания расчетов на видеокарте
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms
    else:
        # для CPU используем стандартное системное время
        start_time = time.time()
        op_func(*args)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        return elapsed_time_ms

# Задание 3.3: Сравнение операций (10 баллов)
print("\nЗадание 3.3: Сравнение операций ")


# Определяем функции для каждой из требуемых операций над тензором A
def op_matmul(x):
    return torch.matmul(x, x)

def op_add(x):
    return x + x

def op_mul(x):
    return x * x

def op_transpose(x):
    return x.transpose(-1, -2)

def op_sum(x):
    return x.sum()

operations = [
    ("Матричное умножение", op_matmul),
    ("Поэлементное сложение", op_add),
    ("Поэлементное умножение", op_mul),
    ("Транспонирование", op_transpose),
    ("Вычисление суммы", op_sum)
]

print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
print("-" * 65)

for name, op in operations:
    # замеряем время на CPU
    cpu_time = measure_time(device_cpu, op, A_cpu)

    # замеряем время на GPU
    if torch.cuda.is_available():
        gpu_time = measure_time(device_gpu, op, A_gpu)
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        speedup_str = f"{speedup:.2f}x"
        gpu_time_str = f"{gpu_time:.2f}"
    else:
        gpu_time_str = "N/A"
        speedup_str = "N/A"

    print(f"{name:<25} | {cpu_time:<10.2f} | {gpu_time_str:<10} | {speedup_str:<10}")

# Задание 3.4: Анализ результатов
print("\nЗадание 3.4: Анализ результатов")
print("1. Какие операции получают наибольшее ускорение на GPU?")
print("ответ: матричное умножение (matmul), это сложная параллельная операция, ")
print("которая идеально разбивается на тысячи вычислительных ядер графического процессора")
print("\n2. Почему некоторые операции могут быть медленнее на GPU?")
print("ответ: простые операции (например, сложение или нахождение суммы) выполняются очень быстро ")
print("Накладные расходы на передачу сигналов управления с CPU на GPU могут оказаться больше, ")
print("чем время непосредственного расчета на видеокарте")
print("\n3. Как размер матриц влияет на ускорение?")
print("ответ: чем больше размер матрицы, тем выше ускорение на GPU ")
print("на маленьких матрицах видеокарта простаивает, а на больших - полностью раскрывает параллельный потенциал.")
print("\n4. Что происходит при передаче данных между CPU и GPU?")
print("ответ: данные копируются из оперативной памяти компьютера (RAM) в видеопамять видеокарты (VRAM) ")
print("через шину PCIe. Это относительно медленный процесс, поэтому данные стараются копировать ")
print("как можно реже (один раз за сессию загрузки)")