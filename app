import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página en modo ancho (wide)
st.set_page_config(page_title="Simulador Runge-Kutta Pro", layout="wide")

st.title("🧮 Simulador Numérico Avanzado: Runge-Kutta (RK4) vs Euler")
st.markdown("Proyecto académico para la materia de **Métodos Numéricos** — 4to Semestre.")

# --- BARRA LATERAL: PARÁMETROS DE ENTRADA ---
st.sidebar.header("⚙️ Parámetros de Simulación")

opcion_eq = st.sidebar.selectbox(
    "Selecciona la Ecuación Diferencial:",
    ("dy/dt = y - t (Crecimiento/Enfriamiento)", "dy/dt = -2*t*y (Decaimiento rápido)")
)

t0 = st.sidebar.number_input("Tiempo Inicial (t0):", value=0.0, step=0.1)
y0 = st.sidebar.number_input("Condición Inicial y(t0):", value=2.0, step=0.5)
t_final = st.sidebar.number_input("Tiempo Final:", value=4.0, step=0.5)
h = st.sidebar.slider("Tamaño de paso (h):", min_value=0.02, max_value=0.5, value=0.1, step=0.01)

# --- DEFINICIÓN DE FUNCIONES ---
if "y - t" in opcion_eq:
    f = lambda t, y: y - t
    sol_exacta = lambda t: t + 1 + (y0 - 1) * np.exp(t)
else:
    f = lambda t, y: -2 * t * y
    sol_exacta = lambda t: y0 * np.exp(-t**2)

# --- ALGORITMOS NUMÉRICOS ---
def resolver_euler(f, t0, y0, h, t_final):
    t_vals = np.arange(t0, t_final + h/2, h)
    y_vals = [y0]
    for i in range(len(t_vals) - 1):
        y_next = y_vals[-1] + h * f(t_vals[i], y_vals[-1])
        y_vals.append(y_next)
    return t_vals, np.array(y_vals)

def resolver_rk4(f, t0, y0, h, t_final):
    t_vals = np.arange(t0, t_final + h/2, h)
    y_vals = [y0]
    for i in range(len(t_vals) - 1):
        t_n = t_vals[i]
        y_n = y_vals[-1]
        
        k1 = f(t_n, y_n)
        k2 = f(t_n + 0.5 * h, y_n + 0.5 * h * k1)
        k3 = f(t_n + 0.5 * h, y_n + 0.5 * h * k2)
        k4 = f(t_n + h, y_n + h * k3)
        
        y_next = y_n + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_vals.append(y_next)
    return t_vals, np.array(y_vals)

# CÁLCULOS
t_num, y_euler = resolver_euler(f, t0, y0, h, t_final)
_, y_rk4 = resolver_rk4(f, t0, y0, h, t_final)

# Alta resolución para una curva analítica súper fluida
t_continuo = np.linspace(t0, t_final, 500)
y_exacta = sol_exacta(t_continuo)

# --- GRÁFICA PRINCIPAL AMPLIADA Y FLUIDA ---
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(t_continuo, y_exacta, label="Solución Exacta (Analítica)", color="#111111", linewidth=2.5, linestyle="--")
ax.plot(t_num, y_euler, label="Método de Euler (RK1)", color="#ff4b4b", marker="o", linewidth=1.5, alpha=0.8)
ax.plot(t_num, y_rk4, label="Runge-Kutta 4to Orden (RK4)", color="#0068c9", marker="s", linewidth=2, alpha=0.9)

ax.set_xlabel("Tiempo (t)", fontsize=12, fontweight='bold')
ax.set_ylabel("y(t)", fontsize=12, fontweight='bold')
ax.set_title("Comparación de Comportamiento y Convergencia", fontsize=14, fontweight='bold')
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=11)

# Mostrar gráfico ocupando todo el ancho disponible de forma grande
st.pyplot(fig, use_container_width=True)

# --- PANEL INFERIOR DE MÉTRICAS Y CONCLUSIONES ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

error_euler = np.abs(sol_exacta(t_num[-1]) - y_euler[-1])
error_rk4 = np.abs(sol_exacta(t_num[-1]) - y_rk4[-1])

with col1:
    st.metric(label="📊 Error final Euler", value=f"{error_euler:.6f}")

with col2:
    st.metric(label="🎯 Error final Runge-Kutta (RK4)", value=f"{error_rk4:.6f}")

with col3:
    st.success("💡 **Conclusión:** Al refinar el paso $h$, se nota claramente cómo el error de RK4 es infinitamente menor que el de Euler.")
