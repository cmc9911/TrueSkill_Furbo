import pandas as pd
import trueskill
import itertools
import streamlit as st

# ============================
# CONFIGURACIÓN TrueSkill
# ============================
env = trueskill.TrueSkill(draw_probability=0.1)
ratings = {}

# ============================
# FUNCIONES
# ============================
def get_rating(jugador):
    return ratings.get(jugador, env.create_rating())

def atributos_a_mu(row):
    # Ponderaciones (no necesitan sumar 1)
    pesos = {
        "Velocidad": 0.2,
        "Resistencia": 0.3,
        "Habilidad Técnica": 0.25,
        "Defensa": 0.2,
        "Ataque": 0.2,
        "Regate": 0.1,
        "Cuerpo": 0.1,
    }
    suma_pesos = sum(pesos.values())
    mu = sum(row[atributo] * (peso / suma_pesos) for atributo, peso in pesos.items())
    return mu * 5  # Escala para que valores iniciales estén en rango TrueSkill

def inicializar_ratings(df_atributos):
    for _, row in df_atributos.iterrows():
        mu_inicial = atributos_a_mu(row)
        ratings[row["Jugador"]] = env.create_rating(mu=mu_inicial, sigma=8.333)

def actualizar_partido(equipo_A, equipo_B, resultado, goles_dict, goles_equipo_A, goles_equipo_B):
    ratings_A = [get_rating(j) for j in equipo_A]
    ratings_B = [get_rating(j) for j in equipo_B]

    if resultado == 1:
        nuevos_A, nuevos_B = env.rate([ratings_A, ratings_B], ranks=[0, 1])
    elif resultado == -1:
        nuevos_B, nuevos_A = env.rate([ratings_B, ratings_A], ranks=[0, 1])
    else:
        nuevos_A, nuevos_B = env.rate([ratings_A, ratings_B], ranks=[0, 0])

    for j, r in zip(equipo_A, nuevos_A):
        ratings[j] = r
    for j, r in zip(equipo_B, nuevos_B):
        ratings[j] = r

    # Ajuste adicional por goles
    ajuste_gol_equipo = 0.5
    ajuste_contra = 0.5
    ajuste_gol = 0.2

    for jugador in equipo_A:
        mu = ratings[jugador].mu
        sigma = ratings[jugador].sigma
        mu += goles_dict.get(jugador, 0) * ajuste_gol
        mu += goles_equipo_A * ajuste_gol_equipo
        mu -= goles_equipo_B * ajuste_contra
        ratings[jugador] = env.create_rating(mu=mu, sigma=sigma)

    for jugador in equipo_B:
        mu = ratings[jugador].mu
        sigma = ratings[jugador].sigma
        mu += goles_dict.get(jugador, 0) * ajuste_gol
        mu += goles_equipo_B * ajuste_gol_equipo
        mu -= goles_equipo_A * ajuste_contra
        ratings[jugador] = env.create_rating(mu=mu, sigma=sigma)

def prob_victoria(equipo_A, equipo_B):
    mu_A = sum(get_rating(j).mu for j in equipo_A)
    mu_B = sum(get_rating(j).mu for j in equipo_B)
    diff = mu_A - mu_B
    return 1 / (1 + 10 ** (-diff / 400))

def equipos_balanceados(jugadores, df_atributos, tamaño=5, equilibrar_portero=False, peso_portero=0.3):
    mejor_diff = float("inf")
    mejor_equipo = None

    portero_map = dict(zip(df_atributos["Jugador"], df_atributos["Portero"]))
    max_mu = max(ratings[j].mu for j in jugadores)
    max_portero = max(portero_map.get(j, 0) for j in jugadores)

    for equipo_A in itertools.combinations(jugadores, tamaño):
        equipo_B = [j for j in jugadores if j not in equipo_A]

        mu_A = sum(get_rating(j).mu for j in equipo_A)
        mu_B = sum(get_rating(j).mu for j in equipo_B)
        diff_mu = abs(mu_A - mu_B) / max_mu if max_mu > 0 else 0

        if equilibrar_portero:
            portero_A = sum(portero_map.get(j, 0) for j in equipo_A)
            portero_B = sum(portero_map.get(j, 0) for j in equipo_B)
            diff_portero = abs(portero_A - portero_B) / max_portero if max_portero > 0 else 0
            diff_total = diff_mu * (1 - peso_portero) + diff_portero * peso_portero
        else:
            diff_total = diff_mu

        if diff_total < mejor_diff:
            mejor_diff = diff_total
            mejor_equipo = (equipo_A, equipo_B)

    return mejor_equipo

# ============================
# APP STREAMLIT
# ============================
st.title("⚽ Balanceador de Equipos con TrueSkill + Stats + Portero")

# Cargar datos desde archivo en el repo
ruta_excel = "datos_futbol.xlsx"
df_partidos = pd.read_excel(ruta_excel, sheet_name="partidos")
df_atributos = pd.read_excel(ruta_excel, sheet_name="atributos")

# Inicializar ratings
inicializar_ratings(df_atributos)

# Procesar partidos históricos
for partido_id, datos_partido in df_partidos.groupby("Partido"):
    equipo_A = datos_partido[datos_partido["Equipo"] == "A"]["Jugador"].tolist()
    equipo_B = datos_partido[datos_partido["Equipo"] == "B"]["Jugador"].tolist()
    goles_equipo_A = datos_partido[datos_partido["Equipo"] == "A"]["Goles"].sum()
    goles_equipo_B = datos_partido[datos_partido["Equipo"] == "B"]["Goles"].sum()
    goles_dict = dict(zip(datos_partido["Jugador"], datos_partido["Goles"]))

    if goles_equipo_A > goles_equipo_B:
        resultado = 1
    elif goles_equipo_A < goles_equipo_B:
        resultado = -1
    else:
        resultado = 0
    
    actualizar_partido(equipo_A, equipo_B, resultado, goles_dict, goles_equipo_A, goles_equipo_B)

# ============================
# MOSTRAR RATINGS
# ============================
st.subheader("📊 Ratings actuales")
df_ratings = pd.DataFrame([
    {"Jugador": j, "μ (Media)": r.mu, "σ (±)": r.sigma}
    for j, r in ratings.items()
])
df_portero = df_atributos[["Jugador", "Portero"]]
df_completo = pd.merge(df_ratings, df_portero, on="Jugador", how="left")
df_completo = df_completo.sort_values(by="μ (Media)", ascending=False)
st.dataframe(df_completo, use_container_width=True)

# ============================
# SELECCIÓN DE JUGADORES
# ============================
seleccionados = st.multiselect(
    "Selecciona exactamente 10 jugadores",
    list(ratings.keys()),
    max_selections=10
    key="seleccion_jugadores"
)

equilibrar_portero = st.checkbox("Equilibrar nivel de portero",
                     key="equilibrar_portero_toggle")
peso_portero = 0.3
if equilibrar_portero:
    peso_portero = st.slider("Peso portero en equilibrio", 0.0, 1.0, 0.3, 0.05)

# ============================
# BALANCEAR EQUIPOS
# ============================
if len(seleccionados) == 10:
    eq_A, eq_B = equipos_balanceados(seleccionados, df_atributos, tamaño=5,
                                     equilibrar_portero=equilibrar_portero,
                                     peso_portero=peso_portero)
    probA = prob_victoria(eq_A, eq_B) * 100
    portero_map = dict(zip(df_atributos["Jugador"], df_atributos["Portero"]))

    sum_mu_A = sum(ratings[j].mu for j in eq_A)
    sum_mu_B = sum(ratings[j].mu for j in eq_B)
    sum_portero_A = sum(portero_map.get(j, 0) for j in eq_A)
    sum_portero_B = sum(portero_map.get(j, 0) for j in eq_B)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🟦 Equipo A")
        st.table(pd.DataFrame({
            "Jugador": eq_A,
            "μ": [ratings[j].mu for j in eq_A],
            "Portero": [portero_map.get(j, 0) for j in eq_A]
        }))
        st.write(f"**Suma μ:** {sum_mu_A:.2f}")
        st.write(f"**Suma Portero:** {sum_portero_A:.2f}")

    with col2:
        st.markdown("### 🟥 Equipo B")
        st.table(pd.DataFrame({
            "Jugador": eq_B,
            "μ": [ratings[j].mu for j in eq_B],
            "Portero": [portero_map.get(j, 0) for j in eq_B]
        }))
        st.write(f"**Suma μ:** {sum_mu_B:.2f}")
        st.write(f"**Suma Portero:** {sum_portero_B:.2f}")

    st.markdown(f"**Probabilidad de victoria del Equipo A:** {probA:.2f}%")
elif len(seleccionados) > 0:
    st.warning("⚠ Debes seleccionar exactamente 10 jugadores.")






