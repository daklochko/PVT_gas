import streamlit as st
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from math import log10, exp, log
from io import BytesIO

st.title("PVT газ")

tab1, tab2, tab3 = st.tabs(["PVT свойства и псевдодавление", "Влагосодержание", "Гидратообразование"])

def log10_array(input_array):
    return np.log10(input_array)

def exp_array(input_array):
    return np.exp(input_array)

def ln_array(input_array):
    return np.log(input_array)

def create_excel_file(dfs, sheet_names):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for df, sheet_name in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

# PVT свойства и псевдодавление
with tab1:
    
    st.header("Ввод данных")

    P_parameter = st.number_input("Введите давление P, бар:", value = 360.0)
    T_parameter = st.number_input("Введите температуру T, ℃:", value = 87.0)
    gamma_g = st.number_input("Введите относительную плотность газа по воздуху $γ_g$:", value = 0.633)
    x_CO2 = st.number_input("Введите значение xCO2, мольн. доли:", value = 0.0035901, format="%.5f")
    x_H2S = st.number_input("Введите значение xH2S, мольн. доли:", value = 0.00, format="%.5f")
    x_N2 = st.number_input("Введите значение xN2, мольн. доли:", value = 0.0078302, format="%.5f")
    
    def Standing(gamma_g):
        """    
        Standing correlation
    
        """
    
        if gamma_g < 0.75: 
        
            T_pc = 93.333 + 180.556 * gamma_g - 6.944 * gamma_g**2
            P_pc = 46.678 + 1.034 * gamma_g - 2.586 * gamma_g**2
        
        else:
            
            T_pc = 103.889 + 183.333 * gamma_g - 39.722 * gamma_g**2
            P_pc = 48.677 - 3.565 * gamma_g - 0.765 * gamma_g**2
        
        return T_pc, P_pc
    
    def Hankinson_Thomas_Phillips(gamma_g):
        """    
        Hankinson, Thomas and Phillips correlation
    
        """
    
        T_pc_HTP = (170.491 + gamma_g * 307.44) / 1.8
        P_pc_HTP = 0.006894 * (709.604 - gamma_g * 58.718) * 10
    
        return T_pc_HTP, P_pc_HTP
    
    def Carr_Kobayashi_Burrows(T_pc, P_pc, x_CO2, x_H2S, x_N2):
        """    
        Carr-Kobayashi-Burrows’s Correction Method
    
        """
    
        T_pc_ = T_pc - 44.444 * x_CO2 + 72.222 * x_H2S - 138.889 * x_N2
        P_pc_ = P_pc + 30.337 * x_CO2 + 41.368 * x_H2S - 11.721 * x_N2
    
        return T_pc_, P_pc_
    
    def viscosity(rho_g, gamma_g, T):
        """
        Viscosity Lee-Gonzales-Eakin’s
    
        """
        K = ((12.58325 + 0.62444 * gamma_g) * T**1.5) / (116.111 + 309.89875 * gamma_g + T)
        X = 3.448 + 547.78 / T + 0.29223 * gamma_g
        Y = 2.447 - 0.2224 * X
    
        mu_g = 10**(-4) * K * exp_array(X * (rho_g / 1000)**Y)
    
        return mu_g
    
    def Beggs_Brill(T_pr, P_pr):
        """    
        Beggs & Brill correlation
    
        """
        
        A = 1.39 * (T_pr - 0.92)**0.5 - 0.36 * T_pr  - 0.101
        C = 0.132 - 0.32 * log10_array(T_pr)
        E = 9 * (T_pr - 1)
        B = (0.62 - 0.23 * T_pr) * P_pr + (0.066 / (T_pr - 0.86) - 0.037) * P_pr**2 + 0.32 * P_pr**2 / 10**E
        F = 0.3106 - 0.49 * T_pr + 0.1824 * T_pr**2
        D = 10**F
        
        z = A + (1 - A) / exp_array(B) + C * P_pr**D
        
        return z
    
    def Latonov_Gurevich(T_pr, P_pr):
        """ 
        Latonov and Gurevich correlation
        """
        
        return (0.4 * log10_array(T_pr) + 0.73)**P_pr + 0.1 * P_pr
    
    def Hall_Yarborough(T_pr, P_pr, tol):
        """
        Hall-Yarborough correlation
        
        :param P_pr: Давление P_pr
        :param T_pr: Температура T_pr
        :param tol: Точность для критерия остановки
        :return: Найденное значение z
        """
        
        t = 1 / T_pr
        alpha = 0.06125 * t * exp_array(-1.2 * (1 - t)**2)
        
        y = 0.001
        
        def f(y):
            """
            function f(y)
            """
            part1 = -alpha * P_pr
            part2 = (y + y**2 + y**3 - y**4) / (1 - y)**3
            part3 = -(14.76 * t - 9.76 * t**2 + 4.58 * t**3) * y**2
            part4 = (90.7 * t - 242.2 * t**2 + 42.4 * t**3) * y**(2.18 + 2.82 * t)
            
            return part1 + part2 + part3 + part4
        
        def df_dy(y):
            """
            derivative df(y)/dy
            """
            part1 = (1 + 4*y + 4*y**2 - 4*y**3 + y**4) / (1 - y)**4
            part2 = -(29.52 * t - 19.52 * t**2 + 9.16 * t**3) * y
            part3 = (2.18 + 2.82 * t) * (90.7 * t - 242.2 * t**2 + 42.4 * t**3) * y**(1.18 + 2.82 * t)
            
            return part1 + part2 + part3
        
        while True:
            f_value = f(y)
            df_value = df_dy(y)
            
            if np.any(df_value == 0):
                raise ValueError("derivative df(y)/dy = 0")
                
            new_y = y - f_value / df_value
            
            if np.all(np.abs(new_y - y) < tol):
                break
                
            y = new_y
            
        return alpha * P_pr / y
    
    def pseudopressure(P, mu, z):
        
        m_values = [P[0]**2 / (mu[0] * z[0])]
        
        for i in range(1, len(P)):
            
            summ = P[i-1] / (mu[i-1] * z[i-1]) + P[i] / (mu[i] * z[i])
            delta_p = P[i] - P[i-1]
            m_values.append(summ * delta_p + m_values[i-1])
            
        return m_values
    
#     def pseudopressure(P, mu, z):
#        
#        m_values = []
#    
#        for i in range(1, len(P)):
#            
#            summ = P[i-1] / (mu[i-1] * z[i-1]) + P[i] / (mu[i] * z[i])
#            delta_p = P[i] - P[i-1]
#            m_values.append(summ * delta_p)
#            
#            m = np.cumsum(m_values)
#            
#        return m
       
    T_pc, P_pc = Standing(gamma_g)
    T_pc_, P_pc_ = Carr_Kobayashi_Burrows(T_pc, P_pc, x_CO2, x_H2S, x_N2)
    T_pc_HTP, P_pc_HTP = Hankinson_Thomas_Phillips(gamma_g)

    z_Beggs_Brill_1 = Beggs_Brill((T_parameter + 273.15) / T_pc_, P_parameter / P_pc_)
    Bg_Beggs_Brill_1 = 0.003456 * (T_parameter + 273.15) * z_Beggs_Brill_1 / P_parameter
    rho_g_Beggs_Brill_1 = 348.339 * P_parameter * gamma_g / (T_parameter + 273.15) / z_Beggs_Brill_1
    mu_g_Beggs_Brill_1 = viscosity(rho_g_Beggs_Brill_1, gamma_g, (T_parameter + 273.15))
    
    z_Latonov_Gurevich_1 = Latonov_Gurevich((T_parameter + 273.15) / T_pc_HTP, P_parameter / P_pc_HTP)
    Bg_Latonov_Gurevich_1 = 0.003456 * (T_parameter + 273.15) * z_Latonov_Gurevich_1 / P_parameter
    rho_g_Latonov_Gurevich_1 = 348.339 * P_parameter * gamma_g / (T_parameter + 273.15) / z_Latonov_Gurevich_1
    mu_g_Latonov_Gurevich_1 = viscosity(rho_g_Latonov_Gurevich_1, gamma_g, (T_parameter + 273.15))
    
    z_Hall_Yarborough_1 = Hall_Yarborough((T_parameter + 273.15) / T_pc_, P_parameter / P_pc_, 1e-10)
    Bg_Hall_Yarborough_1 = 0.003456 * (T_parameter + 273.15) * z_Hall_Yarborough_1 / P_parameter
    rho_g_Hall_Yarborough_1 = 348.339 * P_parameter * gamma_g / (T_parameter + 273.15) / z_Hall_Yarborough_1
    mu_g_Hall_Yarborough_1 = viscosity(rho_g_Hall_Yarborough_1, gamma_g, (T_parameter + 273.15))
    
    st.write("### Результаты")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="z, B&B", value=f"{z_Beggs_Brill_1:.5f}")
        st.metric(label="$B_g$, $\\text{м}^3$/$\\text{м}^3$, B&B", value=f"{Bg_Beggs_Brill_1:.5f}")
        st.metric(label="$ρ_g$, $\\text{кг/м}^3$, B&B", value=f"{rho_g_Beggs_Brill_1:.2f}")
        st.metric(label="$μ_g$, $\\text{сПз}$, B&B", value=f"{mu_g_Beggs_Brill_1:.5f}")
    with col2:
        st.metric(label="z, Л&Г", value=f"{z_Latonov_Gurevich_1:.5f}")
        st.metric(label="$B_g$, $\\text{м}^3$/$\\text{м}^3$,  Л&Г", value=f"{Bg_Latonov_Gurevich_1:.5f}")
        st.metric(label="$ρ_g$, $\\text{кг/м}^3$, Л&Г", value=f"{rho_g_Latonov_Gurevich_1:.2f}")
        st.metric(label="$μ_g$, $\\text{сПз}$, Л&Г", value=f"{mu_g_Latonov_Gurevich_1:.5f}")
    with col3:
        st.metric(label="z, H&Y", value=f"{z_Hall_Yarborough_1:.5f}")
        st.metric(label="$B_g$, $\\text{м}^3$/$\\text{м}^3$, H&Y", value=f"{Bg_Hall_Yarborough_1:.5f}")
        st.metric(label="$ρ_g$, $\\text{кг/м}^3$, H&Y", value=f"{rho_g_Hall_Yarborough_1:.2f}")
        st.metric(label="$μ_g$, $\\text{сПз}$, H&Y", value=f"{mu_g_Hall_Yarborough_1:.5f}")
        
    N = 1000
    P_correlation = np.linspace(1.01325, P_parameter, N)
    
    z_Beggs_Brill = Beggs_Brill(np.full(N, (T_parameter + 273.15) / T_pc_), P_correlation / P_pc_)
    Bg_Beggs_Brill = 0.003456 * np.full(N, (T_parameter + 273.15)) * z_Beggs_Brill / P_correlation
    rho_g_Beggs_Brill = 348.339 * P_correlation * gamma_g / np.full(N, (T_parameter + 273.15)) / z_Beggs_Brill
    mu_g_Beggs_Brill = viscosity(rho_g_Beggs_Brill, gamma_g, np.full(N, (T_parameter + 273.15)))
    m_Beggs_Brill = pseudopressure(P_correlation, mu_g_Beggs_Brill, z_Beggs_Brill)

    z_Latonov_Gurevich = Latonov_Gurevich(np.full(N, (T_parameter + 273.15) / T_pc_HTP), P_correlation / P_pc_HTP)
    Bg_Latonov_Gurevich = 0.003456 * np.full(N, (T_parameter + 273.15)) * z_Latonov_Gurevich / P_correlation
    rho_g_Latonov_Gurevich = 348.339 * P_correlation * gamma_g / np.full(N, (T_parameter + 273.15)) / z_Latonov_Gurevich
    mu_g_Latonov_Gurevich = viscosity(rho_g_Latonov_Gurevich, gamma_g, np.full(N, (T_parameter + 273.15)))  
    m_Latonov_Gurevich = pseudopressure(P_correlation, mu_g_Latonov_Gurevich, z_Latonov_Gurevich)
    
    z_Hall_Yarborough = Hall_Yarborough(np.full(N, (T_parameter + 273.15) / T_pc_), P_correlation / P_pc_, 1e-10)
    Bg_Hall_Yarborough = 0.003456 * np.full(N, (T_parameter + 273.15)) * z_Hall_Yarborough / P_correlation
    rho_g_Hall_Yarborough = 348.339 * P_correlation * gamma_g / np.full(N, (T_parameter + 273.15)) / z_Hall_Yarborough
    mu_g_Hall_Yarborough = viscosity(rho_g_Hall_Yarborough, gamma_g, np.full(N, (T_parameter + 273.15)))
    m_Hall_Yarborough = pseudopressure(P_correlation, mu_g_Hall_Yarborough, z_Hall_Yarborough)
    
#    m_Beggs_Brill = np.append(m_Beggs_Brill, m_Beggs_Brill[-1])
#    m_Latonov_Gurevich = np.append(m_Latonov_Gurevich, m_Latonov_Gurevich[-1])
#    m_Hall_Yarborough = np.append(m_Hall_Yarborough, m_Hall_Yarborough[-1])
    
    df_B_B = pd.DataFrame({
        "P, бар": P_correlation,
        "T, ℃": np.full(N, (T_parameter)),
        "z": z_Beggs_Brill,
        "Bg, м3/м3": Bg_Beggs_Brill,
        "ρg, кг/м3": rho_g_Beggs_Brill,
        "μg, сПз": mu_g_Beggs_Brill,
        "m(P), бар²/сПз": m_Beggs_Brill
        
    })

    df_L_G = pd.DataFrame({
        "P, бар": P_correlation,
        "T, ℃": np.full(N, (T_parameter)),
        "z": z_Latonov_Gurevich,
        "Bg, м3/м3": Bg_Latonov_Gurevich,
        "ρg, кг/м3": rho_g_Latonov_Gurevich,
        "μg, сПз": mu_g_Latonov_Gurevich,
        "m(P), бар²/сПз": m_Latonov_Gurevich
    })


    df_H_Y = pd.DataFrame({
        "P, бар": P_correlation,
        "T, ℃": np.full(N, (T_parameter)),
        "z": z_Hall_Yarborough,
        "Bg, м3/м3": Bg_Hall_Yarborough,
        "ρg, кг/м3": rho_g_Hall_Yarborough,
        "μg, сПз": mu_g_Hall_Yarborough,
        "m(P), бар²/сПз": m_Hall_Yarborough
    })

    
    # Кнопка для выгрузки Excel-файла
    st.write("### Таблицы PVT и псевдодавления")
  
    dfs = [df_B_B, df_L_G, df_H_Y]
    sheet_names = ["B&B", "Л&Г", "H&Y"]

    excel_file = create_excel_file(dfs, sheet_names)

    st.download_button(
        label="Скачать таблицы PVT и псевдодавления",
        data=excel_file,
        file_name="PVT_pseudopressure.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Графики 
    st.write("### Графики")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x = P_correlation, y = z_Beggs_Brill, name = 'B&B'))
    fig.add_trace(go.Scatter(x = P_correlation, y = z_Latonov_Gurevich, name = 'Л&Г'))
    fig.add_trace(go.Scatter(x = P_correlation, y = z_Hall_Yarborough, name = 'H&Y'))

    fig.update_layout(
        xaxis_title="P, бар",
        yaxis_title="z"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    #st.write("### Псевдодавление")
     
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = P_correlation, y = m_Beggs_Brill, name = 'B&B'))
    fig.add_trace(go.Scatter(x = P_correlation, y = m_Latonov_Gurevich, name = 'Л&Г'))
    fig.add_trace(go.Scatter(x = P_correlation, y = m_Hall_Yarborough, name = 'H&Y'))
    
    fig.update_layout(
        xaxis_title="P, бар",
        yaxis_title="m(P), бар²/сПз",
        yaxis=dict(tickformat=".1e")
    )

    st.plotly_chart(fig, use_container_width=True)    

# Влагосодержание
with tab2:
  
    st.header("Ввод данных")

    P_w_1 = st.number_input("Введите значение P, бар:", value = 100.0)
    T_w_1 = st.number_input("Введите значение T, ℃:", value = 32.0)
    
    def Bukacek_H(P, T): 
        
        return 4.67 / (P * 1.01972) * exp_array(0.0735 * T - 0.00027 * T**2) + 0.0418 * exp_array(0.054 * T - 0.0002 * T**2)
    
    def Bukacek_W(P, T):  
        
        A = 10**(10.9351 - 1638.36 / (T + 273.15) - 98162 / (T + 273.15)**2)
        B = 10**(6.69449 - 1713.26 / (T + 273.15))
        
        return 0.016016 * (A / (14.5038 * P) + B)
    
    def Wattenberger(P, T):
        
        A = 10**(10.9351 - 2949.05 / ((T + 273.15) * 1.8) - 318045 / ((T + 273.15) * 1.8)**2)
        B = 10**(6.69449 - 3083.87 / ((T + 273.15) * 1.8))
        
        return (A / (P * 14.5038) + B) * 453.5929094 * 0.0000353147
    
    def Daubert_Danner(P, T):
        
        P_w_s = (10**(-5)) * exp_array(73.649 - 7258.2 / (T + 273.15) - 7.3037 * ln_array(T + 273.15) + 0.0000041653 * (T + 273.15)**2)
        
        return 751.5658 * P_w_s / P
    
    N1 = Bukacek_H(P_w_1, T_w_1)
    N2 = Bukacek_W(P_w_1, T_w_1)
    N3 = Wattenberger(P_w_1, T_w_1)
    N4 = Daubert_Danner(P_w_1, T_w_1)
    
    st.write("### Результаты")
    col1, col2 = st.columns(2)
    with col1:
#        st.metric(label="Бюкачек по Хилько, $\\text{г/м}^3$", value=f"{N1:.4f}")
        st.metric(label="Бюкачек по Ваттенбергеру (в переводе), $\\text{г/м}^3$", value=f"{N2:.4f}")
    with col2:
#        st.metric(label="По Ваттенбергеру (в оригинале), $\\text{г/м}^3$", value=f"{N3:.4f}")
        st.metric(label="Daubert and Danner, $\\text{г/м}^3$", value=f"{N4:.4f}")
        
    P_w = np.linspace(P_w_1, 1.0, 11)
    T_w = np.full(11, T_w_1)

    w_B_H = Bukacek_H(P_w, T_w)
    w_B_W = Bukacek_W(P_w, T_w)
    w_W = Wattenberger(P_w, T_w)
    w_D_D = Daubert_Danner(P_w, T_w)
    
    df_w_B_H = pd.DataFrame({
        "T, ℃": T_w,
        "P, бар": P_w,
        "W, г/м3": w_B_H
    })

    df_w_B_W = pd.DataFrame({
        "T, ℃": T_w,
        "P, бар": P_w,
        "W, г/м3": w_B_W
    })

    df_w_W = pd.DataFrame({
        "T, ℃": T_w,
        "P, бар": P_w,
        "W, г/м3": w_W
    })

    df_D_D = pd.DataFrame({
        "T, ℃": T_w,
        "P, бар": P_w,
        "W, г/м3": w_D_D
    })
    
    # Кнопка для выгрузки Excel-файла
    st.write("### Таблицы влагосодержания")
    
    #dfs = [df_w_B_H, df_w_B_W, df_w_W, df_D_D]
    #sheet_names = ["Бюкачек по Хилько", "По Ваттенбергеру (перевод)", "По Ваттенбергеру (оригинал)", "Daubert and Danner"]
    dfs = [df_w_B_W, df_D_D]
    sheet_names = ["По Ваттенбергеру (перевод)", "Daubert and Danner"]

    excel_file = create_excel_file(dfs, sheet_names)

    st.download_button(
        label="Скачать таблицы влагосодержания",
        data=excel_file,
        file_name="Влагосодержание.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Графики
    st.write("### Кривые влагосодержания")
    
    fig = go.Figure()
    
    #fig.add_trace(go.Scatter(x = P_w, y = w_B_H, mode = 'lines', name = 'Бюкачек по Хилько'))
    fig.add_trace(go.Scatter(x = P_w, y = w_B_W, mode = 'lines', name = 'Бюкачек по Ваттенбергеру'))
    #fig.add_trace(go.Scatter(x = P_w, y = w_W, mode = 'lines', name = 'По Ваттенбергеру'))
    fig.add_trace(go.Scatter(x = P_w, y = w_D_D, mode = 'lines', name = 'D&D'))
    
    fig.update_layout(
        xaxis_title = "P, бар",
        yaxis_title = "W, г/м3"
    )
    
    # Отображение графика
    st.plotly_chart(fig, use_container_width=True)

# Гидратообразование    
with tab3:
    st.header("Ввод данных")

    P_parameter = st.number_input("Введите значение P, МПа:", value = 0.101325)
    gamma_g = st.number_input("Введите относительную плотность газа по воздуху $γ_g$:", value = 0.554)
    
    def СТO_ГП_валанжин(P):
        
        if np.any(P) < 8:
            T_gidr = 7.55*ln_array(P) + 1.87
        if np.any(P) > 10:
            T_gidr = 7.65*ln_array(P)+1.85       
            
        return T_gidr
    
    def СТO_ГП_сеноман(P):
        
        return 9.97*ln_array(P)-9.3
    
    def Towler_Mokhatab(P, gamma_g):
        """
        Корреляция Towler & Mokhatab
        """    
        return 7.48333 * ln_array(P * 10) + 16.5502 * ln_array(gamma_g) - 0.930556 * ln_array(P * 10) * ln_array(gamma_g) - 9.06983
    
    N1 = СТO_ГП_валанжин(P_parameter)
    N2 = СТO_ГП_сеноман(P_parameter)
    N3 = Towler_Mokhatab(P_parameter, gamma_g)
    
    st.write("### Результаты")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="СТО ГП сеноман, °C", value=f"{N1:.4f}")
    with col2:
        st.metric(label="СТО ГП валанжин, °C", value=f"{N2:.4f}")
    with col3:
        st.metric(label="Корреляция Towler & Mokhatab, °C", value=f"{N3:.4f}")
        
    P_corr = np.array([0.101325, 1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40])
    
    T_gidr_GP_val = СТO_ГП_валанжин(P_corr)
    T_gidr_GP_sen = СТO_ГП_сеноман(P_corr)
    T_gidr_T_W = Towler_Mokhatab(P_corr, gamma_g)
    
    df_gidr_GP_val = pd.DataFrame({        
        "P, бар": P_corr,
        "T, ℃": T_gidr_GP_val
    })

    df_gidr_GP_sen = pd.DataFrame({
        "P, бар": P_corr,
        "T, ℃": T_gidr_GP_sen
    })

    df_gidr_T_W = pd.DataFrame({
        "P, бар": P_corr,
        "T, ℃": T_gidr_T_W
    })
        
    # Кнопка для выгрузки Excel-файла
    st.write("### Таблицы гидратообразования")
    
    dfs = [df_gidr_GP_val, df_gidr_GP_sen, df_gidr_T_W]
    sheet_names = ["СТО ГП сеноман", "СТО ГП валанжин", "Корреляция Towler & Mokhatab"]

    excel_file = create_excel_file(dfs, sheet_names)

    st.download_button(
        label="Скачать таблицы гидратообразования",
        data=excel_file,
        file_name="Гидратообразование.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
        
    # Графики   
    st.write("### Кривые гидратообразования")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = T_gidr_GP_val, 
        y = P_corr, mode = 'lines', 
        line=dict(color="blue", width=3), 
        name = 'СТО ГП сеноман'
    ))
    
    fig.add_trace(go.Scatter(
        x = T_gidr_GP_sen, 
        y = P_corr, 
        mode = 'lines', 
        line=dict(color="red", width=3), 
        name = 'СТО ГП валанжин'
    ))
    
    fig.add_trace(go.Scatter(
        x = T_gidr_T_W, 
        y = P_corr, 
        mode = 'lines', 
        line=dict(color="green", width=3), 
        name = 'Корреляция Towler & Mokhatab'
    ))

    x_fill = np.append(T_gidr_GP_val, T_gidr_GP_val[::-1]) 
    y_fill = np.append(P_corr, [max(P_corr)] * len(T_gidr_GP_val))

    fig.add_trace(go.Scatter(
        x=x_fill,
        y=y_fill,
        fill='toself',
        mode='lines',
        fillcolor='rgba(0, 100, 255, 0.1)',
        line=dict(width=0),
        name='Область гидратообразования, сеноман'
    ))

    x_fill = np.append(T_gidr_GP_sen, T_gidr_GP_sen[::-1]) 
    y_fill = np.append(P_corr, [max(P_corr)] * len(T_gidr_GP_sen))

    fig.add_trace(go.Scatter(
        x=x_fill,
        y=y_fill,
        fill='toself',
        mode='lines',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0),
        name='Область гидратообразования, валанжин'
    ))

    x_fill = np.append(T_gidr_T_W, T_gidr_T_W[::-1]) 
    y_fill = np.append(P_corr, [max(P_corr)] * len(T_gidr_T_W))

    fig.add_trace(go.Scatter(
        x=x_fill,
        y=y_fill,
        fill='toself',
        mode='lines',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0),
        name='Область гидратообразования, T&M'
    ))

    
    fig.update_layout(
        xaxis_title="t, °C",
        yaxis_title="P, МПа",
    )
    
    # Отображение графика
    st.plotly_chart(fig, use_container_width=True)