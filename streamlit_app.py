import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CONSTANTES GLOBAIS
# ==============================================================================

st.set_page_config(layout="wide")
st.title("Simulador de Despacho de BESS e Autonomia de Diesel")

# --- Constantes do Modelo (Não alteráveis pela UI) ---
INTERVALOS_POR_HORA = 12 # Intervalos de 5 min (60/12 = 5 min)
DIAS_SIMULACAO_LONGA = 150 # Limite de dias para o gráfico de autonomia
EFICIENCIA_FV = 0.75

# Carga (Dados mantidos)
DADOS_CARGA_HORARIA_STR = "17.000-17.000-17.000-17.000-17.000-20.000-34.000-39.000-45.000-50.000-65.000-85.000-80.000-75.000-60.000-42.000-50.000-84.000-150.000-79.000-61.000-45.000-30.000-25.000"
CARGA_HORARIA_24H = [float(val.replace('.', ''))/1000 for val in DADOS_CARGA_HORARIA_STR.split('-')]

# BESS (Constantes)
BESS_EFICIENCIA_CICLO_COMPLETO = 0.82
EFICIENCIA_CARREGAMENTO = np.sqrt(BESS_EFICIENCIA_CICLO_COMPLETO)
EFICIENCIA_DESCARREGAMENTO = np.sqrt(BESS_EFICIENCIA_CICLO_COMPLETO)
SOC_LIMITE_MAX = 90
SOC_LIMITE_MIN_NORMAL = 40
SOC_LIMITE_MIN_EMERGENCIA = 20
SOC_RAMPA_INICIO = 85 # SOC (%) em que a potência de carga começa a ser reduzida

# Aplicações (Constantes)
ATIVAR_SUAVIZACAO_FV = True
JANELA_SUAVIZACAO_MINUTOS = 30 # Define a "suavidade" da rampa.

# Diesel (Constantes)
CAPACIDADE_TOTAL_DIESEL_L = 12000
SFC = 0.285 # Fator de Consumo Específico: L/kWh

# Perfil de Geração FV (Constante)
FATOR_GERACAO_HORARIA = {
    6: 0.1, 7: 0.3, 8: 0.5, 9: 0.65, 10: 0.72, 11: 0.75, 12: 0.73,
    13: 0.68, 14: 0.58, 15: 0.45, 16: 0.28, 17: 0.1, 18: 0.0
}

# ==============================================================================
# 2. FUNÇÕES DE SIMULAÇÃO (CACHEADAS)
# ==============================================================================

def calcular_consumo_diesel(potencia_saida_kw):
    """Calcula o consumo de diesel em L/h com base na potência gerada."""
    return potencia_saida_kw * SFC

@st.cache_data
def run_short_term_simulation(
    dias_simulacao,
    potencia_pico_fv_base,
    ceu_aberto,
    bess_capacidade_kwh,
    bess_potencia_max_kw,
    soc_inicial_fracao,
    numero_total_gmgs,
    gmg_potencia_unitaria,
    gmg_fator_potencia_eficiente,
    carga_limite_emergencia
):
    """
    Executa a simulação de curto prazo (Gráficos 1 e 3) com base nos parâmetros da UI.
    Usa perfil FV com ruído e lógica de despacho detalhada.
    """
    
    # --- 1. Preparação dos Parâmetros Derivados ---
    numero_de_passos = dias_simulacao * 24 * INTERVALOS_POR_HORA
    passo_de_tempo_h = 1.0 / INTERVALOS_POR_HORA
    vetor_tempo = np.linspace(0, dias_simulacao * 24, numero_de_passos, endpoint=False)
    
    # Carga
    carga_horaria_dias = CARGA_HORARIA_24H * dias_simulacao
    pontos_de_tempo_horarios = np.arange(dias_simulacao * 24)
    vetor_carga = np.interp(vetor_tempo, pontos_de_tempo_horarios, carga_horaria_dias)
    
    # FV
    potencia_pico_fv_curto = potencia_pico_fv_base * EFICIENCIA_FV * ceu_aberto
    
    # BESS
    bess_soc_kwh = bess_capacidade_kwh * soc_inicial_fracao
    
    # GMG
    gmg_potencia_max_por_unidade = gmg_potencia_unitaria * gmg_fator_potencia_eficiente
    
    # Suavização
    janela_suavizacao_passos = int(JANELA_SUAVIZACAO_MINUTOS / (60 / INTERVALOS_POR_HORA))

    # --- 2. Geração de Perfis FV (Curto Prazo - COM RUÍDO) ---
    np.random.seed(42)  # Para reprodutibilidade
    perfil_fv_24h_curto = np.zeros(24 * INTERVALOS_POR_HORA)
    
    for i, t in enumerate(np.linspace(0, 24, 24 * INTERVALOS_POR_HORA, endpoint=False)):
        hora_base = int(t)
        if hora_base in FATOR_GERACAO_HORARIA and (hora_base + 1) in FATOR_GERACAO_HORARIA:
            valor_inicial = FATOR_GERACAO_HORARIA[hora_base]
            valor_final = FATOR_GERACAO_HORARIA[hora_base + 1]
            fracao = t - hora_base
            valor_interpolado = valor_inicial + (valor_final - valor_inicial) * fracao
            ruido = np.random.normal(0, 0.08)
            ruido = np.clip(ruido, -0.10, 0.10)
            valor_com_osc = valor_interpolado * (1 + ruido)
            perfil_fv_24h_curto[i] = max(0, valor_com_osc * potencia_pico_fv_curto)
        elif hora_base in FATOR_GERACAO_HORARIA:
            valor = FATOR_GERACAO_HORARIA[hora_base]
            ruido = np.random.normal(0, 0.08)
            ruido = np.clip(ruido, -0.10, 0.10)
            valor_com_osc = valor * (1 + ruido)
            perfil_fv_24h_curto[i] = max(0, valor_com_osc * potencia_pico_fv_curto)

    vetor_geracao_fv_original = np.tile(perfil_fv_24h_curto, dias_simulacao)
    vetor_geracao_fv_original[vetor_geracao_fv_original < 0] = 0

    if ATIVAR_SUAVIZACAO_FV and janela_suavizacao_passos > 1:
        series_fv = pd.Series(vetor_geracao_fv_original)
        vetor_geracao_fv_suavizada = series_fv.rolling(window=janela_suavizacao_passos, center=True, min_periods=1).mean().to_numpy()
    else:
        vetor_geracao_fv_suavizada = np.copy(vetor_geracao_fv_original)

    # --- 3. Loop Principal da Simulação (Lógica DETALHADA) ---
    vetor_potencia_bess = np.zeros(numero_de_passos)
    vetor_soc_kwh = np.zeros(numero_de_passos)
    vetor_gmg_potencia_despachada = np.zeros(numero_de_passos)
    vetor_gmgs_despachados = np.zeros(numero_de_passos)
    vetor_fv_para_carga = np.zeros(numero_de_passos)
    
    if numero_de_passos > 0:
        vetor_soc_kwh[0] = bess_soc_kwh

    for i in range(numero_de_passos):
        if i > 0:
            bess_soc_kwh = vetor_soc_kwh[i-1] 
            
        soc_percentual_atual = (bess_soc_kwh / bess_capacidade_kwh) * 100 if bess_capacidade_kwh > 1e-6 else 0
        potencia_carga_atual = vetor_carga[i]
        geracao_fv_bruta = vetor_geracao_fv_original[i]
        geracao_fv_meta = vetor_geracao_fv_suavizada[i]
        hora_do_dia = vetor_tempo[i] % 24

        bess_potencia_disponivel_carga = bess_potencia_max_kw * 0.8
        bess_potencia_disponivel_descarga = bess_potencia_max_kw
        potencia_bess_suavizacao = 0

        # Rampa de Carregamento
        fator_rampa_carga = 1.0 
        if soc_percentual_atual > SOC_RAMPA_INICIO:
            fator_rampa_carga = (SOC_LIMITE_MAX - soc_percentual_atual) / (SOC_LIMITE_MAX - SOC_RAMPA_INICIO)
            fator_rampa_carga = max(0, min(1, fator_rampa_carga))

        # Suavização FV
        if ATIVAR_SUAVIZACAO_FV and hora_do_dia >= 6 and hora_do_dia < 18:
            diferenca_fv = geracao_fv_bruta - geracao_fv_meta
            if diferenca_fv > 0: # Carregar BESS
                potencia_carregamento_alvo = min(diferenca_fv, bess_potencia_disponivel_carga)
                potencia_carregamento = potencia_carregamento_alvo * fator_rampa_carga
                espaco_disponivel_kwh = max(0, (bess_capacidade_kwh * SOC_LIMITE_MAX / 100) - bess_soc_kwh)
                energia_a_adicionar = (potencia_carregamento * passo_de_tempo_h) * EFICIENCIA_CARREGAMENTO
                energia_final_adicionada = min(energia_a_adicionar, espaco_disponivel_kwh)
                if energia_final_adicionada > 0:
                    bess_soc_kwh += energia_final_adicionada
                    potencia_bess_suavizacao = (energia_final_adicionada / EFICIENCIA_CARREGAMENTO) / passo_de_tempo_h
                    bess_potencia_disponivel_carga -= potencia_bess_suavizacao
            elif diferenca_fv < 0: # Descarregar BESS
                potencia_descarga = min(-diferenca_fv, bess_potencia_disponivel_descarga)
                soc_min_kwh_atual = bess_capacidade_kwh * (SOC_LIMITE_MIN_EMERGENCIA if potencia_carga_atual > carga_limite_emergencia else SOC_LIMITE_MIN_NORMAL) / 100
                energia_disponivel_kwh = max(0, bess_soc_kwh - soc_min_kwh_atual)
                energia_a_remover_bruta = (potencia_descarga * passo_de_tempo_h) / EFICIENCIA_DESCARREGAMENTO
                energia_final_removida = min(energia_a_remover_bruta, energia_disponivel_kwh)
                if energia_final_removida > 0:
                    bess_soc_kwh -= energia_final_removida
                    potencia_bess_suavizacao = -((energia_final_removida * EFICIENCIA_DESCARREGAMENTO) / passo_de_tempo_h)
                    bess_potencia_disponivel_descarga -= abs(potencia_bess_suavizacao)

        geracao_fv_para_despacho = geracao_fv_meta
        gmg_despacho_para_carga = 0
        bess_despacho_para_carga = 0
        bess_carga_pelo_fv = 0
        fv_despacho_para_carga = 0
        soc_percentual_atual = (bess_soc_kwh / bess_capacidade_kwh) * 100 if bess_capacidade_kwh > 1e-6 else 0
        bess_pode_ajudar = (soc_percentual_atual > SOC_LIMITE_MIN_NORMAL) or \
                         (potencia_carga_atual > carga_limite_emergencia and soc_percentual_atual > SOC_LIMITE_MIN_EMERGENCIA)
        
        # Despacho Noturno e Diurno
        if hora_do_dia < 6 or hora_do_dia >= 17 or geracao_fv_bruta <= 0: # Noite / Sem Sol
            if bess_pode_ajudar:
                if soc_percentual_atual > 75: gmg_meta_para_carga = 0.25 * potencia_carga_atual
                elif soc_percentual_atual > 60: gmg_meta_para_carga = 0.4 * potencia_carga_atual
                elif soc_percentual_atual > 50: gmg_meta_para_carga = 0.5 * potencia_carga_atual
                else: gmg_meta_para_carga = 0.6 * potencia_carga_atual
            else:
                gmg_meta_para_carga = potencia_carga_atual
            potencia_unitaria_a_usar = gmg_potencia_max_por_unidade
            capacidade_eficiente_total = numero_total_gmgs * gmg_potencia_max_por_unidade
            if not bess_pode_ajudar and gmg_meta_para_carga > capacidade_eficiente_total:
                potencia_unitaria_a_usar = gmg_potencia_unitaria
            gmgs_necessarios = np.ceil(gmg_meta_para_carga / potencia_unitaria_a_usar) if potencia_unitaria_a_usar > 0 else float('inf')
            vetor_gmgs_despachados[i] = min(numero_total_gmgs, gmgs_necessarios)
            gmg_despacho_para_carga = min(gmg_meta_para_carga, vetor_gmgs_despachados[i] * potencia_unitaria_a_usar)
            carga_restante = potencia_carga_atual - gmg_despacho_para_carga
            if bess_pode_ajudar:
                bess_despacho_para_carga = min(carga_restante, bess_potencia_disponivel_descarga)
        else: # Dia com Sol
            if geracao_fv_para_despacho >= (potencia_carga_atual * 0.85):
                gmg_meta_para_carga = 0.15 * potencia_carga_atual
                fv_despacho_para_carga = 0.85 * potencia_carga_atual
                excesso_fv_real = geracao_fv_bruta - fv_despacho_para_carga
                bess_carga_pelo_fv = max(0, excesso_fv_real)
            elif geracao_fv_para_despacho > 0 and soc_percentual_atual > 75:
                fv_despacho_para_carga = geracao_fv_para_despacho
                deficit = potencia_carga_atual - fv_despacho_para_carga
                bess_despacho_para_carga = 0.75 * deficit
                gmg_meta_para_carga = 0.25 * deficit
            else:
                fv_despacho_para_carga = geracao_fv_para_despacho
                gmg_meta_para_carga = potencia_carga_atual - fv_despacho_para_carga
                bess_despacho_para_carga = 0
            gmgs_necessarios = np.ceil(gmg_meta_para_carga / gmg_potencia_max_por_unidade) if gmg_potencia_max_por_unidade > 0 else float('inf')
            vetor_gmgs_despachados[i] = min(numero_total_gmgs, gmgs_necessarios)
            gmg_despacho_para_carga = min(gmg_meta_para_carga, vetor_gmgs_despachados[i] * gmg_potencia_max_por_unidade)
            deficit_final = potencia_carga_atual - fv_despacho_para_carga - gmg_despacho_para_carga
            bess_despacho_para_carga = max(bess_despacho_para_carga, deficit_final)

        # Contabilidade Final
        potencia_total_bess = potencia_bess_suavizacao
        
        # Carga BESS por excesso FV
        if bess_carga_pelo_fv > 0 and hora_do_dia < 17:
            potencia_carregamento_alvo_fv = min(bess_carga_pelo_fv, bess_potencia_disponivel_carga)
            potencia_carregamento_max_fv = potencia_carregamento_alvo_fv * fator_rampa_carga
            bess_soc_max_kwh = bess_capacidade_kwh * SOC_LIMITE_MAX / 100
            espaco_disponivel_kwh = max(0, bess_soc_max_kwh - bess_soc_kwh)
            energia_por_potencia = (potencia_carregamento_max_fv * passo_de_tempo_h) * EFICIENCIA_CARREGAMENTO
            energia_final_adicionada = min(energia_por_potencia, espaco_disponivel_kwh)
            if energia_final_adicionada > 0:
                bess_soc_kwh += energia_final_adicionada
                potencia_carregamento_bess_excesso = (energia_final_adicionada / EFICIENCIA_CARREGAMENTO) / passo_de_tempo_h
                potencia_total_bess += potencia_carregamento_bess_excesso

        # Descarga BESS para Carga
        if bess_despacho_para_carga > 0 and (bess_soc_kwh / bess_capacidade_kwh * 100 if bess_capacidade_kwh > 1e-6 else 0) > SOC_LIMITE_MIN_EMERGENCIA: # Usa limite de emergência como piso absoluto
            potencia_descarga_necessaria = min(bess_despacho_para_carga, bess_potencia_disponivel_descarga)
            energia_bruta_drenar = (potencia_descarga_necessaria * passo_de_tempo_h) / EFICIENCIA_DESCARREGAMENTO
            energia_final_drenada = min(energia_bruta_drenar, max(0, bess_soc_kwh - (bess_capacidade_kwh * SOC_LIMITE_MIN_EMERGENCIA / 100))) # Garante não descarregar abaixo do limite
            if energia_final_drenada > 0:
                bess_soc_kwh -= energia_final_drenada
                potencia_entregue_rede = (energia_final_drenada * EFICIENCIA_DESCARREGAMENTO) / passo_de_tempo_h
                potencia_descarga_bess_carga = -potencia_entregue_rede
                potencia_total_bess += potencia_descarga_bess_carga
                
        # Salvar Resultados do Passo
        vetor_fv_para_carga[i] = min(fv_despacho_para_carga, geracao_fv_bruta)
        vetor_gmg_potencia_despachada[i] = gmg_despacho_para_carga
        vetor_potencia_bess[i] = potencia_total_bess
        vetor_soc_kwh[i] = bess_soc_kwh

    return (
        vetor_tempo, vetor_carga, vetor_geracao_fv_original, vetor_geracao_fv_suavizada,
        vetor_gmg_potencia_despachada, vetor_potencia_bess, vetor_soc_kwh, vetor_gmgs_despachados,
        potencia_pico_fv_curto, numero_de_passos, vetor_fv_para_carga
    )


# --- Funções para Autonomia (Gráfico 2 - Lógica Simplificada) ---
# (Mantidas como estavam para o Gráfico 2)
def _simular_autonomia_interna(
    dias_simulacao, fator_irradiacao_fv, soc_inicial_kwh,
    potencia_pico_base_fv, bess_capacidade_kwh, bess_potencia_max_kw,
    numero_total_gmgs, gmg_potencia_max_por_unidade, gmg_potencia_unitaria, carga_limite_emergencia
):
    """Função interna para a simulação de longo prazo (lógica simplificada)."""
    
    numero_de_passos_longo = dias_simulacao * 24 * INTERVALOS_POR_HORA
    passo_de_tempo_h_longo = 1.0 / INTERVALOS_POR_HORA
    vetor_tempo_h = np.linspace(0, dias_simulacao * 24, numero_de_passos_longo, endpoint=False)
    vetor_tempo_dias = vetor_tempo_h / 24

    # Prepara Carga e FV
    carga_horaria_longa = (CARGA_HORARIA_24H * ((dias_simulacao // 1) + 1))[:dias_simulacao * 24]
    pontos_horarios_longa = np.arange(len(carga_horaria_longa))
    vetor_carga_longo = np.interp(vetor_tempo_h, pontos_horarios_longa, carga_horaria_longa)

    potencia_pico_fv = potencia_pico_base_fv * EFICIENCIA_FV * fator_irradiacao_fv

    # Geração de perfil FV com ruído (como estava antes)
    np.random.seed(42) 
    perfil_fv_24h_longo = np.zeros(24 * INTERVALOS_POR_HORA)
    for i, t in enumerate(np.linspace(0, 24, 24 * INTERVALOS_POR_HORA, endpoint=False)):
        hora_base = int(t)
        if hora_base in FATOR_GERACAO_HORARIA and (hora_base + 1) in FATOR_GERACAO_HORARIA:
            valor_inicial = FATOR_GERACAO_HORARIA[hora_base]
            valor_final = FATOR_GERACAO_HORARIA[hora_base + 1]
            fracao = t - hora_base
            valor_interpolado = valor_inicial + (valor_final - valor_inicial) * fracao
            ruido = np.random.normal(0, 0.08) 
            ruido = np.clip(ruido, -0.10, 0.10)
            valor_com_osc = valor_interpolado * (1 + ruido)
            perfil_fv_24h_longo[i] = max(0, valor_com_osc * potencia_pico_fv) 
        elif hora_base in FATOR_GERACAO_HORARIA:
            valor = FATOR_GERACAO_HORARIA[hora_base]
            ruido = np.random.normal(0, 0.08)
            ruido = np.clip(ruido, -0.10, 0.10)
            valor_com_osc = valor * (1 + ruido)
            perfil_fv_24h_longo[i] = max(0, valor_com_osc * potencia_pico_fv)

    vetor_geracao_fv_longo = np.tile(perfil_fv_24h_longo, dias_simulacao)
    vetor_geracao_fv_longo[vetor_geracao_fv_longo < 0] = 0

    # Inicialização do Rastreamento
    bess_soc_kwh_longo = soc_inicial_kwh
    tanque_diesel_litros = CAPACIDADE_TOTAL_DIESEL_L
    dia_fim_autonomia = None
    vetor_nivel_diesel = np.zeros(numero_de_passos_longo)

    for i in range(numero_de_passos_longo):
        soc_percentual_atual = (bess_soc_kwh_longo / bess_capacidade_kwh) * 100 if bess_capacidade_kwh > 1e-6 else 0
        potencia_carga_atual = vetor_carga_longo[i]
        geracao_fv_atual = vetor_geracao_fv_longo[i]
        hora_do_dia = vetor_tempo_h[i] % 24

        gmg_despacho_para_carga = 0
        bess_despacho_para_carga = 0
        bess_carga_pelo_fv = 0
        fv_despacho_para_carga = 0

        # Lógica de Despacho Simplificada
        bess_pode_ajudar = (soc_percentual_atual > SOC_LIMITE_MIN_NORMAL) or \
                         (potencia_carga_atual > carga_limite_emergencia and soc_percentual_atual > SOC_LIMITE_MIN_EMERGENCIA)

        if hora_do_dia < 6 or hora_do_dia >= 17 or geracao_fv_atual == 0:
            gmg_meta_para_carga = 0
            if bess_pode_ajudar:
                if soc_percentual_atual > 75: gmg_meta_para_carga = 0.25 * potencia_carga_atual
                elif soc_percentual_atual > 60: gmg_meta_para_carga = 0.4 * potencia_carga_atual
                elif soc_percentual_atual > 50: gmg_meta_para_carga = 0.5 * potencia_carga_atual
                else: gmg_meta_para_carga = 0.6 * potencia_carga_atual
            else: gmg_meta_para_carga = potencia_carga_atual
            
            potencia_unitaria_a_usar = gmg_potencia_max_por_unidade
            capacidade_eficiente_total = numero_total_gmgs * gmg_potencia_max_por_unidade
            if not bess_pode_ajudar and gmg_meta_para_carga > capacidade_eficiente_total:
                potencia_unitaria_a_usar = gmg_potencia_unitaria 
            
            gmgs_necessarios = np.ceil(gmg_meta_para_carga / potencia_unitaria_a_usar) if potencia_unitaria_a_usar > 0 else float('inf')
            gmg_despacho_para_carga = min(gmg_meta_para_carga, min(numero_total_gmgs, gmgs_necessarios) * potencia_unitaria_a_usar)
            carga_restante = potencia_carga_atual - gmg_despacho_para_carga
            if bess_pode_ajudar:
                bess_despacho_para_carga = min(carga_restante, bess_potencia_max_kw)
            else:
                bess_despacho_para_carga = 0
        else: # Período Diurno
            if geracao_fv_atual > 0:
                if geracao_fv_atual >= (potencia_carga_atual * 0.75):
                    gmg_meta_para_carga = 0.25 * potencia_carga_atual
                    fv_despacho_para_carga = 0.75 * potencia_carga_atual
                    bess_carga_pelo_fv = geracao_fv_atual - fv_despacho_para_carga
                elif geracao_fv_atual > 0 and soc_percentual_atual > 75:
                    fv_despacho_para_carga = geracao_fv_atual
                    deficit = potencia_carga_atual - fv_despacho_para_carga
                    bess_despacho_para_carga = 0.75 * deficit
                    gmg_meta_para_carga = 0.25 * deficit
                else:
                    fv_despacho_para_carga = geracao_fv_atual
                    gmg_meta_para_carga = potencia_carga_atual - fv_despacho_para_carga
                    bess_despacho_para_carga = 0
            gmgs_necessarios = np.ceil(gmg_meta_para_carga / gmg_potencia_max_por_unidade) if gmg_potencia_max_por_unidade > 0 else float('inf')
            gmg_despacho_para_carga = min(gmg_meta_para_carga, min(numero_total_gmgs, gmgs_necessarios) * gmg_potencia_max_por_unidade)
            deficit_final = potencia_carga_atual - fv_despacho_para_carga - gmg_despacho_para_carga
            bess_despacho_para_carga = max(bess_despacho_para_carga, deficit_final)
        
        # Rastreamento de Diesel
        consumo_diesel_lh = calcular_consumo_diesel(gmg_despacho_para_carga)
        gasto_passo_l = consumo_diesel_lh * passo_de_tempo_h_longo
        if tanque_diesel_litros <= 0.1:
            if dia_fim_autonomia is None:
                dia_fim_autonomia = vetor_tempo_dias[i]
            gmg_despacho_para_carga = 0
            gasto_passo_l = 0
        else:
            tanque_diesel_litros -= gasto_passo_l
        vetor_nivel_diesel[i] = max(0, tanque_diesel_litros)

        # Atualização do BESS (Simplificado)
        limite_carga_por_passo_kwh = bess_capacidade_kwh * 0.13 * passo_de_tempo_h_longo
        if bess_carga_pelo_fv > 0 and hora_do_dia < 15:
            potencia_carregamento_max_fv = min(bess_carga_pelo_fv, bess_potencia_max_kw)
            bess_soc_max_kwh = bess_capacidade_kwh * 0.90
            espaco_disponivel_kwh = max(0, bess_soc_max_kwh - bess_soc_kwh_longo)
            energia_por_potencia = (potencia_carregamento_max_fv * passo_de_tempo_h_longo) * EFICIENCIA_CARREGAMENTO
            energia_por_taxa = limite_carga_por_passo_kwh
            energia_final_adicionada = min(energia_por_potencia, espaco_disponivel_kwh, energia_por_taxa)
            if energia_final_adicionada > 0:
                bess_soc_kwh_longo += energia_final_adicionada
        elif bess_despacho_para_carga > 0 and soc_percentual_atual > 20:
            potencia_descarga_necessaria = min(bess_despacho_para_carga, bess_potencia_max_kw)
            energia_bruta_drenar = (potencia_descarga_necessaria * passo_de_tempo_h_longo) / EFICIENCIA_DESCARREGAMENTO
            energia_final_drenada = min(energia_bruta_drenar, bess_soc_kwh_longo)
            if energia_final_drenada > 0:
                bess_soc_kwh_longo -= energia_final_drenada
                
    return vetor_tempo_dias, vetor_nivel_diesel, dia_fim_autonomia

@st.cache_data
def run_long_term_simulation(
    potencia_pico_base_fv,
    p_ceu_aberto_slider, 
    bess_capacidade_kwh,
    bess_potencia_max_kw,
    numero_total_gmgs,
    gmg_potencia_unitaria,
    gmg_fator_potencia_eficiente,
    carga_limite_emergencia
):
    """
    Executa a simulação de longo prazo (Gráfico 2) para vários cenários.
    Usa a lógica SIMPLIFICADA de despacho.
    """
    cenarios_autonomia = {
        f'Dias Normais (Fator {p_ceu_aberto_slider:.2f})': p_ceu_aberto_slider,
        f'Dias Nublados (Fator {p_ceu_aberto_slider * 0.5:.2f})': p_ceu_aberto_slider * 0.5,
        f'Dia com Tempestade (Fator {p_ceu_aberto_slider * 0.2:.2f})': p_ceu_aberto_slider * 0.2,
        'Apenas GMG (Fator 0.0)': 0.0
    }
    resultados_autonomia = {}
    
    gmg_potencia_max_por_unidade = gmg_potencia_unitaria * gmg_fator_potencia_eficiente
    soc_inicial_longo_kwh = bess_capacidade_kwh * 0.90 

    for nome, fator in cenarios_autonomia.items():
        vetor_tempo_dias_longo, vetor_nivel_diesel, dia_fim_autonomia = \
            _simular_autonomia_interna(
                DIAS_SIMULACAO_LONGA, fator, soc_inicial_longo_kwh,
                potencia_pico_base_fv, bess_capacidade_kwh, bess_potencia_max_kw,
                numero_total_gmgs, gmg_potencia_max_por_unidade, gmg_potencia_unitaria, carga_limite_emergencia
            )
        
        resultados_autonomia[nome] = {
            'tempo': vetor_tempo_dias_longo,
            'nivel_diesel': vetor_nivel_diesel,
            'autonomia': dia_fim_autonomia
        }
    return resultados_autonomia


# --- NOVA Função de Simulação Diária (Lógica Detalhada para Gráfico 4) ---
#@st.cache_data # Cache pode ser útil aqui, mas vamos testar sem primeiro
def _simulate_one_day_diesel_detailed(
    fator_irradiacao_fv, soc_inicial_kwh,
    potencia_pico_base_fv, bess_capacidade_kwh, bess_potencia_max_kw,
    numero_total_gmgs, gmg_potencia_max_por_unidade, gmg_potencia_unitaria, carga_limite_emergencia
):
    """
    Simula 1 dia (24h) usando a LÓGICA DETALHADA (igual Gráfico 1) 
    e retorna o consumo TOTAL de diesel para esse dia.
    Usa perfil FV SEM ruído para análise de sensibilidade.
    """
    
    dias_simulacao = 1
    numero_de_passos_dia = dias_simulacao * 24 * INTERVALOS_POR_HORA
    passo_de_tempo_h_dia = 1.0 / INTERVALOS_POR_HORA
    vetor_tempo_h = np.linspace(0, dias_simulacao * 24, numero_de_passos_dia, endpoint=False)

    # Prepara Carga
    carga_horaria = CARGA_HORARIA_24H * dias_simulacao
    pontos_horarios = np.arange(len(carga_horaria))
    vetor_carga = np.interp(vetor_tempo_h, pontos_horarios, carga_horaria)

    # FV - Calcula potência e gera perfil SEM RUÍDO
    potencia_pico_fv = potencia_pico_base_fv * EFICIENCIA_FV * fator_irradiacao_fv
    perfil_fv_24h = np.zeros(24 * INTERVALOS_POR_HORA)
    for i, t in enumerate(np.linspace(0, 24, 24 * INTERVALOS_POR_HORA, endpoint=False)):
        hora_base = int(t)
        if hora_base in FATOR_GERACAO_HORARIA and (hora_base + 1) in FATOR_GERACAO_HORARIA:
            valor_inicial = FATOR_GERACAO_HORARIA[hora_base]
            valor_final = FATOR_GERACAO_HORARIA[hora_base + 1]
            fracao = t - hora_base
            valor_interpolado = valor_inicial + (valor_final - valor_inicial) * fracao
            perfil_fv_24h[i] = valor_interpolado * potencia_pico_fv # Sem ruído
        elif hora_base in FATOR_GERACAO_HORARIA:
            perfil_fv_24h[i] = FATOR_GERACAO_HORARIA[hora_base] * potencia_pico_fv # Sem ruído
            
    vetor_geracao_fv_original = np.tile(perfil_fv_24h, dias_simulacao)
    vetor_geracao_fv_original[vetor_geracao_fv_original < 0] = 0

    # Calcula FV Suavizado (Meta)
    janela_suavizacao_passos = int(JANELA_SUAVIZACAO_MINUTOS / (60 / INTERVALOS_POR_HORA))
    if ATIVAR_SUAVIZACAO_FV and janela_suavizacao_passos > 1:
        series_fv = pd.Series(vetor_geracao_fv_original)
        vetor_geracao_fv_suavizada = series_fv.rolling(window=janela_suavizacao_passos, center=True, min_periods=1).mean().to_numpy()
    else:
        vetor_geracao_fv_suavizada = np.copy(vetor_geracao_fv_original)

    # Inicialização da Simulação Diária
    bess_soc_kwh = soc_inicial_kwh
    total_diesel_consumido_litros = 0.0 # Acumulador

    # Loop de simulação detalhada para 1 dia
    for i in range(numero_de_passos_dia):
        soc_percentual_atual = (bess_soc_kwh / bess_capacidade_kwh) * 100 if bess_capacidade_kwh > 1e-6 else 0
        potencia_carga_atual = vetor_carga[i]
        geracao_fv_bruta = vetor_geracao_fv_original[i]
        geracao_fv_meta = vetor_geracao_fv_suavizada[i]
        hora_do_dia = vetor_tempo_h[i] % 24

        bess_potencia_disponivel_carga = bess_potencia_max_kw * 0.8
        bess_potencia_disponivel_descarga = bess_potencia_max_kw
        potencia_bess_suavizacao = 0

        # Rampa de Carregamento
        fator_rampa_carga = 1.0 
        if soc_percentual_atual > SOC_RAMPA_INICIO:
            fator_rampa_carga = (SOC_LIMITE_MAX - soc_percentual_atual) / (SOC_LIMITE_MAX - SOC_RAMPA_INICIO)
            fator_rampa_carga = max(0, min(1, fator_rampa_carga))

        # Suavização FV
        if ATIVAR_SUAVIZACAO_FV and hora_do_dia >= 6 and hora_do_dia < 18:
            diferenca_fv = geracao_fv_bruta - geracao_fv_meta
            if diferenca_fv > 0: # Carregar BESS
                potencia_carregamento_alvo = min(diferenca_fv, bess_potencia_disponivel_carga)
                potencia_carregamento = potencia_carregamento_alvo * fator_rampa_carga
                espaco_disponivel_kwh = max(0, (bess_capacidade_kwh * SOC_LIMITE_MAX / 100) - bess_soc_kwh)
                energia_a_adicionar = (potencia_carregamento * passo_de_tempo_h_dia) * EFICIENCIA_CARREGAMENTO
                energia_final_adicionada = min(energia_a_adicionar, espaco_disponivel_kwh)
                if energia_final_adicionada > 0:
                    bess_soc_kwh += energia_final_adicionada
                    potencia_bess_suavizacao = (energia_final_adicionada / EFICIENCIA_CARREGAMENTO) / passo_de_tempo_h_dia
                    bess_potencia_disponivel_carga -= potencia_bess_suavizacao
            elif diferenca_fv < 0: # Descarregar BESS
                potencia_descarga = min(-diferenca_fv, bess_potencia_disponivel_descarga)
                soc_min_kwh_atual = bess_capacidade_kwh * (SOC_LIMITE_MIN_EMERGENCIA if potencia_carga_atual > carga_limite_emergencia else SOC_LIMITE_MIN_NORMAL) / 100
                energia_disponivel_kwh = max(0, bess_soc_kwh - soc_min_kwh_atual)
                energia_a_remover_bruta = (potencia_descarga * passo_de_tempo_h_dia) / EFICIENCIA_DESCARREGAMENTO
                energia_final_removida = min(energia_a_remover_bruta, energia_disponivel_kwh)
                if energia_final_removida > 0:
                    bess_soc_kwh -= energia_final_removida
                    potencia_bess_suavizacao = -((energia_final_removida * EFICIENCIA_DESCARREGAMENTO) / passo_de_tempo_h_dia)
                    bess_potencia_disponivel_descarga -= abs(potencia_bess_suavizacao)

        geracao_fv_para_despacho = geracao_fv_meta
        gmg_despacho_para_carga = 0
        bess_despacho_para_carga = 0
        bess_carga_pelo_fv = 0
        fv_despacho_para_carga = 0
        soc_percentual_atual = (bess_soc_kwh / bess_capacidade_kwh) * 100 if bess_capacidade_kwh > 1e-6 else 0
        bess_pode_ajudar = (soc_percentual_atual > SOC_LIMITE_MIN_NORMAL) or \
                         (potencia_carga_atual > carga_limite_emergencia and soc_percentual_atual > SOC_LIMITE_MIN_EMERGENCIA)
        
        # Despacho Noturno e Diurno (Lógica Detalhada)
        if hora_do_dia < 6 or hora_do_dia >= 17 or geracao_fv_bruta <= 0: # Noite / Sem Sol
            if bess_pode_ajudar:
                if soc_percentual_atual > 75: gmg_meta_para_carga = 0.25 * potencia_carga_atual
                elif soc_percentual_atual > 60: gmg_meta_para_carga = 0.4 * potencia_carga_atual
                elif soc_percentual_atual > 50: gmg_meta_para_carga = 0.5 * potencia_carga_atual
                else: gmg_meta_para_carga = 0.6 * potencia_carga_atual
            else:
                gmg_meta_para_carga = potencia_carga_atual
            potencia_unitaria_a_usar = gmg_potencia_max_por_unidade
            capacidade_eficiente_total = numero_total_gmgs * gmg_potencia_max_por_unidade
            if not bess_pode_ajudar and gmg_meta_para_carga > capacidade_eficiente_total:
                potencia_unitaria_a_usar = gmg_potencia_unitaria
            gmgs_necessarios = np.ceil(gmg_meta_para_carga / potencia_unitaria_a_usar) if potencia_unitaria_a_usar > 0 else float('inf')
            # Não precisamos armazenar vetor_gmgs_despachados aqui
            num_gmgs_a_usar = min(numero_total_gmgs, gmgs_necessarios)
            gmg_despacho_para_carga = min(gmg_meta_para_carga, num_gmgs_a_usar * potencia_unitaria_a_usar)
            carga_restante = potencia_carga_atual - gmg_despacho_para_carga
            if bess_pode_ajudar:
                bess_despacho_para_carga = min(carga_restante, bess_potencia_disponivel_descarga)
        else: # Dia com Sol
            if geracao_fv_para_despacho >= (potencia_carga_atual * 0.75):
                gmg_meta_para_carga = 0.25 * potencia_carga_atual
                fv_despacho_para_carga = 0.75 * potencia_carga_atual
                excesso_fv_real = geracao_fv_bruta - fv_despacho_para_carga
                bess_carga_pelo_fv = max(0, excesso_fv_real)
            elif geracao_fv_para_despacho > 0 and soc_percentual_atual > 75:
                fv_despacho_para_carga = geracao_fv_para_despacho
                deficit = potencia_carga_atual - fv_despacho_para_carga
                bess_despacho_para_carga = 0.75 * deficit
                gmg_meta_para_carga = 0.25 * deficit
            else:
                fv_despacho_para_carga = geracao_fv_para_despacho
                gmg_meta_para_carga = potencia_carga_atual - fv_despacho_para_carga
                bess_despacho_para_carga = 0
            gmgs_necessarios = np.ceil(gmg_meta_para_carga / gmg_potencia_max_por_unidade) if gmg_potencia_max_por_unidade > 0 else float('inf')
            num_gmgs_a_usar = min(numero_total_gmgs, gmgs_necessarios)
            gmg_despacho_para_carga = min(gmg_meta_para_carga, num_gmgs_a_usar * gmg_potencia_max_por_unidade)
            deficit_final = potencia_carga_atual - fv_despacho_para_carga - gmg_despacho_para_carga
            bess_despacho_para_carga = max(bess_despacho_para_carga, deficit_final)

        # Contabilidade Final
        potencia_total_bess = potencia_bess_suavizacao
        
        # Carga BESS por excesso FV
        if bess_carga_pelo_fv > 0 and hora_do_dia < 17:
            potencia_carregamento_alvo_fv = min(bess_carga_pelo_fv, bess_potencia_disponivel_carga)
            potencia_carregamento_max_fv = potencia_carregamento_alvo_fv * fator_rampa_carga
            bess_soc_max_kwh = bess_capacidade_kwh * SOC_LIMITE_MAX / 100
            espaco_disponivel_kwh = max(0, bess_soc_max_kwh - bess_soc_kwh)
            energia_por_potencia = (potencia_carregamento_max_fv * passo_de_tempo_h_dia) * EFICIENCIA_CARREGAMENTO
            energia_final_adicionada = min(energia_por_potencia, espaco_disponivel_kwh)
            if energia_final_adicionada > 0:
                bess_soc_kwh += energia_final_adicionada
                # Não precisamos rastrear potência do BESS para o consumo de diesel

        # Descarga BESS para Carga
        if bess_despacho_para_carga > 0 and (bess_soc_kwh / bess_capacidade_kwh * 100 if bess_capacidade_kwh > 1e-6 else 0) > SOC_LIMITE_MIN_EMERGENCIA:
            potencia_descarga_necessaria = min(bess_despacho_para_carga, bess_potencia_disponivel_descarga)
            energia_bruta_drenar = (potencia_descarga_necessaria * passo_de_tempo_h_dia) / EFICIENCIA_DESCARREGAMENTO
            energia_final_drenada = min(energia_bruta_drenar, max(0, bess_soc_kwh - (bess_capacidade_kwh * SOC_LIMITE_MIN_EMERGENCIA / 100)))
            if energia_final_drenada > 0:
                bess_soc_kwh -= energia_final_drenada
                # Não precisamos rastrear potência do BESS para o consumo de diesel
                
        # --- Cálculo do Consumo de Diesel ---
        consumo_diesel_lh = calcular_consumo_diesel(gmg_despacho_para_carga)
        gasto_passo_l = consumo_diesel_lh * passo_de_tempo_h_dia
        total_diesel_consumido_litros += gasto_passo_l # Acumula

    # Retorna o consumo total de diesel acumulado para este dia
    return total_diesel_consumido_litros

# --- Função Atualizada para Calcular Consumo Anual (Chama a Simulação Detalhada) ---
@st.cache_data
def calculate_annual_diesel_consumption(
    potencia_pico_base_fv, bess_capacidade_kwh, bess_potencia_max_kw,
    numero_total_gmgs, gmg_potencia_unitaria, gmg_fator_potencia_eficiente, carga_limite_emergencia
):
    """
    Calcula o consumo anual ponderado de diesel usando a simulação DETALHADA.
    """
    gmg_potencia_max_por_unidade = gmg_potencia_unitaria * gmg_fator_potencia_eficiente
    # Começa cada dia de simulação com 50% SOC para um caso médio
    soc_inicial_kwh = bess_capacidade_kwh * 0.5 

    # Ponderação de dias no ano
    factors_and_weights = {
        1.0: 0.40, # 40% Céu Aberto
        0.5: 0.30, # 30% Nublado
        0.2: 0.20, # 10% Tempestade (Correção: Era 0.2 antes, ajustado para 0.1)
        0.0: 0.10  # 20% Sem Sol (Para somar 100%)
    }
    
    total_diesel_ponderado_diario = 0.0
    
    # Argumentos comuns passados para a simulação diária detalhada
    common_args = (
        potencia_pico_base_fv, bess_capacidade_kwh, bess_potencia_max_kw,
        numero_total_gmgs, gmg_potencia_max_por_unidade, gmg_potencia_unitaria, carga_limite_emergencia
    )

    # Chama a NOVA função _simulate_one_day_diesel_detailed
    for factor, weight in factors_and_weights.items():
        daily_diesel = _simulate_one_day_diesel_detailed(
            factor, soc_inicial_kwh, *common_args
        )
        total_diesel_ponderado_diario += daily_diesel * weight
        
    return total_diesel_ponderado_diario * 365


# ==============================================================================
# 4. FUNÇÕES DE PLOTAGEM
# ==============================================================================

def plot_graph_1(
    vetor_tempo, vetor_carga, vetor_geracao_fv_original, vetor_geracao_fv_suavizada,
    vetor_gmg_potencia_despachada, vetor_potencia_bess, vetor_soc_kwh, vetor_gmgs_despachados,
    potencia_pico_fv_curto, bess_capacidade_kwh, bess_potencia_max_kw, dias_simulacao
):
    """Gera o Gráfico 1: Curvas de Simulação de Curto Prazo"""
    
    figura1, eixos1 = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Gráfico de Potência
    eixos1[0].plot(vetor_tempo, vetor_carga, label='Consumo da Carga (kW)', color='royalblue', linewidth=2.5, zorder=10)
    eixos1[0].plot(vetor_tempo, vetor_geracao_fv_original, label='Geração FV Original (kW)', color='gold', alpha=0.9, linestyle=':', zorder=4)
    eixos1[0].fill_between(vetor_tempo, vetor_geracao_fv_suavizada, label='Geração FV Suavizada (Meta)', color='darkorange', linewidth=2.5, alpha= 0.3, zorder=5)
    eixos1[0].fill_between(vetor_tempo, vetor_gmg_potencia_despachada, color='gray', alpha=0.6, zorder=2, label='Potência GMG Despachada (kW)')
    eixos1[0].fill_between(vetor_tempo, 0, -vetor_potencia_bess, where=(vetor_potencia_bess >= 0), hatch='//', edgecolor='green', facecolor='lightgreen', alpha=0.7, label='BESS Carregando (kW)', zorder=3)
    eixos1[0].fill_between(vetor_tempo, 0, -vetor_potencia_bess, where=(vetor_potencia_bess < 0), hatch='\\', edgecolor='red', facecolor='lightcoral', alpha=0.7, label='BESS Descarregando (kW)', zorder=3)
    eixos1[0].set_ylabel('Potência (kW)', fontsize=12)
    # Ajuste no título para mostrar kWp base (antes da eficiência)
    potencia_fv_kwp_base_display = potencia_pico_fv_curto / EFICIENCIA_FV / (p_ceu_aberto if p_ceu_aberto > 0 else 1.0)
    eixos1[0].set_title(f'Simulação com Suavização FV | BESS: {bess_capacidade_kwh:.0f} kWh | PV: {potencia_fv_kwp_base_display:.0f} kWp ({dias_simulacao*24} Horas)', fontsize=16)
    eixos1[0].legend(loc='upper left')
    eixos1[0].axhline(0, color='black', linewidth=1)
    eixos1[0].set_ylim(-bess_potencia_max_kw * 1.1, None)

    # Gráfico de SOC
    eixos1[1].plot(vetor_tempo, (vetor_soc_kwh / (bess_capacidade_kwh + 1e-6)) * 100, label='SOC do BESS (%)', color='purple', linewidth=2) 
    eixos1[1].axhline(y=SOC_LIMITE_MAX, color='green', linestyle='--', linewidth=1.5, label=f'SOC Máximo ({SOC_LIMITE_MAX}%)')
    eixos1[1].axhline(y=SOC_RAMPA_INICIO, color='orange', linestyle=':', linewidth=2, label=f'Início da Rampa de Carga ({SOC_RAMPA_INICIO}%)')
    eixos1[1].axhline(y=SOC_LIMITE_MIN_NORMAL, color='red', linestyle='--', linewidth=1.5, label=f'SOC Mínimo Normal ({SOC_LIMITE_MIN_NORMAL}%)')
    eixos1[1].axhline(y=SOC_LIMITE_MIN_EMERGENCIA, color='darkred', linestyle=':', linewidth=2, label=f'SOC Mínimo Emergencial ({SOC_LIMITE_MIN_EMERGENCIA}%)')
    
    for i_hora in range(0, dias_simulacao * 24, 2):
        indice_passo = i_hora * INTERVALOS_POR_HORA
        if indice_passo < len(vetor_gmgs_despachados):
            # Garante que o valor seja um inteiro antes de formatar
            num_gmgs = int(np.nan_to_num(vetor_gmgs_despachados[indice_passo]))
            eixos1[1].text(i_hora, 5, f'{num_gmgs} GMGs', ha='center', va='bottom', fontsize=9, color='black', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.6))
    
    eixos1[1].set_xlabel('Hora', fontsize=12)
    eixos1[1].set_ylabel('Estado de Carga (%)', fontsize=12)
    eixos1[1].set_ylim(-5, 105)
    eixos1[1].legend(loc='lower right')

    plt.xticks(np.arange(0, dias_simulacao * 24 + 1, 2))
    plt.tight_layout(pad=2.0)
    
    return figura1

def plot_graph_2(resultados_autonomia):
    """Gera o Gráfico 2: Curvas de Autonomia de Diesel"""
    
    cores = ['green', 'orange', 'red', 'gray']
    figura2 = plt.figure(figsize=(18, 8))

    for (nome, resultado), cor in zip(resultados_autonomia.items(), cores):
        autonomia_valor = resultado['autonomia']
        rotulo = f"{nome}"
        if autonomia_valor is not None:
             rotulo += f" (Autonomia: {autonomia_valor:.2f} Dias)"
        plt.plot(resultado['tempo'], resultado['nivel_diesel'], label=rotulo, color=cor, linewidth=2)
        if autonomia_valor is not None:
            plt.plot(autonomia_valor, 0, marker='o', color=cor, markersize=8)

    plt.axhline(CAPACIDADE_TOTAL_DIESEL_L, color='black', linestyle='--', alpha=0.4, label='Capacidade Máxima do Tanque')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title(f'Análise de Autonomia do Diesel em {DIAS_SIMULACAO_LONGA} Dias (Cenários de Irradiação FV)', fontsize=16)
    plt.xlabel('Dias de Simulação', fontsize=12)
    plt.ylabel('Nível de Diesel (Litros)', fontsize=12)
    plt.xlim(0, DIAS_SIMULACAO_LONGA)
    plt.ylim(0, CAPACIDADE_TOTAL_DIESEL_L * 1.1)
    step_x_longo = max(1, DIAS_SIMULACAO_LONGA // 15)
    plt.xticks(np.arange(0, DIAS_SIMULACAO_LONGA + step_x_longo, step_x_longo))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    return figura2

def plot_graph_3(
    vetor_carga, vetor_fv_para_carga, vetor_gmg_potencia_despachada,
    vetor_potencia_bess, numero_de_passos, dias_simulacao
):
    """Gera o Gráfico 3: Gráfico de Barras da Composição da Carga (Média do 2º Dia)"""
    
    indice_inicio_dia2 = 24 * INTERVALOS_POR_HORA
    # Verifica se há dados suficientes para o dia 2
    if dias_simulacao >= 2 and numero_de_passos >= indice_inicio_dia2 + (24 * INTERVALOS_POR_HORA):
        slice_dia2 = slice(indice_inicio_dia2, indice_inicio_dia2 + (24 * INTERVALOS_POR_HORA))
        
        carga_dia2 = vetor_carga[slice_dia2]
        fv_para_carga_dia2 = vetor_fv_para_carga[slice_dia2]
        gmg_dia2 = vetor_gmg_potencia_despachada[slice_dia2]
        potencia_bess_total_dia2 = vetor_potencia_bess[slice_dia2]
        
        bess_descarga_dia2 = np.maximum(0, -potencia_bess_total_dia2)
        
        # Calcula médias horárias
        carga_horaria = carga_dia2.reshape(-1, INTERVALOS_POR_HORA).mean(axis=1)
        fv_carga_horaria = fv_para_carga_dia2.reshape(-1, INTERVALOS_POR_HORA).mean(axis=1)
        gmg_horaria = gmg_dia2.reshape(-1, INTERVALOS_POR_HORA).mean(axis=1)
        bess_descarga_horaria = bess_descarga_dia2.reshape(-1, INTERVALOS_POR_HORA).mean(axis=1)

        # Garante que temos 24 médias horárias
        if len(carga_horaria) == 24:
            horas_dia = np.arange(1, 25)
            figura3, eixos3 = plt.subplots(figsize=(18, 8))
            
            eixos3.bar(horas_dia, gmg_horaria, label='GMG', color='gray', alpha=0.8)
            eixos3.bar(horas_dia, fv_carga_horaria, bottom=gmg_horaria, label='FV para Carga', color='orange', alpha=0.8)
            eixos3.bar(horas_dia, bess_descarga_horaria, bottom=gmg_horaria + fv_carga_horaria, label='BESS (descarga)', color='crimson', alpha=0.8)
            
            eixos3.plot(horas_dia, carga_horaria, label='Carga Total Média', color='blue', linestyle='--', marker='o')

            eixos3.set_xlabel('Horas', fontsize=12)
            eixos3.set_ylabel('Potência Média (kW)', fontsize=12)
            eixos3.set_title('Composição Média do Atendimento da Carga (2º Dia)', fontsize=16)
            eixos3.set_xticks(horas_dia)
            eixos3.set_ylim(0, max(carga_horaria) * 1.2 if max(carga_horaria) > 0 else 100)
            eixos3.legend(loc='upper left')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            return figura3
        else:
             st.warning("Não foi possível calcular as médias horárias para o Gráfico 3.")
             return None
    else:
        # Se a simulação for menor que 2 dias, não plota o gráfico 3
        return None

def plot_graph_4(
    p_numero_total_gmgs, p_gmg_potencia_unitaria, p_gmg_fator_potencia_eficiente, p_carga_limite_emergencia
):
    """
    Gera o Gráfico 4: Análise de Sensibilidade do Consumo Anual de Diesel
    Eixo X: Capacidade BESS (kWh) [250 a 1250 kWh]
    Eixo Y: Consumo Anual de Diesel (L)
    Linhas: Diferentes tamanhos de Potência FV (kWp) [250 a 1250 kWp]
    """
    st.header("Gráfico 4: Análise de Sensibilidade (Consumo Anual de Diesel)")
    
    st.markdown("""
    **Como este gráfico é calculado:**
    1.  **Variação de BESS e FV:** Simulamos vários cenários alterando a **Capacidade do BESS (kWh)** no eixo X e a **Potência Pico do FV (kWp)** (cada linha representa um valor de FV). A potência do BESS (kW) é assumida como 50% da sua capacidade (0.5C).
    2.  **Lógica de Despacho Detalhada:** Para cada combinação, usamos a **mesma lógica de despacho detalhada dos Gráficos 1 e 3** (incluindo suavização FV, rampas, etc.) para simular o consumo de diesel.
    3.  **Tipos de Dia:** Para cada combinação de BESS e FV, calculamos o consumo de diesel para um dia típico em 4 condições diferentes de irradiação solar (Fator Céu Aberto), usando um perfil FV **sem ruído** para estabilidade:
        * Céu Aberto (Fator 1.0)
        * Nublado (Fator 0.5)
        * Tempestade (Fator 0.2)
        * Sem Sol (Fator 0.0)
    4.  **Média Anual Ponderada:** Assumimos uma distribuição anual desses dias:
        * 40% Céu Aberto
        * 30% Nublado
        * 10% Tempestade
        * 20% Sem Sol
        Calculamos a média diária de consumo de diesel usando esses pesos.
    5.  **Consumo Anual:** Multiplicamos a média diária ponderada por 365 para estimar o consumo anual total de diesel em Litros.
    """)
    
    # Adiciona um botão para iniciar a simulação demorada
    if st.button("Executar Análise de Sensibilidade (Gráfico 4)", key="run_sens_analysis"):
        with st.spinner("Executando análise de sensibilidade (Gráfico 4)... Isso pode levar alguns minutos."):
            fig, ax = plt.subplots(figsize=(14, 8))

            # --- NOVOS RANGES ---
            bess_range_kwh = np.linspace(250, 1250, 11) # 250, 350, ..., 1250 kWh
            fv_range_kwp = np.linspace(250, 1250, 5)   # 250, 500, 750, 1000, 1250 kWp

            total_sims = len(bess_range_kwh) * len(fv_range_kwp)
            progress_bar = st.progress(0.0)
            sim_count = 0

            for fv_kwp in fv_range_kwp:
                diesel_results = []
                # Feedback no terminal/log é útil, mas st.write pode poluir a UI
                # print(f"Simulando para FV = {fv_kwp:.0f} kWp...") 
                for bess_kwh in bess_range_kwh:
                    bess_kw = bess_kwh * 0.5 # Assume 0.5C
                    bess_kwh_safe = max(bess_kwh, 1e-6) # Garante não ser zero
                    bess_kw_safe = max(bess_kw, 1e-6)

                    # Chama a função cacheada que agora usa a lógica detalhada
                    diesel = calculate_annual_diesel_consumption(
                        fv_kwp, bess_kwh_safe, bess_kw_safe,
                        p_numero_total_gmgs, p_gmg_potencia_unitaria, 
                        p_gmg_fator_potencia_eficiente, p_carga_limite_emergencia
                    )
                    diesel_results.append(diesel)
                    
                    sim_count += 1
                    # Atualiza a barra de progresso dentro do loop interno
                    progress_bar.progress(sim_count / total_sims, text=f"Calculando... {sim_count}/{total_sims} cenários")


                ax.plot(bess_range_kwh, diesel_results, label=f'FV {fv_kwp:.0f} kWp', marker='o', markersize=5)

            progress_bar.empty() # Remove a barra de progresso

            ax.set_xlabel('Capacidade BESS (kWh)')
            ax.set_ylabel('Consumo Anual Estimado de Diesel (L)')
            ax.set_title('Consumo de Diesel vs. Dimensionamento BESS (variando FV)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x))) # Formato sem decimais
            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x))) # Formato sem decimais
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Clique no botão acima para gerar o Gráfico 4 (Análise de Sensibilidade). A execução pode demorar.")


# ==============================================================================
# 5. INTERFACE DO USUÁRIO (STREAMLIT SIDEBAR)
# ==============================================================================

st.sidebar.header("Parâmetros de Simulação")

st.sidebar.subheader("Geral")
p_dias_simulacao = st.sidebar.number_input(
    "Dias de Simulação (Gráficos 1 & 3)", 
    min_value=1, value=3, step=1,
    help="Duração da simulação de curto prazo."
)
p_carga_limite_emergencia = st.sidebar.number_input(
    "Carga Limite de Emergência (kW)", 
    min_value=50.0, value=100.0, step=10.0,
    help="Nível de carga que aciona o SOC de emergência do BESS."
)

st.sidebar.subheader("Sistema Fotovoltaico (FV)")
p_potencia_pico_base_fv = st.sidebar.number_input(
    "Potência Pico FV (kWp Base)", 
    min_value=0.0, value=450.0, step=10.0, # Permitir 0 para análise
    help="Potência de pico instalada do sistema FV."
)
p_ceu_aberto = st.sidebar.slider(
    "Fator Céu Aberto (Irradiação)", 
    min_value=0.0, max_value=1.0, value=1.0, step=0.05,
    help="Fator de ajuste da irradiação (1.0 = céu limpo, 0.0 = sem sol)."
)

st.sidebar.subheader("Bateria (BESS)")
p_bess_capacidade_kwh = st.sidebar.number_input(
    "Capacidade BESS (kWh)", 
    min_value=0.0, value=750.0, step=50.0 # Permitir 0 para análise
)
p_bess_potencia_max_kw = st.sidebar.number_input(
    "Potência BESS (kW)", 
    min_value=0.0, value=200.0, step=10.0 # Permitir 0 para análise
)
p_soc_inicial_percent = st.sidebar.slider(
    "SOC Inicial BESS (%)", 
    min_value=0.0, max_value=100.0, value=38.0, step=1.0,
    help="Estado de Carga inicial para a simulação de curto prazo."
)

st.sidebar.subheader("Geradores (GMG)")
p_numero_total_gmgs = st.sidebar.number_input(
    "Número de Geradores (GMG)", 
    min_value=1, value=10, step=1
)
p_gmg_potencia_unitaria = st.sidebar.number_input(
    "Potência Unitária GMG (kW)", 
    min_value=10.0, value=20.0, step=1.0
)
p_gmg_fator_potencia_eficiente = st.sidebar.slider(
    "Fator de Potência Eficiente GMG", 
    min_value=0.1, max_value=1.0, value=0.80, step=0.05,
    help="Fator de carga para operação eficiente do GMG."
)


# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL E PLOTAGEM
# ==============================================================================

# Converte o SOC inicial de % para fração
p_soc_inicial_fracao = p_soc_inicial_percent / 100.0

# Garante que a capacidade/potência não seja zero para evitar divisão por zero nos cálculos
# Usaremos estes valores seguros nas chamadas das funções
p_bess_capacidade_kwh_safe = max(p_bess_capacidade_kwh, 1e-6) 
p_bess_potencia_max_kw_safe = max(p_bess_potencia_max_kw, 1e-6)


# --- Executa Simulação de Curto Prazo ---
with st.spinner("Executando simulação de curto prazo..."):
    (
        vetor_tempo, vetor_carga, vetor_geracao_fv_original, vetor_geracao_fv_suavizada,
        vetor_gmg_potencia_despachada, vetor_potencia_bess, vetor_soc_kwh, vetor_gmgs_despachados,
        potencia_pico_fv_curto, numero_de_passos, vetor_fv_para_carga
    ) = run_short_term_simulation(
        p_dias_simulacao,
        p_potencia_pico_base_fv,
        p_ceu_aberto,
        p_bess_capacidade_kwh_safe, # Usa valor seguro
        p_bess_potencia_max_kw_safe,  # Usa valor seguro
        p_soc_inicial_fracao,
        p_numero_total_gmgs,
        p_gmg_potencia_unitaria,
        p_gmg_fator_potencia_eficiente,
        p_carga_limite_emergencia
    )

# --- Executa Simulação de Longo Prazo (Autonomia - Gráfico 2) ---
with st.spinner("Executando simulação de autonomia de longo prazo..."):
    resultados_autonomia = run_long_term_simulation(
        p_potencia_pico_base_fv,
        p_ceu_aberto, 
        p_bess_capacidade_kwh_safe, # Usa valor seguro
        p_bess_potencia_max_kw_safe,  # Usa valor seguro
        p_numero_total_gmgs,
        p_gmg_potencia_unitaria,
        p_gmg_fator_potencia_eficiente,
        p_carga_limite_emergencia
    )


# --- Plotagem dos Gráficos ---

st.header(f"Gráfico 1: Simulação de Operação ({p_dias_simulacao} Dias)")
fig1 = plot_graph_1(
    vetor_tempo, vetor_carga, vetor_geracao_fv_original, vetor_geracao_fv_suavizada,
    vetor_gmg_potencia_despachada, vetor_potencia_bess, vetor_soc_kwh, vetor_gmgs_despachados,
    potencia_pico_fv_curto, p_bess_capacidade_kwh_safe, p_bess_potencia_max_kw_safe, p_dias_simulacao
)
st.pyplot(fig1)

st.header("Gráfico 2: Análise de Autonomia de Diesel (Longo Prazo)")
fig2 = plot_graph_2(resultados_autonomia)
st.pyplot(fig2)

st.header("Gráfico 3: Composição Média do Atendimento (2º Dia)")
fig3 = plot_graph_3(
    vetor_carga, vetor_fv_para_carga, vetor_gmg_potencia_despachada, 
    vetor_potencia_bess, numero_de_passos, p_dias_simulacao # Passa dias_simulacao
)
if fig3:
    st.pyplot(fig3)
elif p_dias_simulacao >= 2:
     st.warning("Não foi possível gerar o Gráfico 3 (2º dia). Verifique os dados da simulação.")
else:
    st.warning("Simulação muito curta para gerar o gráfico do 2º dia. (Requer pelo menos 2 dias de simulação)")


# --- PLOTAGEM DO GRÁFICO 4 (Executado sob demanda) ---
plot_graph_4(
    p_numero_total_gmgs, 
    p_gmg_potencia_unitaria, 
    p_gmg_fator_potencia_eficiente, 
    p_carga_limite_emergencia
)
