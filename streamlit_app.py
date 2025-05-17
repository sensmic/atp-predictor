
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Ładowanie danych
@st.cache_data
def load_data():
    df = pd.read_csv("atp_matches_2023.csv")
    df = df.dropna(subset=['winner_rank', 'loser_rank', 'winner_age', 'loser_age'])
    df['higher_rank_winner'] = (df['winner_rank'] < df['loser_rank']).astype(int)
    df['rank_diff'] = df['loser_rank'] - df['winner_rank']
    df['age_diff'] = df['loser_age'] - df['winner_age']
    return df

df = load_data()
X = df[['rank_diff', 'age_diff']]
y = df['higher_rank_winner']

# Trening modelu
model = LogisticRegression()
model.fit(X, y)

# Interfejs użytkownika
st.title("🎾 Prognoza wyniku meczu tenisowego (ATP)")

st.markdown("Podaj dane dwóch zawodników, a model przewidzi szanse zwycięstwa zawodnika oraz sprawdzi czy warto postawić zakład (value bet).")

rank_diff = st.number_input("Różnica rankingów (przeciwnik - zawodnik):", value=50)
age_diff = st.number_input("Różnica wieku (przeciwnik - zawodnik):", value=2)
kurs = st.number_input("Kurs bukmachera na zwycięstwo zawodnika:", value=2.10)

if st.button("Oblicz"):
    proba = model.predict_proba([[rank_diff, age_diff]])[0][1]
    value = proba * kurs - 1

    st.write(f"📊 **Szansa na wygraną (model):** {proba:.2%}")
    st.write(f"💸 **Expected Value (EV):** {value:.2f}")
    if value > 0:
        st.success("✅ To jest value bet!")
    else:
        st.warning("⚠️ Brak value. Nie warto grać.")
