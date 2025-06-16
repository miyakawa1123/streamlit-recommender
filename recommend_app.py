# Streamlit版 有望人材レコメンドツール

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 仮の人材プロフィールデータ
data = [
    {"name": "鈴木 花子", "affiliation": "東京大学 松尾研究室", "skills": "自然言語処理, GPT, Python, 論文多数", "link": "https://researchmap.jp/hanako_suzuki", "summary": "自然言語処理に関する研究を行い、特に多言語対応のLLMに取り組む。論文も多数発表。"},
    {"name": "田中 一郎", "affiliation": "京都大学 情報学研究科", "skills": "画像認識, CNN, PyTorch", "link": "https://github.com/ichirotanaka", "summary": "CNNを用いた画像分類の研究に注力しており、医療画像処理などの応用にも携わる。"},
    {"name": "山田 太郎", "affiliation": "大阪大学 AI研究センター", "skills": "生成AI, Stable Diffusion, Python", "link": "https://researchmap.jp/taro_yamada", "summary": "画像生成モデルに関する研究で、多数のデモアプリを開発。Stable Diffusionに関心あり。"},
    {"name": "佐藤 美咲", "affiliation": "東京工業大学 数理・計算科学系", "skills": "最適化, 数理モデリング, Python", "link": "https://github.com/misaki-sato", "summary": "企業との共同研究で物流最適化問題を扱い、実践的な数理モデリングに強み。"},
    {"name": "中村 翼", "affiliation": "東北大学 情報科学研究科", "skills": "強化学習, ロボティクス, C++", "link": "https://researchmap.jp/tsubasa_nakamura", "summary": "強化学習を用いたロボット制御が専門。ROSやC++を用いた実機実験も経験豊富。"},
    {"name": "小林 結衣", "affiliation": "早稲田大学 基幹理工学部", "skills": "データ分析, 機械学習, R, Python", "link": "https://github.com/yui-kobayashi", "summary": "ビジネスデータを用いた機械学習モデルの構築経験があり、Kaggle参加経験あり。"},
    {"name": "松本 大地", "affiliation": "九州大学 統計科学センター", "skills": "統計解析, 回帰分析, R", "link": "https://researchmap.jp/daichi_matsumoto", "summary": "回帰モデルや因子分析など統計解析を得意とし、教育分野への応用にも取り組む。"},
    {"name": "高橋 直人", "affiliation": "名古屋大学 情報科学研究科", "skills": "自然言語処理, 機械翻訳, BERT", "link": "https://github.com/naoto-takahashi", "summary": "機械翻訳システムの評価と改良に取り組み、BERTベースのモデル構築経験あり。"},
    {"name": "林 さくら", "affiliation": "北海道大学 AI研究所", "skills": "時系列解析, LSTM, Python", "link": "https://researchmap.jp/sakura_hayashi", "summary": "金融データやIoTデータを用いた時系列解析の研究に従事。LSTMによる予測モデル構築経験あり。"},
    {"name": "青木 健", "affiliation": "筑波大学 システム情報系", "skills": "クラスタリング, 教師なし学習, 可視化", "link": "https://github.com/ken-aoki", "summary": "教師なし学習によるユーザー行動のセグメンテーションと可視化技術に取り組む。"}
]

df = pd.DataFrame(data)
df["combined_text"] = df["skills"] + " " + df["summary"]

# Streamlit UI
st.title("🔍 有望人材レコメンドツール")
st.markdown("人材要件（例: 自然言語処理 GPT 多言語対応）を入力してください。")

user_input = st.text_input("人材要件を入力", "自然言語処理 GPT 多言語対応")

if user_input:
    corpus = df["combined_text"].tolist() + [user_input]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    user_vector = tfidf_matrix[-1]
    profile_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(user_vector, profile_vectors).flatten()

    df["match_score"] = similarities
    result = df.sort_values(by="match_score", ascending=False)

    st.subheader("📋 レコメンド結果")
    st.dataframe(result[["name", "affiliation", "skills", "match_score", "link"]])

