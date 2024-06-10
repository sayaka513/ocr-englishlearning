import streamlit as st
import cv2
import pytesseract
from PIL import Image
import spacy
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
import pandas as pd


# spaCyの英語モデルをロード
nlp = spacy.load("en_core_web_sm")

# データのダウンロード
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ストップワードのリストを取得
stop_words = set(stopwords.words('english'))

# ページ設定
st.set_page_config(
    page_title="For English learning using OCR",
    layout="wide",
    initial_sidebar_state="expanded"
)


tab1, tab2, tab3 = st.tabs(["アップロード", "読み取り結果を確認", "単語リスト"])
ocr_text = None
selected_words_list = None

# サイドバー
st.sidebar.header("使い方")
st.sidebar.markdown("""
① 画像をアップロード
""")

# 回転
st.sidebar.markdown("""
② 水平になるように角度を調整
""")
rotation_angle = st.sidebar.slider("回転角度", 0, 360, 0)

# グレースケールで読み込むチェックボックス
st.sidebar.markdown("""
③ 背景の色が一定ではない場合、チェックを入れてください
""")
grayscale = st.sidebar.checkbox("グレースケールで読み込む", value=False)

# C値
if grayscale:
    C = st.sidebar.slider("画像の明るさ調整", 0, 50, 20, help="なるべく背景が均一になるように明るさを調整してください")

# グレースケールモードの場合、他のスライダーを表示
#if grayscale:
#    clip_limit = st.sidebar.slider("明るさ調整", 1.0, 10.0, 2.0)# CLAHE クリップ制限
#    tile_grid_size = st.sidebar.slider("細部調整", 1, 16, 8)# CLAHE タイルグリッドサイズ
#    threshold_value = st.sidebar.slider("コントラスト設定", 0, 255, 127)# 二値化の閾値
#    median_blur_ksize = st.sidebar.slider("ぼかし設定", 1, 15, 5, step=2)# メディアンフィルタのカーネルサイズ
#    dilation_iterations = st.sidebar.slider("輪郭強調設定", 0, 5, 1)# 膨張の回数
#    erosion_iterations = st.sidebar.slider("輪郭弱調設定", 0, 5, 1)# 侵食の回数

with tab1:

# 画像のアップロード
    uploaded_file = st.file_uploader(r"$\textsf{\large 英文画像をアップロード}$", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
            # 画像を読み込む（グレースケールモードで）
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            if grayscale:
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

                # 画像の高さと幅の取得
                (height, width) = img.shape[:2]
                # 画像スケールに対してブロックサイズを計算
                blockSize = int(min(width, height) * 0.05)  # ブロックサイズは画像の小さい方の辺の5%
                if blockSize % 2 == 0:
                    blockSize += 1  # ブロックサイズは奇数

                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)

                # CLAHEを適用
                #clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
                #img = clahe.apply(img)

                # 二値化
                #_, img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

                # ノイズ除去
                #img = cv2.medianBlur(img, median_blur_ksize)

                # 膨張と侵食
                #kernel = np.ones((5, 5), np.uint8)
                #img = cv2.dilate(img, kernel, iterations=dilation_iterations)
                #img = cv2.erode(img, kernel, iterations=erosion_iterations)
            else:
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                (height, width) = img.shape[:2]

            # 回転
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            img = cv2.warpAffine(img, M, (width, height))

            # 表示
            st.image(Image.fromarray(img), width=500)

        # OCR適用ボタン
            if st.button("読み取り"):
                custom_config = r' -c preserve_interword_spaces=1'
                ocr_text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
                # 改行をスペースに変換
                st.session_state["ocr_text"] = ocr_text.replace("\n", " ")
                st.success("「読み取り結果」を確認してください")

    else:
        st.warning("画像をアップロードしてください")


with tab2:

    st.write(r"$\textsf{\normalsize 必要に応じて文章を編集してください：}$")
    if uploaded_file is not None and "ocr_text" in st.session_state:
        edited_text = st.text_area("【読み取り結果】", st.session_state.get("ocr_text"), height=300)
        st.session_state["edited_text"] = edited_text  # ユーザーの編集内容を保存

        # 文単位に分割
        doc = nlp(edited_text)
        sentences = list(doc.sents)

        for sentence in sentences:
            st.markdown(f"> {sentence.text}")

        # 罫線を挿入する
        st.divider()

        # ストップワード、アルファベットのみで構成されていない単語、固有名詞を削除
        words_with_sense = []
        for token in doc:
            if token.text.lower() not in stop_words and token.is_alpha and token.pos_ != "PROPN":
            # 文脈に応じた単語の意味を特定し、意味のない単語を削除
                sense = lesk(edited_text.split(), token.text)
                if sense:
                    words_with_sense.append(token.text)

        # ユーザーが単語を選択
        selected_words = st.multiselect(r"$\textsf{\large 単語を選択してください：}$", words_with_sense, placeholder="選択")

        if selected_words:
            word_data = []  # 選択された単語の情報を保持
            for selected_word in selected_words:
                token = nlp(selected_word)[0]
                # 原形を取得
                lemma = token.lemma_
                # 文脈に応じた単語の意味を特定
                sense = lesk(edited_text.split(), token.text)

                # ユーザーが選択した単語の意味を表示
                definition = sense.definition()

                word_data.append({"単語": selected_word, "原型": lemma, "品詞": token.pos_, "意味": definition})

                if 'selected_words' not in st.session_state:
                    st.session_state.selected_words = []

                #selected_words_list = []
            word_data_df = pd.DataFrame(word_data)
            st.table(word_data_df)
            if st.button("単語リスト作成"):
                #st.session_state.selected_words.append(selected_word, token.pos_, definition])
                for data in word_data:
                    st.session_state.selected_words.append(data)
                st.success("選択した単語のリストを作成しました")

    else:
        st.warning("画像を読み取ってください")



with tab3:
    if 'selected_words' in st.session_state and st.session_state.selected_words:
        selected_words_df = pd.DataFrame(st.session_state.selected_words)
        #st.data_editor(selected_words_df, disabled=("単語","原型","品詞","意味"), num_rows="dynamic")
        edited_df = st.data_editor(selected_words_df, disabled=("単語","原型","品詞","意味"), num_rows="dynamic", use_container_width=True)

        # 削除された行を検出
        if len(edited_df) < len(selected_words_df):
            deleted_rows = selected_words_df[~selected_words_df.isin(edited_df)].dropna(how='all')
            st.session_state.selected_words = edited_df.to_dict('records')

    else:
        st.warning("単語を保存してください")
