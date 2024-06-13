# OCR-englishlearning
English Learning app using pytesseract and streamlit

<a href="https://ocr-englishlearning-tlpyxbwodxhbjzsn8gvizy.streamlit.app/">Try my App!</a>

This is an application for creating a list of unknown words from English text.

① Upload an image containing English text.
② Use the left-hand bar to adjust the image so that it is horizontal.
③ If the background color is not uniform, check the "グレースケールで読み込む(Load in Grayscale)" option to standardize the background color.
④ After pressing the "読み取り(Read)" button, move to the "読み取り結果を確認)Confirm Reading Results" page.
⑤ Edit the text displayed in the "読み取り結果(Reading Results)" section as necessary.
⑥ Select the words.
⑦ Press the "単語リスト作成(Create Word List)" button to generate a list on the "単語リスト(Word List)" page.
You can repeat steps ①-⑥ to create a word list.

Using spaCy:
・Split the reading results into sentences for display.
・Exclude proper nouns and stop words from the selectable words.
・The word list is designed to output the lemma and the context-specific part of speech and meaning of each word.

Does not support handwritten characters yet...
