# OCR-Englishlearning App
English Learning app using pytesseract and streamlit<br>
<br>
<a href="https://ocr-englishlearning-tlpyxbwodxhbjzsn8gvizy.streamlit.app/">Try my App!</a>
<br>
An application for creating a list of unknown words from English text.<br>
This is useful when you read an English text on paper and there are several words you don't understand.<br>
<br>
① Upload an image containing English text.<br>
② Use the left-hand bar to adjust the image so that it is horizontal.<br>
③ If the background color is not uniform, check the "グレースケールで読み込む(Load in Grayscale)" option to standardize the background color.<br>
④ After pressing the "読み取り(Read)" button, move to the "読み取り結果を確認)Confirm Reading Results" page.<br>
⑤ Edit the text displayed in the "読み取り結果(Reading Results)" section as necessary.<br>
⑥ Select the words.<br>
⑦ Press the "単語リスト作成(Create Word List)" button to generate a list on the "単語リスト(Word List)" page.<br>
You can repeat steps ①-⑥ to create a word list.<br>
<br>
Using spaCy:<br>
・Split the reading results into sentences for display.<br>
・Exclude proper nouns and stop words from the selectable words.<br>
・The word list is designed to output the lemma and the context-specific part of speech and meaning of each word.<br>
<br>
In the future, I aim to support handwritten characters and Japanese OCR.
