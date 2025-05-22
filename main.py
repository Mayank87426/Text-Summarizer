from flask import Flask, render_template, request
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from collections import Counter
import random
from string import punctuation
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename
from googletrans import Translator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# PDF text extraction
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# Translator helper
def translate_text(text, target_lang='en'):
    if not text or target_lang == 'en':
        return text
    translator = Translator()
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# Summarizer
def summarizer(rawdocs, num_lines=5):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            word_freq[word.text] = word_freq.get(word.text, 0) + 1

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    sent_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq:
                sent_scores[sent] = sent_scores.get(sent, 0) + word_freq[word.text.lower()]

    all_sentences = sorted(sent_scores, key=sent_scores.get, reverse=True)
    random.shuffle(all_sentences)
    summary_sentences = all_sentences[:num_lines]
    final_summary = ' '.join([sent.text for sent in summary_sentences])
    return final_summary


# Keyword frequency for chart
def extract_keyword_frequencies(text, top_n=10):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    keywords = [token.text for token in doc if token.is_alpha and token.pos_ in ["NOUN", "PROPN"] and token.text not in STOP_WORDS]
    freq = Counter(keywords)
    return freq.most_common(top_n)


# MCQ generation
def generate_mcqs(text, num_questions=5):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    random.shuffle(sentences)
    mcqs = []
    fallbacks = ["concept", "element", "term", "method", "idea", "principle"]

    for sentence in sentences:
        if len(mcqs) >= num_questions:
            break

        sent_doc = nlp(sentence)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
        if not nouns:
            continue

        noun_counts = Counter(nouns)
        subject = noun_counts.most_common(1)[0][0]
        distractors = list(set(nouns) - {subject})

        if len(distractors) < 3:
            needed = 3 - len(distractors)
            distractors += random.sample(fallbacks, needed)

        question_stem = sentence.replace(subject, "______")
        answer_choices = random.sample(distractors, 3) + [subject]
        random.shuffle(answer_choices)
        correct_answer = chr(65 + answer_choices.index(subject))

        mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs


# Explanation generator
def generate_explanations(mcqs):
    explanations = []
    for q, choices, correct in mcqs:
        correct_index = ord(correct) - 65
        explanation = f"The correct answer is '{choices[correct_index]}', as it best completes the sentence: “{q.replace('______', choices[correct_index])}”"
        explanations.append(explanation)
    return explanations


# Topic extractor
def extract_main_topic(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    noun_phrases = [
        chunk.text.strip().lower() for chunk in doc.noun_chunks
        if chunk.root.pos_ not in ["PRON", "DET"]
        and chunk.root.text.lower() not in STOP_WORDS
        and chunk.text.lower() not in {"it", "this", "that", "they", "he", "she", "we", "i"}
        and len(chunk.text.strip().split()) <= 4
        and len(chunk.text.strip()) > 3
    ]
    if not noun_phrases:
        return "Main_topic"
    topic_counts = Counter(noun_phrases)
    best_topic = topic_counts.most_common(1)[0][0]
    return best_topic.replace(" ", "_").capitalize()


# ROUTES

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = ""
    if 'pdf_file' in request.files and request.files['pdf_file'].filename != '':
        pdf = request.files['pdf_file']
        filename = secure_filename(pdf.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf.save(pdf_path)
        text = extract_text_from_pdf(pdf_path)
    else:
        text = request.form.get('text', '')

    action = request.form.get('action')
    summary_lines = int(request.form.get('summary_lines', 5))
    num_questions = int(request.form.get('num_questions', 5))
    target_lang = request.form.get('language', 'en')

    if action == 'Summarize':
        summary = summarizer(text, num_lines=summary_lines)
        translated_summary = translate_text(summary, target_lang)
        return render_template('summary.html', summary=translated_summary, original_text=text, summary_lines=summary_lines, language=target_lang)

    elif action == 'MCQ Quiz Mode':
        mcqs = generate_mcqs(text, num_questions=num_questions)
        indexed_mcqs = [(i + 1, q, choices, ans) for i, (q, choices, ans) in enumerate(mcqs)]
        translated_mcqs = []
        for index, question, choices, correct_answer in indexed_mcqs:
            tq = translate_text(question, target_lang)
            tchoices = [translate_text(c, target_lang) for c in choices]
            translated_mcqs.append((index, tq, tchoices, correct_answer))
        main_topic = extract_main_topic(text)
        return render_template('quiz_mode.html', mcqs=translated_mcqs, original_text=text, num_questions=num_questions, main_topic=main_topic, language=target_lang)

    return render_template('index.html')


@app.route('/evaluate_mcqs', methods=['POST'])
def evaluate_mcqs():
    total = int(request.form['total'])
    target_lang = request.form.get('language', 'en')
    results = []
    questions_for_explanations = []

    for i in range(1, total + 1):
        user_answer = request.form.get(f'q{i}')
        correct_answer = request.form.get(f'correct_q{i}')
        topic = request.form.get(f'topic_q{i}')
        qtext = request.form.get(f'question_q{i}')
        choices = request.form.getlist(f'choices_q{i}')
        choices = [c for c in choices if c.strip()]
        is_correct = user_answer == correct_answer
        results.append({
            'question_no': i,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'topic': topic,
            'choices': choices,
            'question': qtext,
            'is_correct': is_correct,
        })
        questions_for_explanations.append((qtext, choices, correct_answer))

    main_topic = request.form.get('main_topic', 'Main_topic')
    wiki_link = f"https://en.wikipedia.org/wiki/{main_topic}"
    keyword_freqs = extract_keyword_frequencies(request.form.get('text', ''), top_n=10)
    explanations = generate_explanations(questions_for_explanations)
    translated_explanations = [translate_text(e, target_lang) for e in explanations]

    translated_results = []
    for r in results:
        r['question'] = translate_text(r['question'], target_lang)
        r['choices'] = [translate_text(c, target_lang) for c in r['choices']]
        translated_results.append(r)

    return render_template(
        'mcq_results.html',
        results=translated_results,
        score=sum([1 for r in translated_results if r['is_correct']]),
        total=total,
        wiki_link=wiki_link,
        keyword_freqs=keyword_freqs,
        explanations=translated_explanations
    )


@app.route('/refresh_summary', methods=['POST'])
def refresh_summary():
    text = request.form['text']
    summary_lines = int(request.form['summary_lines'])
    target_lang = request.form.get('language', 'en')
    summary = summarizer(text, num_lines=summary_lines)
    translated_summary = translate_text(summary, target_lang)
    return render_template('summary.html', summary=translated_summary, original_text=text, summary_lines=summary_lines, language=target_lang)


if __name__ == '__main__':
    app.run(debug=True)
