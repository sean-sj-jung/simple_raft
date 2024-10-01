import datasets
from datasets import Dataset
import random, os
import pypdf
from api_openai import chat as ChatCompletion


def read_document(data_path: str) -> dict:
    """
    Reads PDF documents from the specified directory and extracts text.

    Args:
        data_path (str): Path to the directory containing PDF files.

    Returns:
        dict: A dictionary with document filenames as keys and extracted text as values.
    """

    docs = [x for x in os.listdir(data_path) if x.endswith('pdf')]
    contexts = {}

    for doc in docs:
        if doc.endswith('pdf'):
            reader = PyPDF2.PdfReader('/'.join([data_path, doc]))
            text = [reader.pages[x].extract_text() for x in range(len(reader.pages))]
        contexts[doc] = '\n'.join(text)

    return contexts


def generate_question(context: str, num_question: int = 5) -> str:
    """
    Generates questions based on a given context using a chat model.

    Args:
        context (str): The context from which questions are generated.
        num_question (int): The number of questions to generate.

    Returns:
        str: Generated questions as a formatted string.
    """

    prompt_header = f""" You are a synthetic question generator.
    Instructions:
    - Given a chunk of context about some topic(s), generate {num_question} example questions a user could ask
    - Questions should be answerable using only information from the chunk.
    - Generate one question per line
    - Generate only questions
    - Questions should be succinct

    Here are some samples:
    Context: GPT-4o is a step towards much more natural human-computer interaction. It accepts as input any combination of text, audio, image, and video \
        and generates any combination of text, audio, and image outputs. It can respond to audio inputs in as little as 232 milliseconds, with an average \
        of 320 milliseconds, which is similar to human response time in a conversation. It matches GPT-4 Turbo performance on text \
        in English and code, with significant improvement on text in non-English languages, while also being much faster and 50% cheaper in the API. \
        GPT-4o is especially better at vision and audio understanding compared to existing models.\
        Prior to GPT-4o, you could use Voice Mode to talk to ChatGPT with latencies of 2.8 seconds (GPT-3.5) and 5.4 seconds (GPT-4) on average. \
        To achieve this, Voice Mode is a pipeline of three separate models: one simple model transcribes audio to text, GPT-3.5 or GPT-4 takes in text \
        and outputs text, and a third simple model converts that text back to audio. This process means that the main source of intelligence, GPT-4, \
        loses a lot of information. It can't directly observe tone, multiple speakers, or background noises, and it can't output laughter, singing, or express emotion.\
        With GPT-4o, we trained a single new model end-to-end across text, vision, and audio, meaning that all inputs and outputs are processed by the same neural network. \
        Because GPT-4o is our first model combining all of these modalities, we are still just scratching the surface of exploring what the model can do \
        and its limitations.
    Question: 
    \nWhat input types is GPT-4o capable of handling?
    \nOn average, what is the output latency a user can expect?
    \nIs it more expensive than the previous models?
    \nHow is it different from the Voice Mode of older models?
    \nDoes GPT-4o support languages other than English?

    Context: {context}
    Question:
    """

    input = prompt_header.format(context=context)
    resp = ChatCompletion(input)

    return resp.choices[0].message.content


def generate_answer(context: str, question: str) -> str:
    """
    Generates an answer to a given question based on the context using a chat model.

    Args:
        context (str): The context containing the information for answering the question.
        question (str): The question to answer.

    Returns:
        str: The answer generated from the context and question.
    """

    prompt = f"""
    Question: {question}
    Context: {context}
    
    Answer this question using the information given in the context above.
    
    Instructions:
    - Provide step-by-step reasoning on how to answer the question.
    - Explain which parts of the context are meaningful and why.
    - Copy paste the relevant sentences from the context in ##begin_quote## and ##end_quote##.
    - Provide a summary of how you reached your answer.
    - End your response with the final answer in the form <ANSWER>: $answer, the answer should be succinct.
    - You MUST begin your final answer with the tag "<ANSWER>:".
    """

    answer = ChatCompletion(prompt)
    return answer.choices[0].message.content


if __name__ == "__main__":
    """
    Create a dataset from documents in a specified directory.
    
    Process:
    1. Read all documents from DATA_PATH and extract their contents.
    2. Generate questions from the extracted contexts.
    3. Generate answers to the questions using the original document as the oracle.
    4. Create a datasets.Dataset object with questions, answers, oracle context, and distractor documents.
    """

    DATA_PATH = './documents'
    num_question = 5
    num_distractor = 3

    # Read documents and extract contexts
    contexts = read_document(data_path=DATA_PATH)
    filenames = contexts.keys()
    ds = None
    for oracle_file in filenames:
        oracle_context = contexts[oracle_file]

        # Generate questions based on the oracle context        
        five_q = generate_question(oracle_context)
        five_q = [x.strip() for x in five_q.split('\n') if x.strip()]
        for q in five_q:
            temp_dict = {
                "question":q,
                "oracle_context":oracle_context,
                "context":None,
                "cot_answer":None
            }

            # Select distractor documents
            candidates = list(filenames)
            candidates.remove(oracle_file)
            distractor_files = random.sample(candidates, num_distractor)

            # Collect oracle context and distractor contexts
            docs = [oracle_context]
            for d_file in distractor_files:
                docs.append(contexts[d_file])
            random.shuffle(docs)

            # Prepare context data structure
            d = {
                "title": [],
                "sentences": []
            }
            d['title'].append(["placeholder"]*(num_distractor+1))
            d['sentences'].append(docs)
            temp_dict['context'] = d

            # Generate and save Chain of Thought (CoT) answer
            answer = generate_answer(context=oracle_context, question=q)
            temp_dict['cot_answer'] = answer

            # Construct model instruction with document tags
            inst_text = ""
            for doc in docs:
                inst_text += '\n'.join(["<DOCUMENT>", doc, "</DOCUMENT>"])
            inst_text += '\n'+q
            temp_dict['instruction'] = inst_text

            # Save into dataset
            if not ds:
                # Create new dataset if it doesn't exist yet                
                temp_dict['question'] = [temp_dict['question']]
                temp_dict['context'] = [temp_dict['context']]
                temp_dict['oracle_context'] = [temp_dict['oracle_context']]
                temp_dict['cot_answer'] = [temp_dict['cot_answer']]
                temp_dict['instruction'] = [temp_dict['instruction']]
                ds = Dataset.from_dict(temp_dict)
            else:
                # Append to existing dataset                
                ds = ds.add_item(temp_dict)
                
    # Save the final dataset to disk
    ds.save_to_disk('dataset4raft.hf')