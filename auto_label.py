from openai import OpenAI
import json
import sys
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")


# Function to read a text file and return its content
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_start_idx(context, substring):
    pass

# Function to create a SQuAD format annotation from user inputs
def create_squad_annotation(title, text, questions_and_answers, sentences):
    squad_data = {
        "data": [
            {
                "title": title,
                "paragraphs": [
                    {
                        "context": text,
                        "qas": []
                    }
                ]
            }
        ]
    }

    for qa in questions_and_answers:
        question, answer = qa
        
        print("gpt4 question: ", question)
        print("gpt4 answer: ", answer)
        
        ans_start = text.find(answer)
        
        # first word use lowercase
        if ans_start == -1:
          
          lowercase_answer = answer[0].lower() + answer[1:]
          ans_start = text.find(lowercase_answer)
          
          
          # determine by sentence similarity (spacy)
          if ans_start == -1:
              
              similarities = []
              
              target_doc = nlp(answer)
              for sentence in sentences:
                  
                  doc = nlp(sentence)
                  similarity = doc.similarity(target_doc)
                  similarities.append(similarity)
              
              max_similarity = max(similarities)
              max_similarity_index = similarities.index(max_similarity)
              
              
              print("similarity: ", max(similarities))
              print("transcript: ", sentences[max_similarity_index])
              
              if max_similarity >= 0.85:
                  
                  ans_start = text.find(sentences[max_similarity_index])
                  
              else:
                  print("Skip this error gpt response")
                  ans_start = -1
              
              
              
          else:
              print("Find! The first word in transcript is lowercase.")
              
        else:
            print("Find! Completely match.")
              
              
        print()
        
        if ans_start == -1:
            continue

        
        squad_data["data"][0]["paragraphs"][0]["qas"].append({
            "question": question,
            "id": str(len(squad_data["data"][0]["paragraphs"][0]["qas"]) + 1),
            "answers": [
                {
                    "text": answer,
                    "answer_start": ans_start
                }
            ]
        })

    return squad_data

# post-processing function, if there are quotation marks then remove them
def remove_outer_quotes(input_str):
    if (input_str.startswith('"') and input_str.endswith('"')) or (input_str.startswith("'") and input_str.endswith("'")):
        return input_str[1:-1]
    else:
        return input_str
    


# Main function
def main(folder, title, number):
    
    # use your own key
    api_key = "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"
    client = OpenAI(api_key=api_key)
    model_id = "gpt-4-1106-preview"
        
    file_path = f"C:/Users/ee527/Desktop/Big Data Analytics/final_project/podcast_transcript/{folder}/{title}.txt"
    output_path = f"C:/Users/ee527/Desktop/Big Data Analytics/final_project/podcast_transcript/{folder}/{title}.json"
    
    with open(file_path, 'r') as file:
        file_contents = file.read()
        
    sentences = file_contents.split(". ")
    
    print("number of sentences: ", len(sentences))

    
    # prompt here

    messages = [
    
        {"role": "user", "content": f"Read this transcript first: [{file_contents}]. Then give me {number} questions and the complete reference sentence you find in the transcript. Remember, there are some advertisement inside it, please recognize them and dont use those information to ask question. Remember, musn't abbreviate or paraphrase the reference sentence, the refernce sentence must be completely same in the transcript. Lastly, the response should be this format: Question: Who is Donald? Reference: I dont know. Question: Where does he live? Reference: He lives in NYC"}

    ]

    print()
    print("Running GPT-4 ==================================\n")

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )

    response = completion.choices[0].message.content
    # print(response)



    lines = [line.strip() for line in response.split("\n") if line.strip()]


    # post-processing logic here
    questions_and_answers = []
    for line in lines:
        
        print(line)
        

        if line[-1] == "?":
            # print("q")
            
            colon_index = line.index(':')

            # Extract the sentence after the first colon
            question = line[colon_index + 1:].strip()
            
            
        elif "Reference" in line:
            # print("a")
            
            colon_index = line.index(':')

            # Extract the sentence after the first colon
            answer = line[colon_index + 1:].strip()
            answer = remove_outer_quotes(answer)
            answer = answer[:-1]
            questions_and_answers.append((question, answer))
            
        else:
            continue
     
        
    print()
    print("Post-processing ==================================\n")
    # Create SQuAD format annotation
    squad_data = create_squad_annotation(title, file_contents, questions_and_answers,sentences)

    # Output to a JSON file

    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(squad_data, output_file, indent=2)

    print(f"SQuAD format annotation saved to {output_path}")
  
  
if __name__ == "__main__":
    
    
    # sub folder number, transcript name, number of questions
    main("4","5 million In damages", 15)




