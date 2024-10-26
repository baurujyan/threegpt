from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
import time
from typing import List, Dict

app = Flask(__name__)

# Add this new route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Add this new route for the About page
@app.route('/about')
def about():
    return render_template('about.html')

# Update API configuration for Groq
try:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    if not GROQ_API_KEY:
        with open('config.txt', 'r') as f:
            GROQ_API_KEY = f.read().strip()
    print(f"API Key loaded: {GROQ_API_KEY[:8]}...")
    
    # Initialize OpenAI client with Groq configuration
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
except Exception as e:
    print(f"Error loading API key: {str(e)}")
    exit(1)

# Update the PERSONALITIES dictionary
PERSONALITIES = {
    'agent1': """You're James, you are helping Mike with her problem. Keep your messages to 2-3 short sentences and use everyday language.""",
    
    'agent2': """You're Mike, seeking advice on the forum, Just focus on the problem and ask questions to get the best solution. Use everyday language and keep your messages to 2-3 short sentences.""",
    
    'agent3': """Your name is Bob. You MUST:
    - Summarize the main solution in ONE clear sentence
    - No greetings, no formatting, no extra words
    - Focus only on the practical solution discussed""",
    
    'validator': """You are a strict conversation validator focusing ONLY on these essential rules:

    For James:
    - Must be 2-3 sentences only
    - Must ONLY answer Mike's question/concern directly
    - Must use natural language
    - NO apostrophes or unnecessary punctuation
    - NO meta-text
    - Just plain text answers focused on the topic
    
    For Mike:
    - Must be 2-3 sentences only
    - Must ONLY discuss the topic at hand
    - Must use natural language
    - NO apostrophes or unnecessary punctuation
    - NO meta-text
    - Just plain text questions or responses
    
    If validating, respond ONLY in this format:
    If message meets ALL rules:
    "meets"
    
    If message breaks any rules:
    "doesn't meet: [list broken rules]"
    Example: "doesn't meet: contains symbols; has unnecessary comments"
    
    If asked to correct, provide a clean response that:
    - Contains ONLY relevant information
    - Uses plain text with plain language
    - Follows the 2-3 sentence rule
    - Removes all commentaries outside of the message in parentheses
    - If message starts and ends with parentheses, remove them
    Prefix your correction with "correction:"
    
    Message to check: {message}
    Speaker: {speaker}"""
}

def send_message_to_groq(message: str, system_prompt: str, max_retries: int = 3) -> str:
    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Groq's model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                top_p=0.7,
                max_tokens=1024,
                stream=False
            )
            
            if hasattr(completion, 'choices') and len(completion.choices) > 0:
                if hasattr(completion.choices[0], 'message'):
                    response = completion.choices[0].message.content
                    # Clean up formatting and greetings
                    response = response.replace('**', '').replace('Hey', '').replace('Hi', '')
                    response = response.replace('James:', '').replace('Mike:', '').strip()
                    
                    # Only validate messages from James and Mike, and only after the initial question
                    if "You're James" in system_prompt:
                        validated_response = validate_message(response, "James")
                        if "meets" in validated_response.lower():  # Changed from startswith
                            return response.strip()
                        elif validated_response.startswith("correction:"):
                            return validated_response[10:].strip()
                    elif "You're Mike" in system_prompt and "Your question was:" in message:
                        validated_response = validate_message(response, "Mike")
                        if "meets" in validated_response.lower():  # Changed from startswith
                            return response.strip()
                        elif validated_response.startswith("correction:"):
                            return validated_response[10:].strip()
                    
                    return response.strip()
            
            raise ValueError("Invalid response structure from API")
            
        except Exception as e:
            print(f"API request error: {str(e)}")
            if retry_count < max_retries - 1:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return f"Error: Failed to communicate with Groq API after {max_retries} attempts: {str(e)}"

# Update all references to send_message_to_nvidia to use send_message_to_groq
@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        initial_topic = data.get('topic', 'Tell me something interesting')
        continue_conversation = data.get('continue_conversation', False)
        initial_question = data.get('initial_question', '')
        last_agent = data.get('last_agent', '')
        
        conversation = []
        
        if continue_conversation:
            # Ensure proper agent alternation
            current_agent = 'agent2' if last_agent == 'agent1' else 'agent1'
            
            # Simplified prompts for clearer context
            if current_agent == 'agent2':
                prompt = f"Question: {initial_question}\nJames's advice: {initial_topic}\nRespond to his advice."
            else:
                prompt = f"Question: {initial_question}\nMike's response: {initial_topic}\nGive practical advice."
            
            response = send_message_to_groq(prompt, PERSONALITIES[current_agent])
            
            if isinstance(response, str) and response.startswith("Error:"):
                return jsonify({'error': response}), 500
            
            conversation.append({current_agent: response})
        else:
            # Initial conversation start - Don't validate the first user question
            formatted_question = f"James, {initial_topic}"
            conversation.append({'agent2': formatted_question})
            
            # James's first response - This will be validated
            response1 = send_message_to_groq(
                f"Mike asks: {initial_topic}\nGive practical advice.",
                PERSONALITIES['agent1']
            )
            
            if isinstance(response1, str) and response1.startswith("Error:"):
                return jsonify({'error': response1}), 500
            
            conversation.append({'agent1': response1})
            
            # Mike's response - This will be validated
            response2 = send_message_to_groq(
                f"Your question was: {initial_topic}\nJames's advice: {response1}\nRespond to his advice.",
                PERSONALITIES['agent2']
            )
            
            if isinstance(response2, str) and response2.startswith("Error:"):
                return jsonify({'error': response2}), 500
            
            conversation.append({'agent2': response2})
        
        return jsonify(conversation)
        
    except Exception as e:
        print(f"Error in start_conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_summary', methods=['POST'])
def get_summary():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'No messages provided'}), 400
            
        conversation = data.get('messages', [])
        if not conversation:
            return jsonify({'error': 'Empty conversation'}), 400
        
        conversation_text = "Here's the discussion that took place:\n\n"
        for msg in conversation:
            if not isinstance(msg, dict):
                continue
            agent = list(msg.keys())[0]
            content = list(msg.values())[0]
            conversation_text += f"{agent}: {content}\n\n"
        
        summary_prompt = (
            "Please provide a concise summary of the discussion above. "
            "Focus on: 1) The main points discussed 2) Any agreements or disagreements "
            "3) The final conclusion or outcome if any. Keep it brief but comprehensive."
        )
        
        summary = send_message_to_groq(
            conversation_text + "\n" + summary_prompt,
            PERSONALITIES['agent3']
        )
        
        if isinstance(summary, str) and summary.startswith("Error:"):
            return jsonify({'error': summary}), 500
            
        return jsonify({'agent3': summary})
        
    except Exception as e:
        print(f"Error in get_summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_quote', methods=['GET'])
def get_quote():
    prompt = "Generate an inspiring quote of the day that's brief and meaningful. Don't attribute it to anyone, just the quote itself."
    quote = send_message_to_groq(prompt, "You are a wise quote generator. Keep quotes short and impactful.")
    return jsonify({'quote': quote})

def get_nvidia_response(message: str) -> str:
    response = send_message_to_groq(message, PERSONALITIES['agent1'])
    
    while not meets_requirements(response):
        response = send_message_to_groq(message, PERSONALITIES['agent1'])
    
    return response

def meets_requirements(response: str) -> bool:
    # This check is no longer relevant
    max_length = 100
    required_keywords = ['performance', 'graphics']  # These keywords don't match current use case
    
    if len(response) > max_length:
        return False
    
    if not any(keyword in response.lower() for keyword in required_keywords):
        return False
    
    return True

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    nvidia_response = get_nvidia_response(message)  # This should be updated or removed
    return jsonify({'response': nvidia_response})

def validate_message(message: str, speaker: str, max_retries: int = 5) -> str:
    retry_count = 0
    original_message = message
    
    # Clean the message of unnecessary characters before validation
    def clean_message(text: str) -> str:
        import re
        text = re.sub(r'[^\w\s.,?!]', '', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    message = clean_message(message)
    
    while retry_count < 3:  # Only try 3 times before using validator's correction
        validation_prompt = PERSONALITIES['validator'].format(
            message=message,
            speaker=speaker
        )
        
        try:
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": validation_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                top_p=0.7,
                max_tokens=1024,
                stream=False
            )
            
            validation_result = completion.choices[0].message.content.lower().strip()
            
            # If the validator says it meets requirements, return the message immediately
            if "meets" in validation_result:
                return message
                
            # Only continue if it explicitly doesn't meet requirements
            if not validation_result.startswith("doesn't meet:"):
                return message  # Return message if validation response is unclear
                
            reasons = validation_result.replace("doesn't meet:", "").strip()
            print(f"Attempt {retry_count + 1} rejected - Reasons: {reasons}")
            
            # Rest of the retry logic...
            agent_type = 'agent1' if speaker == 'James' else 'agent2'
            retry_system_prompt = f"""{PERSONALITIES[agent_type]}
            Additional strict rules:
            - Use ONLY plain text in your response
            - NO special characters or symbols
            - NO comments or meta-text
            - Focus only on the exact answer or question
            - Use basic punctuation only when necessary"""
            
            retry_completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": retry_system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                top_p=0.7,
                max_tokens=1024,
                stream=False
            )
            
            message = clean_message(retry_completion.choices[0].message.content.strip())
            
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            # Should probably return the original message here instead of continuing the loop
            return original_message  # Add this line
            
        retry_count += 1
    
    # If we get here after 3 attempts, return the last cleaned message
    return message

if __name__ == '__main__':
    app.run(debug=True)
