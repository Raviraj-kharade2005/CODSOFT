import re

def get_response(message):
    msg = message.lower()
    # Greetings
    if re.search(r"hi|hello|hey", msg):
        return "Hello! How can I help you today?"
    # Ask time
    if re.search(r"time", msg):
        from datetime import datetime
        return f"Current time is {datetime.now().strftime('%H:%M:%S')}"
    # Fallback
    return "I'm not sure I understand. Can you rephrase?"

if __name__ == "__main__":
    print("Chatbot: Hi there! (type 'quit' to exit)")
    while True:
        user = input("You: ")
        if user.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        print(f"Chatbot: {get_response(user)}")