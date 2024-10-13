from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from ai_agent import create_agent_for_user

app = Flask(__name__)

# Dicionário para manter o agente e a memória de cada usuário
agent_by_user = {}

@app.route('/bot', methods=['POST'])
def bot():
  incoming_msg = request.values.get('Body', '').strip()
  user_number = request.values.get('From')  # Número do usuário
  response = process_user_message(user_number, incoming_msg)
  resp = MessagingResponse()
  msg = resp.message()
  msg.body(response)
  return str(resp)

def process_user_message(user_number, message):
  # Verifica se já existe um agente para o usuário
  if user_number not in agent_by_user:
      agent = create_agent_for_user()
      agent_by_user[user_number] = agent
  else:
      agent = agent_by_user[user_number]
  
  # Processa a mensagem do usuário
  response = agent.run(message)
  return response

if __name__ == '__main__':
  app.run()
