# ai_agent.py
import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import ast

# Função para carregar os dados do cliente
def load_client_data():
  df = pd.read_csv('data/client_data.csv')
  client_info = df.iloc[0].to_dict()
  # Converte strings de listas e dicionários para objetos Python
  client_info['historico_pagamentos'] = ast.literal_eval(client_info['historico_pagamentos'])
  client_info['cartoes_vencidos'] = ast.literal_eval(client_info['cartoes_vencidos'])
  return client_info

# Função para carregar os planos de internet disponíveis
def load_internet_plans():
  df = pd.read_csv('data/internet_plans.csv')
  plans = df.to_dict(orient='records')
  return plans

# Funções que o agente pode realizar
def get_debt_info(client_info):
  return f"Vejo que você tem uma conta de R$ {client_info['valor']} de {client_info['divida']}, vencida no dia {client_info['vencimento']}."

def get_payment_options(client_info):
  valor = float(client_info['valor'])
  options = [
      f"2 parcelas de R$ {valor/2:.2f}",
      f"4 parcelas de R$ {valor/4:.2f}"
  ]
  options_text = "\n".join(options)
  return f"Aqui estão as opções de parcelamento para o valor de R$ {valor:.2f}:\n{options_text}\nQual opção você prefere?"

def confirm_payment(option):
  return f"Perfeito, Maria! Confirme, por favor: você vai parcelar a dívida em {option}. Deseja confirmar a operação?"

def payment_confirmed(option):
  return f"Pagamento confirmado! Sua dívida foi parcelada em {option}."

def alert_card_expiry(client_info):
  return f"Maria, notei que seu cartão cadastrado termina em {client_info['cartao_ativo']} e vai expirar em {client_info['cartao_vencimento']}. Você gostaria de cadastrar outro cartão agora?"

def offer_charge_any_card():
  return "Além disso, você pode cadastrar mais de um cartão e ativar a função Cobrar em Qualquer Cartão. Assim, se um cartão estiver sem limite, o sistema tenta automaticamente o outro. Isso evita que seus serviços sejam suspensos por falta de pagamento. Deseja ativar?"

def charge_any_card_activated():
  return "Função ativada com sucesso! Agora, se houver qualquer problema com limite de um cartão, o outro será usado automaticamente. Isso garante que você continue utilizando seus serviços da Claro sem interrupções! 🎉"

def suggest_due_date_change(client_info):
  atrasos = int(client_info['atrasos_recentes'])
  if atrasos >= 3:
      return f"Maria, percebi que seus últimos 3 pagamentos foram feitos com 15 dias de atraso, próximos ao dia {client_info['preferencia_vencimento']}. Para evitar futuros atrasos e custos extras, gostaria de sugerir uma troca na data de vencimento. Você pode escolher uma data que seja mais conveniente.\nGostaria de ajustar o vencimento?"
  else:
      return ""

def due_date_changed(client_info):
  return f"Pronto! A partir da próxima fatura, o vencimento será no dia {client_info['preferencia_vencimento']} de cada mês. Vou te enviar uma notificação de pagamento quando estiver próximo do vencimento. Isso deve te ajudar a evitar atrasos e evitar custos adicionais. 😉"

def offer_internet_package(client_info):
  plans = load_internet_plans()
  additional_plan = next((plan for plan in plans if plan['franquia'] == '15GB'), None)
  if additional_plan:
      return f"Com base no seu uso, você pode contratar mais 5GB de internet por apenas {additional_plan['valor_mensal']} mensais. Interessada?"
  else:
      return "No momento, não temos planos adicionais disponíveis."

def internet_package_activated():
  return "Perfeito! Vou ativar a oferta para você e a partir do próximo mês, terá 5GB adicionais. E claro, se precisar de mais alguma coisa, estou sempre por aqui!"

def offer_consumption_alerts():
  return "Podemos te enviar notificações quando estiver perto de usar todo o seu pacote de dados para evitar surpresas. Gostaria de ativar?"

def consumption_alerts_activated():
  return "Alertas de consumo ativados com sucesso! Você receberá notificações quando estiver perto de atingir sua franquia de dados."

def conclude_interaction():
  return "Tudo certo, Maria! 🙌 Você parcelou sua conta, cadastrou um novo cartão e ajustou a data de vencimento. Agora, com o aumento de franquia, você também terá mais internet no próximo mês. Qualquer dúvida ou necessidade, estarei sempre disponível! 😊"

# Inicializa o agente
def create_agent_for_user():
  client_info = load_client_data()
  llm = OpenAI(temperature=0)
  tools = [
      Tool(
          name="Get Debt Info",
          func=lambda x: get_debt_info(client_info),
          description="Obtém informações sobre a dívida do cliente."
      ),
      Tool(
          name="Get Payment Options",
          func=lambda x: get_payment_options(client_info),
          description="Fornece opções de pagamento para o cliente."
      ),
      Tool(
          name="Confirm Payment",
          func=confirm_payment,
          description="Confirma o pagamento com base na opção escolhida."
      ),
      Tool(
          name="Payment Confirmed",
          func=payment_confirmed,
          description="Informa que o pagamento foi confirmado."
      ),
      Tool(
          name="Alert Card Expiry",
          func=lambda x: alert_card_expiry(client_info),
          description="Alerta sobre o cartão que está para expirar."
      ),
      Tool(
          name="Offer Charge Any Card",
          func=lambda x: offer_charge_any_card(),
          description="Oferece a função Cobrar em Qualquer Cartão."
      ),
      Tool(
          name="Charge Any Card Activated",
          func=lambda x: charge_any_card_activated(),
          description="Confirma que a função Cobrar em Qualquer Cartão foi ativada."
      ),
      Tool(
          name="Suggest Due Date Change",
          func=lambda x: suggest_due_date_change(client_info),
          description="Sugere a mudança da data de vencimento."
      ),
      Tool(
          name="Due Date Changed",
          func=lambda x: due_date_changed(client_info),
          description="Confirma que a data de vencimento foi alterada."
      ),
      Tool(
          name="Offer Internet Package",
          func=lambda x: offer_internet_package(client_info),
          description="Oferece aumento de franquia de internet."
      ),
      Tool(
          name="Internet Package Activated",
          func=lambda x: internet_package_activated(),
          description="Confirma que o pacote de internet foi ativado."
      ),
      Tool(
          name="Offer Consumption Alerts",
          func=lambda x: offer_consumption_alerts(),
          description="Oferece alertas de consumo de dados."
      ),
      Tool(
          name="Consumption Alerts Activated",
          func=lambda x: consumption_alerts_activated(),
          description="Confirma que os alertas de consumo foram ativados."
      ),
      Tool(
          name="Conclude Interaction",
          func=lambda x: conclude_interaction(),
          description="Conclui a interação com o cliente."
      ),
  ]

  prompt_template = PromptTemplate(
      input_variables=["input", "chat_history"],
      template="""
Você é a Bia, a inteligência artificial de pagamentos da Bemobi. Auxilie a cliente Maria de forma cordial e eficiente.

Histórico da Conversa:
{chat_history}

Instrução:
{input}

Responda de forma clara, seguindo as políticas da empresa e garantindo a satisfação da cliente.
"""
  )

  llm_chain = LLMChain(
      llm=llm,
      prompt=prompt_template,
      memory=ConversationBufferMemory(memory_key="chat_history")
  )

  agent = initialize_agent(
      tools,
      llm_chain,
      agent="conversational-react-description",
      verbose=True
  )

  return agent