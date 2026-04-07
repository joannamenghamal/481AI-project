import os
import json
from openai import OpenAI
from system_prompt import SYSTEM_PROMPT
from financial_net import recommend, ACTION_DOMAIN
from dotenv import load_dotenv

load_dotenv()
# load openai api key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TOOLS = [
    {
        "type": "function",
        "name": "get_recommendation",
        "description": (
            "Query the Bayesian network for a financial recommendation. "
            "Call this once you have collected enough information from the user. "
            "You must provide at least DebtRatio, EmergencyFund, and RiskTolerance. "
            "Include any other variables you know."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "Income":               {"type": "string", "enum": ['low', 'medium', 'high']},
                "DebtRatio":            {"type": "string", "enum": ['low', 'medium', 'high']},
                "EmergencyFund":        {"type": "string", "enum": ['none', 'partial', 'adequate']},
                "RiskTolerance":        {"type": "string", "enum": ['conservative', 'moderate', 'aggressive']},
                "MarketConditions":      {"type": "string", "enum": ['bear', 'neutral', 'bull']},
                "InvestmentHorizon":    {"type": "string", "enum": ['short', 'medium', 'long']},
            },
            "required": ["DebtRatio", "EmergencyFund", "RiskTolerance"],
        },
    },
]

# bridge between LLM world and BN
def get_recommendation(arguments: dict) -> str:
    """
    Execute the Bayesian network with the arguments the LLM chose.
    Returns a plain-text result the LLM will read and explain to the user.
    """
    dist = recommend(arguments)

    ranked = sorted(ACTION_DOMAIN, key=lambda a: dist[a], reverse=True)

    lines = ["Bayesian network results (probabilityof each action):"]
    for action in ranked:
        lines.append(f" {action}: {dist[action]:.1%}")

    return "\n".join(lines)


def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("BayesFin: Hi! I'm your Financial Advisor. Tell me about your financial situation.\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye", "goodbye", "q"):
            print("BayesFin: Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Inner loop: let the model run until it produces a message for the user
        while True:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            has_tool_calls = False

            for item in response.output:
                if item.type == "function_call":
                    has_tool_calls = True
                    arguments = json.loads(item.arguments)
                    print(f"\n[Calling Bayesian network with: {arguments}]\n")
                    result = get_recommendation(arguments)
                    messages.append(item)
                    messages.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result,
                    })

                elif item.type == "message":
                    reply = item.content[0].text
                    messages.append({"role": "assistant", "content": reply})
                    print(f"\nBayesFin: {reply}\n")

            # After processing all items: if no tool was called, the model
            # produced text for the user — break inner loop and wait for input
            if not has_tool_calls:
                break

                    

# Declare variables
VARIABLES = [
    'Income',
    'DebtRatio',
    'EmergencyFund',
    'RiskTolerance',
    'MarketCondition',
    'InvestmentHorizon',
]

# Valid values for each variable - the BN only understands these exact strings
DOMAINS = {
    'Income':               ['low', 'medium', 'high'],
    'DebtRatio':            ['low', 'medium', 'high'],
    'EmergencyFund':        ['none', 'partial', 'adequate'],
    'RiskTolerance':        ['conservative', 'moderate', 'aggressive'],
    'MarketCondition':      ['bear', 'neutral', 'bull'],
    'InvestmentHorizon':    ['short', 'medium', 'long'],
}

# def extract_variables(user_message: str, current_state: dict) -> dict:
#     """
#     Ask the LLM to extract financial variables from the user's message.
#     Returns an updated copy of current_state with any newly identified values filled in.
#     """
#     system_prompt = f""" You are a financial data extractor.
# Extract financial variables from the user's message and return ONLY valid JSON.str

# Variables to extract and their ONLY valid values:
# - Income: {DOMAINS['Income']}
# - DebtRatio: {DOMAINS['DebtRatio']}
# - Emergency Fund: {DOMAINS['EmergencyFund']}
# - RiskTolerance: {DOMAINS['RiskTolerance']}
# - MarketCondition: {DOMAINS['MarketCondition']}
# - InvestmentHorizon: {DOMAINS['InvestmentHorizon']}

# Current known values (do not overwrite unless the user corrects them):
# {[json.dumps(current_state, indent=2)]}

# Rules:
# - Return ONLY a JSON object with the 6 variable names as keys
# - Use null for variables you cannot determine
# - Use ONLY the exact string values listed above
# - Do not invent values outside the lists above
# """
    
#     response = client.responses.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message},
#         ],
#         response_format={"type": "json_object"},
#     )

#     extracted = json.loads(response.ouputs[0].message.content)

#     # Merge: only update state where we got a valid non-null value
#     updated = current_state.copy()
#     for var in VARIABLES:
#         val = extracted.get(var)
#         if val in DOMAINS[var]:     # only accept valid domain values
#             updated[var] = val

#     return updated