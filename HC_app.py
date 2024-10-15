#Requirements
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#LLM Model
llama = ChatGroq(
    model="LLaMA3-70B-8192",
    groq_api_key='gsk_Testij5mRZLIYEh3rwxWWGdyb3FYxSCONQEPGNX5ETg08Xr0XlGy',
    temperature=0.0
)

##################################### Diet ###########################################
base_diet_chain = (
    ChatPromptTemplate.from_template("""
You are a nutritionist known for best personalized meal planning.  You have to create a meal plan for any given personalizations {personalizations}, 
you have to provide a meal with balanced calories containing of protien, carbohydrates and fats suitable for a healthy body.
Generate 2 different meals, one for vegetarian and the other for non-vegetarian with their nutrition distribution.
""")
    | llama
    | StrOutputParser()
    |{"base_response": RunnablePassthrough()}
)

nut_chain = (
    ChatPromptTemplate.from_template("""
Get a list of foods that provides balanced nutrition for the requested plan from the {base_response}.
""")
    | llama
    | StrOutputParser()
)

bal_chain = (
    ChatPromptTemplate.from_template("""
Set the foods from the list for a meal suitable for the given requested plan with their nutrition distribution of each meal from the {base_response}.
""")
    | llama
    | StrOutputParser()
)     

Diet_chain = (
    base_diet_chain
    | nut_chain
    | bal_chain
)    
      

############################### HealthAdvisor #################################################
ha_chain = (
    ChatPromptTemplate.from_template("""
You are a AI Health care advisor who loves to care about human wellbeing.  You give health care  advices to the user, according to the problem: {problem} provided by the user, 
you also generate some key tips and plans to follow for the user to get better.
""")
    | llama
    | StrOutputParser()
)


############################################ Diagnose-Treatment ######################################
base_dt_chain = (
    ChatPromptTemplate.from_template("""
You are an AI Doctor who loves to care about human wellbeing. You diagnose any disease by checking some symptoms provided to you, according to the the problem: {problem} provided by the user, 
you also generate a list of symptoms recognized, a detailed list of possible diseases and suggest common treatment methods and ayurvedic treatment methods for each of the listed diseases.
""")
    | llama
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

symptoms_chain = (
    ChatPromptTemplate.from_template(
        "You are a expert in recognizing symptoms from the provided base informatioin : {base_response}. Generate a list of recognized symptoms."
    )
    | llama
    | StrOutputParser()
)
diseases_chain = (
    ChatPromptTemplate.from_template(
        "You are a expert doctor in diagnosing diseases from the provided symptoms in: {base_response}. Generate a list of recognized diseases with their breif information."
    )
    | llama
    | StrOutputParser()
)

treatment_ct_chain = (
    ChatPromptTemplate.from_template(
        "You are a expert doctor in giving common treatments by some given diseases provided to you in: {base_response}. Generate a list of common treatments."
    )
    | llama
    | StrOutputParser()
)

treatment_ayu_chain = (
    ChatPromptTemplate.from_template(
        "You are a expert doctor in giving ayurvedic treatments by some given diseases provided to you in: {base_response}. Generate a list of ayurvrdic treatments."
    )
    | llama
    | StrOutputParser()
)

responder_dt_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "Symptoms:\n{results_1}\n\nDiseases:\n{results_2}\n\nCommonTreatment:\n{results_3}\n\nAyurvedicTreatment:\n{results_4}"),
            ("system", "Generate a final detailed response with lists of symptoms, diseases and both type of treatments with a brief from the results."),
        ]
    )
    | llama
    | StrOutputParser()
)


DT_chain = (
    base_dt_chain
    | {
        "original_response": itemgetter("base_response"),
        "results_1": symptoms_chain,
        "results_2": diseases_chain,
        "results_3": treatment_ct_chain,
        "results_4": treatment_ayu_chain,
        
    }
    | responder_dt_chain
)



####################################### Workout ################################
base_workout_chain = (
    ChatPromptTemplate.from_template("""
You are an AI Gym Trainer who is known for making the best personalized workout plan. Personalization:{personalization} can contain attributes such as calorie focused, body weight, bmi, difficulty level, time constrained, place, environment, type of physical activity or any other specified personalizations given by the user. 
You are also specialized in other physical activities such as yoga and calisthenics.
You generate workout plans for 'weight training' and 'bodyweight training'.
You also give special tips to help the user achieve the best form for every exercise. 
""")
    | llama
    | StrOutputParser()
)



####################### Mental Health ######################################
base_mh_chain = (
    ChatPromptTemplate.from_template("""
You are an AI Psychiatrist who loves to care about human wellbeing. You are proficient in identifying, preventing and treating mental-health-related disorders. You diagnose the disease by checking some symptoms provided to you, from the the problem: {problem} provided by the user, 
you also generate a list of symptoms recognized, a detailed list of possible diseases and give treatment for each of the listed diseases.
""")
    | llama
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

symptoms_mh_chain = (
    ChatPromptTemplate.from_template(
        "You are an expert Psychiatrist who recognizes symptoms from the provided base informatioin : {base_response}. Generate a list of recognized symptoms."
    )
    | llama
    | StrOutputParser()
)
diseases_mh_chain = (
    ChatPromptTemplate.from_template(
        "You are an expert Psychiatrist who diagnose diseases from the provided symptoms in: {base_response}. Generate a list of recognized diseases with their breif information."
    )
    | llama
    | StrOutputParser()
)

treatment_mh_chain = (
    ChatPromptTemplate.from_template(
        "You are a expert Psychiatrist who gives treatments for the given diseases provided to you in: {base_response}. Generate a list of common treatments."
    )
    | llama
    | StrOutputParser()
)

responder_mh_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "Symptoms:\n{results_1}\n\nDiseases:\n{results_2}\n\nTreatment:\n{results_3}"),
            ("system", "Generate a final detailed response with lists of symptoms, diseases and treatments with a brief from the results."),
        ]
    )
    | llama
    | StrOutputParser()
)

MH_chain = (
    base_mh_chain
    | {
        "original_response": itemgetter("base_response"),
        "results_1": symptoms_mh_chain,
        "results_2": diseases_mh_chain,
        "results_3": treatment_mh_chain,
        
    }
    | responder_mh_chain
)


####################### Cognitive behavioral therapy (CBT) #####################################
cbt_chain = (
    ChatPromptTemplate.from_template("""
You are an AI Psychiatrist who loves to care about human wellbeing. You are proficient in Cognitive behavioral therapy (CBT), Interpersonal therapy (IPT), Graded exercise therapy, Cognitive-behavioral therapy for insomnia (CBT-I). 
The patient  shares their problem: {problem} to you intially.
Provide treatment to the patient by your chosen therapy.
""")
    | llama
    | StrOutputParser()
)
pchain = (
    ChatPromptTemplate.from_template("""
Explain the following Steps {step} for an experiment
""")
    | llama
    | StrOutputParser()
)

######################## Streamlit app #####################################

# Title of the app
st.title("AI Health Partner by PARTH")
#st.markdown("### ***Select your bot from options first.***")

# Introduction text
with st.expander("About app..."):
    st.write("""
        Welcome to your Personal Health Partner (PHP) ! 
        My AI-powered (by LLaMA 3 70B) app is designed to provide you with reliable and accessible health information at your fingertips.\n
        This app compiles every field of your life, you want guidance for. Areas such as Health- Mental, Physical are coverd.\n
        Whether you have a specific health query or simply want to learn more about wellness,  PHP is here to assist you.\n
    """)

# Sidebar for additional information

option= st.selectbox(
    "Select your bot from options given below. Description of bots are provided in the sidebar.",
    ("Health advisor", "Diet", "Workout", "Diagnose-Treatment", "Mental Health", "Mental Therapy", "lm"),
    index=None,
    placeholder="Select any...",
)
st.sidebar.title(option)
st.sidebar.header("Description")

class info:

    ha_info = """
    My **AI Health Advisor** is your go-to app for reliable and personalized health information.\n 
    My app provides instant answers to your health questions, offers tailored advice, and supports you on your wellness journey.
    """
    diet_info= """
    My **AI Diet Planner** is your go-to app for making best personalized diet plans.\n 
    - You can provide any personalizations (e.g. goals, bodyweight, calories, protien goals and much more.)\n
    - You can provide a list of foods that you want for your diet plan.\n
    - It gives you 2 diet plans: A. Vegetarian B. Non-vegetarian
    """
    mh_info= """
    My **AI Mental Health Doctor** is your go-to app for helping you restore your mental health back to 100%.\n
    ## How to use?\n
    ### Input:\n
    You can share your problem or symptoms you notice while self observation as an input.\n
    ### Function:\n
    The doctor will diagnose you based on your symptoms and give you treatment for the diagnosed disease.\n
    ### Extension:\n
    If you want a CBT, CBT-I or GET therapy for mental health then there is a doctor ready for you at ***Mental Therapy***
    """
    mt_info= """
    My **AI Mental Therapist** is your go-to app if you want a therapy for ***CBT, CBT-I, GET***.\n
    - My therapist is proficient for the mentioned therapies.\n
    - Don't worry if you haven't diagnosed with any therapy yet.\n
    - My therapist will diagnose your ***symptoms***, ***choose the best therapy*** and provide you the best ***treatment***.
    ### How to use?:\n
    Just chat and answer with honesty.
    """
    DT_info="""
    My **AI Doctor** is your go-to app if you are looking for a Generel Doctor at your fingertips.\n
    - Just as you talk with a doctor share your symptoms or problem for the input.\n
    - The doctor will diagnose you and gives you a list of all possible diseases.\n
    - Also, treatment will be provided in 2 types:\n
    a) Common Treatment\n b) Ayurvedic Treatment
    """
    workout_info= """
    My **AI Trainer** is your go-to app for making the best personalized workout plan.\n
    - You can ask for any type of workout such as ***yoga, calisthenics, home workout, body weight or gym workout.***\n
    - You can also specify your goals such as ***Fatloss, Muscle gain, Strength building, flexibility and much more***.\n
    """
    
    if option == "Health advisor":
        st.sidebar.markdown(ha_info)
    elif option == "Mental Health":
        st.sidebar.markdown(mh_info)
    elif option == "Diet":
        st.sidebar.markdown(diet_info)
    elif option == "Diagnose-Treatment":
        st.sidebar.markdown(DT_info)
    elif option == "Workout":
        st.sidebar.markdown(workout_info)
    elif option == "Mental Therapy":
        st.sidebar.markdown(mt_info)
    
    
st.write("You selected:", option)

# Load Groq compiled LLaMA model (replace with your actual model path)
@st.cache_resource

# Generate a response from the model
def generate_response(userinput):
    match option:
        case "Health advisor":
            return ha_chain.invoke(userinput)
        case "Mental Health":
            return MH_chain.invoke(userinput)
        case "Diet":
            return Diet_chain.invoke(userinput)
        case "Workout":
            return base_workout_chain.invoke(userinput)
        case "Diagnose-Treatment":
            return DT_chain.invoke(userinput)
        case "Mental Therapy":
            return cbt_chain.invoke(userinput)
        case default:
            return "Nothing"
  
 
if "messages" not in st.session_state:
    st.session_state.messages = []

# getting User input
userinput = st.chat_input("Say something")
with st.chat_message("user"):
        st.write(userinput)

if userinput:
    message = st.chat_message("assistant")
    #message.write(cbt_chain.invoke(user_input))
    st.session_state.messages.append({"role": "user", "content": userinput})
    bot_response = generate_response(userinput)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

     
# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Bot: {message['content']}")

