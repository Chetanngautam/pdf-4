"""
============================================================
INTEGRATED AI/ML INTERNSHIP PROJECT
============================================================

Project 1: Autonomous Customer Support Copilot
Project 2: End-to-End ML Deployment Platform
Project 3: Financial Fraud Intelligence Engine

TOPICS COVERED:
------------------------------------------------------------
1. NLP Processing
2. LLM Integration (Mock/Template-based prototype)
3. Intent Detection and Routing
4. RAG - Retrieval Augmented Generation
5. Feedback Learning Loop
6. Customer Ticket Resolution
7. Knowledge Base Integration
8. Automatic Ticket Escalation
9. Resolution Metrics Tracking
10. Data Pipeline
11. Airflow Pipeline Simulation
12. Model Training Pipeline
13. Automatic Model Deployment
14. CI/CD Simulation
15. Model Monitoring
16. Data Drift Detection
17. Automatic Retraining
18. Streaming ML Simulation
19. Anomaly Detection
20. Graph Fraud Detection
21. Fraud Risk Scoring
22. Fraud Visualization Dashboard
============================================================
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ============================================================
# PART 1
# AUTONOMOUS CUSTOMER SUPPORT COPILOT
# ============================================================

print("\n" + "=" * 70)
print("PROJECT 1: AUTONOMOUS CUSTOMER SUPPORT COPILOT")
print("=" * 70)


# ============================================================
# STEP 1: CREATE CUSTOMER SUPPORT DATASET
# ============================================================

support_data = pd.DataFrame({

    "ticket": [

        "I forgot my password",
        "How can I reset my password",
        "Unable to login to my account",
        "My account is locked",

        "Payment failed",
        "Money deducted but transaction failed",
        "My card payment is not working",
        "Payment was declined",

        "I want a refund",
        "Refund has not arrived",
        "How can I cancel and get refund",
        "Refund is pending",

        "Application is crashing",
        "Website is not opening",
        "Software is running very slow",
        "I found a technical error",

        "Someone hacked my account",
        "There is an unauthorized transaction",
        "My financial account is compromised",
        "I detected suspicious account activity"
    ],

    "intent": [

        "password",
        "password",
        "password",
        "password",

        "payment",
        "payment",
        "payment",
        "payment",

        "refund",
        "refund",
        "refund",
        "refund",

        "technical",
        "technical",
        "technical",
        "technical",

        "security",
        "security",
        "security",
        "security"
    ]
})


print("\nCustomer Support Dataset:")
print(support_data.head())


# ============================================================
# STEP 2: NLP TEXT VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_support = vectorizer.fit_transform(support_data["ticket"])

y_support = support_data["intent"]


# ============================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_support,
    y_support,

    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_support
)


# ============================================================
# STEP 4: INTENT DETECTION MODEL
# ============================================================

intent_model = LogisticRegression(
    max_iter=1000
)

intent_model.fit(
    X_train,
    y_train
)


# ============================================================
# STEP 5: MODEL EVALUATION
# ============================================================

intent_predictions = intent_model.predict(X_test)

intent_accuracy = accuracy_score(
    y_test,
    intent_predictions
)

print("\nIntent Detection Accuracy:")
print(intent_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        intent_predictions,
        zero_division=0
    )
)


# ============================================================
# STEP 6: COMPANY KNOWLEDGE BASE
# ============================================================

knowledge_base = pd.DataFrame({

    "topic": [

        "password",
        "payment",
        "refund",
        "technical",
        "security"
    ],

    "information": [

        """
        To reset your password, open the login page,
        click Forgot Password and follow the instructions
        sent to your registered email.
        """,

        """
        For failed payments, verify your payment details,
        check your account balance and retry the transaction.
        Contact support if money was deducted.
        """,

        """
        Refunds generally require processing time.
        Check your refund status in the payment section
        or contact the billing support department.
        """,

        """
        Restart the application, clear cache and verify
        your internet connection. Contact technical support
        if the problem continues.
        """,

        """
        Immediately change your password and contact the
        security department if you notice unauthorized
        transactions or suspicious activity.
        """
    ]
})


# ============================================================
# STEP 7: RAG SYSTEM
# ============================================================

knowledge_vectorizer = TfidfVectorizer(
    stop_words="english"
)

knowledge_vectors = knowledge_vectorizer.fit_transform(
    knowledge_base["information"]
)


def retrieve_knowledge(query):

    """
    Retrieve the most relevant information
    from the company knowledge base.
    """

    query_vector = knowledge_vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        knowledge_vectors
    )

    best_document_index = similarity_scores.argmax()

    best_score = similarity_scores.max()

    information = knowledge_base.iloc[
        best_document_index
    ]["information"]

    return information.strip(), best_score


# ============================================================
# STEP 8: LLM RESPONSE GENERATION SIMULATION
# ============================================================

def generate_llm_response(ticket, context):

    """
    Prototype LLM integration.

    In a production project, this function can be connected
    with an LLM API.
    """

    response = f"""
Hello,

Based on your customer support request:

"{ticket}"

Recommended Solution:

{context}

If the issue continues, your ticket can be transferred
to a human customer support agent.

Thank you.
"""

    return response.strip()


# ============================================================
# STEP 9: AUTOMATIC ESCALATION
# ============================================================

def should_escalate(intent, confidence):

    complex_intents = [
        "security"
    ]

    if intent in complex_intents:
        return True

    if confidence < 0.50:
        return True

    return False


# ============================================================
# STEP 10: CHATBOT
# ============================================================

resolution_metrics = {

    "total_tickets": 0,
    "automatically_resolved": 0,
    "escalated": 0,
    "positive_feedback": 0,
    "negative_feedback": 0
}


feedback_data = []


def customer_support_chatbot(ticket):

    resolution_metrics["total_tickets"] += 1

    ticket_vector = vectorizer.transform([ticket])

    intent = intent_model.predict(
        ticket_vector
    )[0]

    probabilities = intent_model.predict_proba(
        ticket_vector
    )[0]

    confidence = np.max(probabilities)

    print("\nDetected Intent:", intent)

    print(
        "Intent Confidence:",
        round(confidence, 2)
    )

    # Retrieve company knowledge

    context, rag_score = retrieve_knowledge(ticket)

    print(
        "RAG Similarity Score:",
        round(rag_score, 2)
    )

    # Automatic escalation

    escalation = should_escalate(
        intent,
        confidence
    )

    if escalation:

        resolution_metrics["escalated"] += 1

        return """
This issue has been identified as complex or sensitive.

The ticket has been automatically escalated
to a human customer support agent.
"""

    else:

        resolution_metrics[
            "automatically_resolved"
        ] += 1

        return generate_llm_response(
            ticket,
            context
        )


# ============================================================
# STEP 11: FEEDBACK LEARNING LOOP
# ============================================================

def collect_feedback(ticket, feedback):

    feedback_data.append({

        "ticket": ticket,
        "feedback": feedback
    })

    if feedback.lower() == "positive":

        resolution_metrics[
            "positive_feedback"
        ] += 1

    else:

        resolution_metrics[
            "negative_feedback"
        ] += 1


# ============================================================
# TEST CUSTOMER SUPPORT CHATBOT
# ============================================================

test_ticket = "My payment failed and money was deducted"

response = customer_support_chatbot(test_ticket)

print("\nChatbot Response:")
print(response)

collect_feedback(
    test_ticket,
    "positive"
)


# ============================================================
# CUSTOMER SUPPORT METRICS
# ============================================================

print("\nCustomer Support Resolution Metrics:")

for key, value in resolution_metrics.items():

    print(key, ":", value)


# ============================================================
# PART 2
# END-TO-END ML DEPLOYMENT PLATFORM
# ============================================================

print("\n" + "=" * 70)
print("PROJECT 2: END-TO-END ML DEPLOYMENT PLATFORM")
print("=" * 70)


# ============================================================
# STEP 1: DATA PIPELINE
# ============================================================

def data_pipeline():

    """
    Simulates data collection and preprocessing.
    """

    print("\nRunning Data Pipeline...")

    number_of_samples = 2000

    data = pd.DataFrame({

        "feature_1":
            np.random.normal(
                50,
                10,
                number_of_samples
            ),

        "feature_2":
            np.random.normal(
                100,
                20,
                number_of_samples
            ),

        "feature_3":
            np.random.randint(
                1,
                10,
                number_of_samples
            ),

        "target":
            np.random.randint(
                0,
                2,
                number_of_samples
            )
    })

    # Handle missing values

    data = data.dropna()

    # Remove duplicates

    data = data.drop_duplicates()

    print(
        "Data Pipeline Completed."
    )

    print(
        "Dataset Shape:",
        data.shape
    )

    return data


# ============================================================
# STEP 2: AIRFLOW PIPELINE SIMULATION
# ============================================================

def airflow_pipeline():

    """
    Simulates Airflow DAG tasks.
    """

    print("\nAIRFLOW PIPELINE")

    print("Task 1: Data Collection")

    data = data_pipeline()

    print("Task 2: Data Validation")

    print("Task 3: Model Training")

    model, accuracy, scaler = train_ml_model(data)

    print("Task 4: Model Evaluation")

    print("Task 5: Model Deployment")

    deploy_model(
        model,
        accuracy
    )

    return (
        data,
        model,
        accuracy,
        scaler
    )


# ============================================================
# STEP 3: MODEL TRAINING PIPELINE
# ============================================================

def train_ml_model(data):

    X = data.drop(
        "target",
        axis=1
    )

    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = RandomForestClassifier(

        n_estimators=100,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "\nModel Accuracy:",
        round(accuracy, 4)
    )

    return (
        model,
        accuracy,
        scaler
    )


# ============================================================
# STEP 4: AUTOMATIC MODEL DEPLOYMENT
# ============================================================

def deploy_model(model, accuracy):

    deployment_threshold = 0.50

    if accuracy >= deployment_threshold:

        print(
            "\nMODEL DEPLOYMENT SUCCESSFUL"
        )

        print(
            "Model passed deployment threshold."
        )

    else:

        print(
            "\nMODEL DEPLOYMENT FAILED"
        )

        print(
            "Model requires additional training."
        )


# ============================================================
# STEP 5: CI/CD PIPELINE SIMULATION
# ============================================================

def cicd_pipeline():

    print("\nRunning CI/CD Pipeline...")

    stages = [

        "Source Code Validation",

        "Unit Testing",

        "Model Testing",

        "Security Check",

        "Build Application",

        "Deploy Model"
    ]

    for stage in stages:

        print(
            "Running:",
            stage
        )

        time.sleep(0.2)

    print(
        "\nCI/CD Pipeline Completed Successfully."
    )


# ============================================================
# STEP 6: MODEL PERFORMANCE MONITORING
# ============================================================

def monitor_model_performance(
        model_accuracy,
        threshold=0.50
):

    print("\nMonitoring Model Performance...")

    if model_accuracy < threshold:

        print(
            "WARNING: Model Performance Decreased."
        )

        return False

    else:

        print(
            "Model Performance is Stable."
        )

        return True


# ============================================================
# STEP 7: DATA DRIFT DETECTION
# ============================================================

def detect_data_drift(
        reference_data,
        new_data,
        threshold=10
):

    print("\nChecking Data Drift...")

    drift_detected = False

    for column in reference_data.columns:

        if column == "target":
            continue

        reference_mean = reference_data[
            column
        ].mean()

        new_mean = new_data[
            column
        ].mean()

        difference = abs(
            reference_mean -
            new_mean
        )

        print(
            column,
            "Mean Difference:",
            round(difference, 2)
        )

        if difference > threshold:

            drift_detected = True

    if drift_detected:

        print(
            "WARNING: DATA DRIFT DETECTED"
        )

    else:

        print(
            "No Significant Data Drift."
        )

    return drift_detected


# ============================================================
# STEP 8: AUTOMATIC MODEL RETRAINING
# ============================================================

def automatic_retraining(
        drift_detected,
        new_data
):

    if drift_detected:

        print(
            "\nStarting Automatic Model Retraining..."
        )

        new_model, new_accuracy, new_scaler = train_ml_model(
            new_data
        )

        print(
            "Model Retraining Completed."
        )

        return (
            new_model,
            new_accuracy,
            new_scaler
        )

    else:

        print(
            "\nRetraining is not required."
        )

        return None


# ============================================================
# RUN END-TO-END ML PIPELINE
# ============================================================

ml_data, ml_model, ml_accuracy, ml_scaler = airflow_pipeline()

cicd_pipeline()

monitor_model_performance(
    ml_accuracy
)


# ============================================================
# CREATE NEW DATA FOR DRIFT TEST
# ============================================================

new_ml_data = ml_data.copy()

new_ml_data["feature_1"] = (
    new_ml_data["feature_1"] + 20
)

drift_status = detect_data_drift(

    ml_data,
    new_ml_data
)

automatic_retraining(

    drift_status,
    new_ml_data
)


# ============================================================
# PART 3
# FINANCIAL FRAUD INTELLIGENCE ENGINE
# ============================================================

print("\n" + "=" * 70)
print("PROJECT 3: FINANCIAL FRAUD INTELLIGENCE ENGINE")
print("=" * 70)


# ============================================================
# STEP 1: CREATE FINANCIAL TRANSACTION DATA
# ============================================================

number_of_transactions = 3000

fraud_data = pd.DataFrame({

    "transaction_id":
        range(
            1,
            number_of_transactions + 1
        ),

    "sender":
        np.random.randint(
            1,
            300,
            number_of_transactions
        ),

    "receiver":
        np.random.randint(
            1,
            300,
            number_of_transactions
        ),

    "amount":
        np.random.exponential(
            5000,
            number_of_transactions
        ),

    "transaction_hour":
        np.random.randint(
            0,
            24,
            number_of_transactions
        ),

    "transaction_frequency":
        np.random.randint(
            1,
            50,
            number_of_transactions
        )
})


print("\nFinancial Transaction Dataset:")
print(fraud_data.head())


# ============================================================
# STEP 2: ANOMALY DETECTION
# ============================================================

fraud_features = [

    "amount",
    "transaction_hour",
    "transaction_frequency"
]


isolation_forest = IsolationForest(

    contamination=0.05,
    random_state=RANDOM_STATE
)


fraud_data["anomaly"] = isolation_forest.fit_predict(

    fraud_data[fraud_features]
)


fraud_data["is_suspicious"] = (

    fraud_data["anomaly"] == -1

).astype(int)


print(
    "\nNumber of Suspicious Transactions:",
    fraud_data["is_suspicious"].sum()
)


# ============================================================
# STEP 3: FRAUD RISK SCORING MODEL
# ============================================================

fraud_data["amount_risk"] = (

    fraud_data["amount"] /
    fraud_data["amount"].max()
)


fraud_data["frequency_risk"] = (

    fraud_data["transaction_frequency"] /
    fraud_data["transaction_frequency"].max()
)


fraud_data["night_risk"] = (

    (
        fraud_data["transaction_hour"] < 5
    )

    |

    (
        fraud_data["transaction_hour"] > 23
    )

).astype(int)


fraud_data["risk_score"] = (

    fraud_data["amount_risk"] * 40

    +

    fraud_data["frequency_risk"] * 30

    +

    fraud_data["night_risk"] * 10

    +

    fraud_data["is_suspicious"] * 20
)


fraud_data["risk_score"] = fraud_data[
    "risk_score"
].clip(
    0,
    100
)


# ============================================================
# STEP 4: FRAUD RISK CLASSIFICATION
# ============================================================

def classify_risk(score):

    if score >= 70:

        return "HIGH RISK"

    elif score >= 40:

        return "MEDIUM RISK"

    else:

        return "LOW RISK"


fraud_data["risk_level"] = fraud_data[
    "risk_score"
].apply(
    classify_risk
)


print("\nFraud Risk Distribution:")

print(
    fraud_data[
        "risk_level"
    ].value_counts()
)


# ============================================================
# STEP 5: STREAMING ML SIMULATION
# ============================================================

def streaming_fraud_detection(data):

    print("\nStarting Real-Time Fraud Detection Simulation...")

    sample_transactions = data.sample(10)

    for index, transaction in sample_transactions.iterrows():

        print("\nTransaction ID:", transaction["transaction_id"])

        print(
            "Amount:",
            round(
                transaction["amount"],
                2
            )
        )

        print(
            "Risk Score:",
            round(
                transaction["risk_score"],
                2
            )
        )

        print(
            "Risk Level:",
            transaction["risk_level"]
        )

        if transaction["risk_level"] == "HIGH RISK":

            print(
                "ALERT: Fraud Investigation Required"
            )

        time.sleep(0.2)


streaming_fraud_detection(
    fraud_data
)


# ============================================================
# STEP 6: GRAPH FRAUD DETECTION
# ============================================================

print("\nCreating Fraud Transaction Network...")


fraud_graph = nx.Graph()


for index, row in fraud_data.iterrows():

    fraud_graph.add_edge(

        int(row["sender"]),

        int(row["receiver"]),

        weight=row["amount"],

        risk=row["risk_score"]
    )


print(
    "Number of Accounts:",
    fraud_graph.number_of_nodes()
)

print(
    "Number of Transaction Connections:",
    fraud_graph.number_of_edges()
)


# ============================================================
# STEP 7: DETECT HIGHLY CONNECTED ACCOUNTS
# ============================================================

degree_centrality = nx.degree_centrality(
    fraud_graph
)


suspicious_accounts = sorted(

    degree_centrality.items(),

    key=lambda x: x[1],

    reverse=True

)[:10]


print("\nTop 10 Highly Connected Accounts:")

for account, score in suspicious_accounts:

    print(
        "Account:",
        account,
        "Centrality Score:",
        round(score, 4)
    )


# ============================================================
# STEP 8: FRAUD NETWORK VISUALIZATION
# ============================================================

plt.figure(
    figsize=(12, 8)
)


sample_graph_nodes = list(
    fraud_graph.nodes()
)[:50]


sample_graph = fraud_graph.subgraph(
    sample_graph_nodes
)


nx.draw(

    sample_graph,

    with_labels=True,

    node_size=500,

    font_size=8
)


plt.title(
    "Financial Fraud Transaction Network"
)

plt.show()


# ============================================================
# STEP 9: FRAUD RISK DASHBOARD
# ============================================================

risk_counts = fraud_data[
    "risk_level"
].value_counts()


plt.figure(
    figsize=(8, 5)
)


risk_counts.plot(
    kind="bar"
)


plt.title(
    "Financial Fraud Risk Dashboard"
)

plt.xlabel(
    "Risk Level"
)

plt.ylabel(
    "Number of Transactions"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 10: TRANSACTION AMOUNT DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 5)
)


plt.hist(

    fraud_data["amount"],

    bins=50
)


plt.title(
    "Transaction Amount Distribution"
)

plt.xlabel(
    "Transaction Amount"
)

plt.ylabel(
    "Frequency"
)

plt.show()


# ============================================================
# STEP 11: RESOLUTION AND PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT PERFORMANCE REPORT")
print("=" * 70)


print("\nCUSTOMER SUPPORT COPILOT")

print(
    "Intent Detection Accuracy:",
    round(
        intent_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Total Customer Tickets:",
    resolution_metrics["total_tickets"]
)

print(
    "Automatically Resolved:",
    resolution_metrics["automatically_resolved"]
)

print(
    "Escalated Tickets:",
    resolution_metrics["escalated"]
)


print("\nML DEPLOYMENT PLATFORM")

print(
    "ML Model Accuracy:",
    round(
        ml_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Data Drift Status:",
    drift_status
)


print("\nFINANCIAL FRAUD INTELLIGENCE ENGINE")

print(
    "Total Transactions:",
    len(fraud_data)
)

print(
    "Suspicious Transactions:",
    fraud_data["is_suspicious"].sum()
)

print(
    "High Risk Transactions:",
    (
        fraud_data["risk_level"]
        ==
        "HIGH RISK"
    ).sum()
)


# ============================================================
# STEP 12: SAVE PROJECT RESULTS
# ============================================================

support_data.to_csv(
    "customer_support_data.csv",
    index=False
)


ml_data.to_csv(
    "ml_pipeline_data.csv",
    index=False
)


fraud_data.to_csv(
    "fraud_detection_results.csv",
    index=False
)


feedback_dataframe = pd.DataFrame(
    feedback_data
)


feedback_dataframe.to_csv(
    "customer_feedback.csv",
    index=False
)


print("\nProject Results Saved Successfully.")


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "ALL THREE AI/ML PROJECT MODULES COMPLETED SUCCESSFULLY"
)

print("=" * 70)


print("""
IMPLEMENTED FEATURES:

AUTONOMOUS CUSTOMER SUPPORT COPILOT
-----------------------------------
✓ NLP
✓ Intent Detection
✓ Intent Routing
✓ RAG
✓ Company Knowledge Base
✓ LLM Response Simulation
✓ Automatic Escalation
✓ Feedback Learning Loop
✓ Resolution Metrics


END-TO-END ML DEPLOYMENT PLATFORM
---------------------------------
✓ Data Pipeline
✓ Airflow Pipeline Simulation
✓ Model Training
✓ Automatic Deployment
✓ CI/CD Pipeline
✓ Model Monitoring
✓ Data Drift Detection
✓ Automatic Retraining


FINANCIAL FRAUD INTELLIGENCE ENGINE
-----------------------------------
✓ Streaming ML Simulation
✓ Anomaly Detection
✓ Fraud Risk Scoring
✓ Graph Fraud Detection
✓ Fraud Network Visualization
✓ Fraud Dashboard
✓ Real-Time Fraud Alerts


PROJECT EXECUTION COMPLETED.
""")