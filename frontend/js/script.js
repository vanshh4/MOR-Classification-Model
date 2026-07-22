/*
  This file controls the interactive developer workflow panel.

  When the developer clicks on a pipeline step, the right-side panel updates
  with the purpose and tasks related to that stage of the MOR ML pipeline.
*/


// Data for each workflow step
const workflowDetails = {
  rulebook: {
    title: "Rulebook Conversion",
    description:
      "The MOR rulebook PDF is converted into a structured Excel file where each reportable condition is stored as a separate rule.",
    tasks: [
      "Extract MOR conditions from the PDF rulebook",
      "Create a unique rule_id for every condition",
      "Store domain, category, sub-category and keywords",
      "Use this file as a rule-based reference during model development"
    ]
  },

  "data-cleaning": {
    title: "Data Cleaning",
    description:
      "Historical safety reports are cleaned and standardized before they are used for training the ML model.",
    tasks: [
      "Remove duplicate or incomplete reports",
      "Clean unnecessary symbols, extra spaces and email-like text",
      "Normalize aviation abbreviations such as ACFT, RWY, TWY, APU and GPU",
      "Standardize labels such as Bird Hit, Birdstrike and Wildlife Strike"
    ]
  },

  split: {
    title: "Training and Testing Split",
    description:
      "The cleaned dataset is divided into training and testing data so that the model can be trained on one part and evaluated on unseen data.",
    tasks: [
      "Use stratified train-test split to preserve MOR and Non-MOR ratio",
      "Keep training data for model learning",
      "Keep testing data for unbiased model evaluation",
      "Avoid data leakage between training and testing datasets"
    ]
  },

  training: {
    title: "Model Training",
    description:
      "The machine learning model learns patterns from historical report text and verified MOR labels.",
    tasks: [
      "Convert report text into numerical vectors using TF-IDF or embeddings",
      "Train MOR / Non-MOR classifier",
      "Train category and sub-category classifier for MOR cases",
      "Save trained model files using joblib or pickle"
    ]
  },

  testing: {
    title: "Model Testing",
    description:
      "The trained model is tested on unseen reports to measure whether it can correctly identify MOR cases.",
    tasks: [
      "Generate predictions on testing data",
      "Calculate accuracy, precision, recall and F1-score",
      "Review confusion matrix",
      "Identify categories where the model is weak"
    ]
  },

  prediction: {
    title: "New Report Prediction",
    description:
      "A new safety report written in normal English is passed through preprocessing and the trained model to generate MOR prediction.",
    tasks: [
      "Clean incoming report text",
      "Run MOR classifier",
      "Generate confidence score",
      "Run category classifier if MOR is predicted",
      "Return structured prediction output"
    ]
  },

  feedback: {
    title: "Human Review and Feedback",
    description:
      "Low-confidence or conflicting predictions are sent for review by the Flight Safety team. Their corrections are later used to improve the model.",
    tasks: [
      "Flag low-confidence predictions",
      "Compare ML result with rule-based result",
      "Send uncertain cases for manual review",
      "Store corrected labels for future retraining"
    ]
  }
};


// Select all workflow step cards
const pipelineSteps = document.querySelectorAll(".pipeline-step");

// Select developer panel elements
const panelTitle = document.getElementById("panel-title");
const panelDescription = document.getElementById("panel-description");
const panelTaskList = document.getElementById("panel-task-list");


// Add click event to every workflow step
pipelineSteps.forEach((step) => {
  step.addEventListener("click", () => {
    const selectedStep = step.getAttribute("data-step");
    const selectedData = workflowDetails[selectedStep];

    // Remove active class from all steps
    pipelineSteps.forEach((item) => item.classList.remove("active"));

    // Add active class to clicked step
    step.classList.add("active");

    // Update developer panel title and description
    panelTitle.textContent = selectedData.title;
    panelDescription.textContent = selectedData.description;

    // Clear old task list
    panelTaskList.innerHTML = "";

    // Add new tasks dynamically
    selectedData.tasks.forEach((task) => {
      const li = document.createElement("li");
      li.textContent = task;
      panelTaskList.appendChild(li);
    });
  });
});